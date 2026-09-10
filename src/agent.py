"""
agent.py
--------
The main AI support agent for AmazonHelp.

Three capabilities:
  1. classify(text)    → predicted intent label
  2. reply(text)       → drafted reply grounded in historical data (RAG)
  3. decide(text, intent) → ("AUTO" | "ESCALATE", reason)

Uses Groq API (llama-3.3-70b-versatile) for LLM calls.
Falls back to keyword heuristics for robustness if API fails.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import json
import time
import logging
from typing import Optional, Tuple, List, Dict, Any

from groq import Groq
from dotenv import load_dotenv

from intent_taxonomy import (
    INTENTS,
    LABELS,
    INTENT_BY_LABEL,
    label_description_block,
    few_shot_block,
)
from embeddings import retrieve

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Groq client ───────────────────────────────────────────────────────────────
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set. Add it to .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _call_groq(
    messages: List[Dict[str, str]],
    model: str = "qwen/qwen3.8-27b",
    temperature: float = 0.2,
    max_tokens: int = 512,
    retries: int = 3,
) -> str:
    """Call Groq with retry on rate-limit."""
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq call failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("All Groq retries exhausted.")


# ── Keyword fallback classifier ───────────────────────────────────────────────
def _keyword_classify(text: str) -> str:
    """Fast keyword-match fallback classifier."""
    text_lower = text.lower()
    best_label = "GENERAL_INQUIRY"
    best_count = 0
    for intent in INTENTS:
        count = sum(1 for kw in intent.keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_label = intent.label
    return best_label


# ── Entity Slot Extractor & Guardrail ──────────────────────────────────────────
def extract_slots(text: str) -> Dict[str, Any]:
    """
    Extract operational slots to prevent repetitive or misdirected support prompts.
    Solves Failure Mode #1 (Order ID Fixation).
    """
    text_lower = text.lower()
    
    # 1. Order ID pattern (e.g. 112-9876543-1234567 or #123456)
    order_match = re.search(r"\b(\d{3}-\d{7}-\d{7}|#\d{5,10})\b", text)
    order_id = order_match.group(1) if order_match else None
    
    # 2. Tracking ID pattern (e.g. TBA..., 1Z..., or 10-22 digit tracking numbers)
    tracking_match = re.search(r"\b(TBA\d{10,14}|1Z[0-9A-Z]{16}|\d{12,22})\b", text, re.IGNORECASE)
    has_tracking_kw = any(w in text_lower for w in ["tracking id", "tracking number", "tracking #", "track id"])
    tracking_id = tracking_match.group(1) if tracking_match else ("provided" if has_tracking_kw else None)
    
    # 3. Currency amounts
    amount_match = re.search(r"([$£€]\s*\d+(\.\d{2})?|\b\d+\s*(dollars|pounds|euros))", text_lower)
    amount = amount_match.group(1) if amount_match else None

    return {
        "order_id": order_id,
        "tracking_id": tracking_id,
        "amount": amount,
    }


def sanitize_retrieved_reply(text: str) -> str:
    """
    Strip historical monetary compensation promises to prevent RAG context leakage.
    Solves Failure Mode #3 (Hallucinated Resolution Artifacts).
    """
    # Neutralize specific dollar credits/gift card claims from historical corpus
    sanitized = re.sub(
        r"[\$£€]\d+(\.\d{2})?\s*(promotional\s*certificate|credit|gift\s*card|refund)?",
        "appropriate resolution",
        text,
        flags=re.IGNORECASE,
    )
    return sanitized.strip()


# ── 1. Classify ───────────────────────────────────────────────────────────────
CLASSIFY_SYSTEM = (
    "You are an expert Amazon customer support intent classifier.\n"
    "Classify the customer message into EXACTLY ONE of these labels:\n"
    "ORDER_STATUS, REFUND_RETURN, PRODUCT_ISSUE, ACCOUNT_ACCESS, "
    "DELIVERY_PROBLEM, BILLING_CHARGE, PRIME_MEMBERSHIP, GENERAL_INQUIRY.\n\n"
    "Disambiguation Rules:\n"
    "- If an item is physically broken, damaged, cracked, defective, or not working -> PRODUCT_ISSUE (even if return/refund is requested).\n"
    "- If package is marked delivered but not on porch, stolen, or handed to wrong house -> DELIVERY_PROBLEM.\n"
    "- Where is my order, shipping delay, tracking query, or expected arrival -> ORDER_STATUS.\n"
    "- Login loop, password reset, 2FA, OTP, hacked, locked account -> ACCOUNT_ACCESS.\n"
    "- Unwanted charge, duplicate debit, unknown fee, billing error -> BILLING_CHARGE.\n"
    "- Return policy, return label, sending undamaged item back -> REFUND_RETURN.\n"
    "- Prime membership fee, Prime video, cancellation -> PRIME_MEMBERSHIP.\n"
    "- General inquiries, packaging, app bugs, search -> GENERAL_INQUIRY.\n\n"
    "Respond with ONLY the exact uppercase label."
)


def classify(
    text: str,
    use_llm: bool = True,
) -> str:
    """
    Classify a customer tweet into one of the 8 intents.
    Returns: label string (e.g. "ORDER_STATUS")
    """
    if not use_llm:
        return _keyword_classify(text)

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": f'Customer message: "{text}"\nIntent:'},
    ]
    try:
        raw = _call_groq(messages, temperature=0.0, max_tokens=20)
        label = re.sub(r"[^A-Z_]", "", raw.strip().upper())
        if label in LABELS:
            return label
        for l in LABELS:
            if l in raw.upper():
                return l
        logger.warning(f"Unrecognised label from LLM: '{raw}' — falling back to keyword")
        return _keyword_classify(text)
    except Exception as e:
        logger.error(f"classify() LLM error: {e} — falling back to keyword")
        return _keyword_classify(text)


# ── 2. Draft Reply ────────────────────────────────────────────────────────────
REPLY_SYSTEM = (
    "You are AmazonHelp on Twitter. Write a concise, empathetic, professional "
    "support reply (<= 240 characters) offering clear next steps. "
    "Use 'DM', ground in past Amazon responses. Never invent monetary compensation."
)


def draft_reply(
    customer_text: str,
    intent: str,
    retrieved: List[Dict[str, Any]],
    slots: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Draft a support reply grounded in retrieved historical examples and constrained by slots.
    """
    if slots is None:
        slots = extract_slots(customer_text)

    # Build slot guardrail instructions
    guardrails = []
    if slots.get("tracking_id"):
        guardrails.append("- Tracking ID is ALREADY provided. Do NOT ask for tracking number again.")
    if intent == "ACCOUNT_ACCESS":
        guardrails.append("- Account security issue. Do NOT ask for an Order ID. Direct user to secure account help: [URL].")
    if slots.get("order_id"):
        guardrails.append(f"- Order ID {slots['order_id']} is already provided. Acknowledge and ask them to DM details.")

    guardrail_block = "\n".join(guardrails) if guardrails else ""

    # Compact RAG context block (sanitized)
    context_lines = []
    for r in retrieved[:2]:
        clean_past = sanitize_retrieved_reply(r["brand_reply"][:150])
        context_lines.append(f'- Past reply: "{clean_past}"')
    context_block = "\n".join(context_lines) if context_lines else "(none)"

    user_prompt = (
        f"Intent: {intent}\n"
        f"Customer: \"{customer_text}\"\n"
        f"Grounding:\n{context_block}\n"
    )
    if guardrail_block:
        user_prompt += f"Guardrails to follow:\n{guardrail_block}\n"
    user_prompt += "Draft AmazonHelp reply:"

    messages = [
        {"role": "system", "content": REPLY_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return _call_groq(messages, temperature=0.4, max_tokens=150)
    except Exception as e:
        logger.error(f"draft_reply() LLM error: {e}")
        # Fallback: return best historical reply
        if retrieved:
            return retrieved[0]["brand_reply"]
        return (
            "We're sorry for the trouble! Please DM us with your order details "
            "and we'll get this sorted right away."
        )


# ── 3. Escalation Decision ────────────────────────────────────────────────────
# Rules that always escalate regardless of LLM decision
ESCALATE_SIGNALS = [
    r"\bescalat\b",
    r"\b(lawyer|legal|court|sue|lawsuit)\b",
    r"\b(fraud|scam|stolen|hack(ed)?)\b",
    r"\b(refund|money|charge).{0,40}(never|not|didn.?t|no)\b",
    r"\bfbi\b",
    r"\b(media|news|tweet|post|review).{0,30}(about|against)\b",
]
ESCALATE_INTENTS = {"ACCOUNT_ACCESS", "BILLING_CHARGE"}  # high-stakes intents


def decide_escalation(
    customer_text: str,
    intent: str,
    retrieved: List[Dict[str, Any]],
    use_llm: bool = True,
) -> Tuple[str, str]:
    """
    Decide whether to AUTO-handle or ESCALATE.
    Returns: (decision, reason)  where decision ∈ {"AUTO", "ESCALATE"}
    """
    text_lower = customer_text.lower()

    # ── Rule-based hard escalation signals ────────────────────────────────────
    for pattern in ESCALATE_SIGNALS:
        if re.search(pattern, text_lower):
            return "ESCALATE", f"Hard escalation rule matched: '{pattern}'"

    # ── High-stakes intents → LLM decides ─────────────────────────────────────
    if intent in ESCALATE_INTENTS and use_llm:
        prompt = (
            f"Intent: {intent}\n"
            f"Customer message: \"{customer_text}\"\n\n"
            "Should this message be handled automatically (AUTO) or escalated to a "
            "human agent (ESCALATE)? Consider account/billing sensitivity.\n"
            "Reply with exactly one word: AUTO or ESCALATE, then a dash, then a "
            "one-sentence reason.\nFormat: DECISION - reason"
        )
        messages = [
            {"role": "system", "content": "You are a triage specialist for AmazonHelp."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = _call_groq(messages, temperature=0.0, max_tokens=80)
            m = re.match(r"(AUTO|ESCALATE)\s*[-–—]\s*(.+)", raw.strip(), re.IGNORECASE)
            if m:
                dec = m.group(1).upper()
                reason = m.group(2).strip()
                return dec, reason
        except Exception as e:
            logger.error(f"decide_escalation() LLM error: {e}")

    if intent in ESCALATE_INTENTS:
        return "ESCALATE", f"Sensitive {intent} query requires human verification of credentials/billing."

    # ── Default: low-stakes → AUTO ─────────────────────────────────────────────
    return "AUTO", f"Standard {intent} query — can be handled by automated response."


# ── Full agent pipeline ────────────────────────────────────────────────────────
def run_agent(
    customer_text: str,
    faiss_index=None,
    faiss_meta: Optional[dict] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: classify → retrieve → reply → decide.

    Returns a dict with keys:
        intent, draft_reply, escalation_decision, escalation_reason, retrieved
    """
    # 1. Classify
    intent = classify(customer_text, use_llm=use_llm)

    # 2. Extract operational slots & guardrails
    slots = extract_slots(customer_text)

    # 3. Retrieve similar historical examples
    retrieved = []
    if faiss_index is not None and faiss_meta is not None:
        retrieved = retrieve(customer_text, faiss_index, faiss_meta, k=5)

    # 4. Draft reply (slot-constrained)
    reply = draft_reply(customer_text, intent, retrieved, slots=slots)

    # 5. Escalation decision
    decision, reason = decide_escalation(customer_text, intent, retrieved, use_llm=use_llm)

    return {
        "customer_text":       customer_text,
        "intent":              intent,
        "slots":               slots,
        "draft_reply":         reply,
        "escalation_decision": decision,
        "escalation_reason":   reason,
        "retrieved":           retrieved[:3],   # top-3 for inspection
    }


if __name__ == "__main__":
    # Quick smoke test (no FAISS needed)
    from dotenv import load_dotenv
    load_dotenv()

    test_msgs = [
        "My order from last week still hasn't shipped! What's going on?",
        "I want to return this broken headphone I received yesterday.",
        "Someone unauthorized accessed my Amazon account!!",
        "I was charged twice for the same item. Please refund one.",
        "How do I cancel Amazon Prime?",
    ]
    for msg in test_msgs:
        result = run_agent(msg, use_llm=True)
        print(f"\n{'='*60}")
        print(f"Customer: {msg}")
        print(f"Intent:   {result['intent']}")
        print(f"Reply:    {result['draft_reply']}")
        print(f"Decision: {result['escalation_decision']} — {result['escalation_reason']}")
