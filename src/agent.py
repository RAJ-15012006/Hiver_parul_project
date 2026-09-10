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
    model: str = "llama-3.3-70b-versatile",
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


# ── 1. Classify ───────────────────────────────────────────────────────────────
CLASSIFY_SYSTEM = (
    "You are an expert customer-support intent classifier for Amazon. "
    "You must respond with ONLY the intent label — no explanation, no punctuation.\n\n"
    + label_description_block()
    + "\n\n"
    + few_shot_block(n_per_class=2)
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
        # Normalise: strip punctuation, uppercase
        label = re.sub(r"[^A-Z_]", "", raw.strip().upper())
        if label in LABELS:
            return label
        # Fuzzy fallback: check if any label is a substring
        for l in LABELS:
            if l in raw.upper():
                return l
        logger.warning(f"Unrecognised label from LLM: '{raw}' — falling back to keyword")
        return _keyword_classify(text)
    except Exception as e:
        logger.error(f"classify() LLM error: {e} — falling back to keyword")
        return _keyword_classify(text)


# ── 2. Draft Reply ────────────────────────────────────────────────────────────
REPLY_SYSTEM = """\
You are AmazonHelp, Amazon's official customer support agent on Twitter.
Your replies must be:
- Empathetic and professional
- Concise (≤ 280 characters when possible, but correctness > brevity)
- Action-oriented: tell the customer exactly what to do next
- Grounded in the historical examples provided

Use "DM" instead of "Direct Message". Reference order numbers if mentioned.
Do NOT use excessive exclamation marks or robotic filler phrases.
"""


def draft_reply(
    customer_text: str,
    intent: str,
    retrieved: List[Dict[str, Any]],
) -> str:
    """
    Draft a support reply grounded in retrieved historical examples.
    """
    # Build RAG context block
    context_lines = []
    for r in retrieved[:4]:
        context_lines.append(
            f'  [Customer]: "{r["customer_text"]}"\n'
            f'  [AmazonHelp reply]: "{r["brand_reply"]}"'
        )
    context_block = "\n---\n".join(context_lines) if context_lines else "(none)"

    user_prompt = (
        f"Intent: {intent}\n"
        f"Customer message: \"{customer_text}\"\n\n"
        f"Historical AmazonHelp replies to similar issues:\n{context_block}\n\n"
        "Please draft a single, concise reply from AmazonHelp."
    )

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

    # 2. Retrieve similar historical examples
    retrieved = []
    if faiss_index is not None and faiss_meta is not None:
        retrieved = retrieve(customer_text, faiss_index, faiss_meta, k=5)

    # 3. Draft reply
    reply = draft_reply(customer_text, intent, retrieved)

    # 4. Escalation decision
    decision, reason = decide_escalation(customer_text, intent, retrieved, use_llm=use_llm)

    return {
        "customer_text":       customer_text,
        "intent":              intent,
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
