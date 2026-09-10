"""
build_golden_set.py
--------------------
Semi-automated golden evaluation set construction.

Strategy:
  1. Load all AmazonHelp conversation pairs
  2. Stratified sample ~25-30 examples per intent class (= ~200 total)
  3. Use keyword seeds for initial labelling
  4. Apply LLM to verify/correct labels and set escalation ground truth
  5. Save as data/golden_eval.csv with columns:
       customer_text, intent, escalation, brand_reply, human_score, notes

Run this script once to produce the golden set.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from intent_taxonomy import INTENTS, LABELS, INTENT_BY_LABEL, label_description_block

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent / "data"
GOLDEN_CSV  = DATA_DIR / "golden_eval.csv"
CONVS_CSV   = DATA_DIR / "conversations.csv"
TARGET_PER_CLASS = 28      # ~224 total across 8 classes
MIN_TEXT_LEN = 20

ESCALATE_INTENTS = {"ACCOUNT_ACCESS", "BILLING_CHARGE"}


# ── Keyword-based initial labeller ────────────────────────────────────────────
def keyword_label(text: str) -> str:
    text_lower = text.lower()
    best, best_cnt = "GENERAL_INQUIRY", 0
    for intent in INTENTS:
        cnt = sum(1 for kw in intent.keywords if kw in text_lower)
        if cnt > best_cnt:
            best_cnt, best = cnt, intent.label
    return best


# ── LLM verification (batch) ──────────────────────────────────────────────────
_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


VERIFY_SYSTEM = (
    "You are an expert Amazon customer support analyst. "
    "You will classify customer messages into intents and decide escalation.\n\n"
    + label_description_block()
    + "\n\nFor each message respond in JSON: "
    '{"intent": "LABEL", "escalation": "AUTO"|"ESCALATE", "reason": "one sentence"}\n'
    "Respond with JSON only, no extra text."
)


def llm_verify(text: str, retries: int = 3) -> dict:
    """Verify/correct label via LLM."""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": f'Customer message: "{text}"'},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                d = json.loads(m.group())
                label = d.get("intent", "GENERAL_INQUIRY").strip().upper()
                if label not in LABELS:
                    label = "GENERAL_INQUIRY"
                esc = d.get("escalation", "AUTO").strip().upper()
                if esc not in ("AUTO", "ESCALATE"):
                    esc = "AUTO"
                return {"intent": label, "escalation": esc, "reason": d.get("reason", "")}
        except Exception as e:
            logger.warning(f"LLM verify attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return {"intent": keyword_label(text), "escalation": "AUTO", "reason": "fallback"}


# ── Main builder ──────────────────────────────────────────────────────────────
def build_golden_set(
    convs_csv: Path = CONVS_CSV,
    target_per_class: int = TARGET_PER_CLASS,
    use_llm_verify: bool = True,
) -> pd.DataFrame:
    logger.info(f"Loading conversations from {convs_csv} …")
    convs = pd.read_csv(convs_csv)
    logger.info(f"Total pairs: {len(convs):,}")

    # Basic quality filter
    convs = convs[convs["customer_text"].str.len() >= MIN_TEXT_LEN].copy()
    convs = convs.drop_duplicates(subset="customer_text").reset_index(drop=True)

    # Keyword-based initial labelling
    logger.info("Applying keyword labels …")
    convs["intent_kw"] = convs["customer_text"].apply(keyword_label)

    # Stratified sample per class
    samples = []
    for label in LABELS:
        subset = convs[convs["intent_kw"] == label]
        n = min(len(subset), target_per_class * 4)  # oversample then LLM-trim
        if n == 0:
            logger.warning(f"No examples for {label} in dataset!")
            continue
        sampled = subset.sample(n=n, random_state=42)
        samples.append(sampled)

    pool = pd.concat(samples, ignore_index=True)
    logger.info(f"Pool before LLM verification: {len(pool):,}")

    # LLM verification
    final_rows = []
    label_counts = {l: 0 for l in LABELS}

    for idx, row in pool.iterrows():
        cust = row["customer_text"]
        if use_llm_verify:
            result = llm_verify(cust)
            time.sleep(0.3)   # gentle rate-limiting
        else:
            kw_l = row["intent_kw"]
            esc = "ESCALATE" if kw_l in ESCALATE_INTENTS else "AUTO"
            result = {"intent": kw_l, "escalation": esc, "reason": "keyword"}

        label = result["intent"]
        if label_counts[label] >= target_per_class:
            continue

        label_counts[label] += 1
        final_rows.append({
            "customer_text":  cust,
            "intent":         label,
            "escalation":     result["escalation"],
            "brand_reply":    row.get("brand_reply", ""),
            "human_score":    None,   # to be filled manually or by judge
            "notes":          result.get("reason", ""),
        })

        total = sum(label_counts.values())
        if total % 50 == 0:
            logger.info(f"  Progress: {total} examples — {label_counts}")
        if all(c >= target_per_class for c in label_counts.values()):
            break

    golden = pd.DataFrame(final_rows)
    logger.info(f"\nGolden set distribution:\n{golden['intent'].value_counts().to_string()}")
    logger.info(f"\nEscalation distribution:\n{golden['escalation'].value_counts().to_string()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden.to_csv(GOLDEN_CSV, index=False)
    logger.info(f"\nGolden eval set saved → {GOLDEN_CSV}  ({len(golden):,} rows)")
    return golden


if __name__ == "__main__":
    build_golden_set(use_llm_verify=True)
