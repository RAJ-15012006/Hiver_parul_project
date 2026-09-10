"""
llm_judge.py
------------
LLM-as-judge for reply quality evaluation.

Rubric (5 dimensions, each scored 1–5):
  1. Helpfulness    — Does it answer the customer's core issue?
  2. Empathy        — Is the tone warm and acknowledging?
  3. Accuracy       — Is the advice correct and actionable?
  4. Conciseness    — Is it appropriately brief (Twitter-context)?
  5. Brand Voice    — Does it sound like professional Amazon support?

Overall score = mean of the 5 dimensions.
Agreement with human judgements is measured via Spearman correlation.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_SYSTEM = """\
You are an expert evaluator of customer support reply quality for Amazon Twitter support.
You will be given:
  - The customer's original message
  - The intent it was classified as
  - The drafted reply from AmazonHelp

Score the reply on each of the following dimensions from 1 (very poor) to 5 (excellent):
  1. Helpfulness: Does the reply address the customer's core problem?
  2. Empathy: Is the tone warm, understanding, and non-robotic?
  3. Accuracy: Is the advice correct and actionable?
  4. Conciseness: Is the reply appropriately brief for Twitter context?
  5. BrandVoice: Does it sound like professional Amazon support?

Respond ONLY in valid JSON with this exact structure:
{
  "helpfulness": <int 1-5>,
  "empathy": <int 1-5>,
  "accuracy": <int 1-5>,
  "conciseness": <int 1-5>,
  "brand_voice": <int 1-5>,
  "overall": <float, mean of above>,
  "explanation": "<one sentence>"
}
"""


_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def judge_reply(
    customer_text: str,
    intent: str,
    draft_reply: str,
    retries: int = 3,
) -> Dict[str, float | str]:
    """
    Score one reply. Returns a dict with dimension scores and explanation.
    Falls back to neutral scores (3.0) on failure.
    """
    user_prompt = (
        f"Customer message: \"{customer_text}\"\n"
        f"Intent: {intent}\n"
        f"AmazonHelp reply: \"{draft_reply}\"\n\n"
        "Score this reply."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    client = _get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON even if surrounded by markdown
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                # Ensure overall is computed correctly
                dims = ["helpfulness", "empathy", "accuracy", "conciseness", "brand_voice"]
                scores["overall"] = round(
                    sum(scores.get(d, 3) for d in dims) / len(dims), 2
                )
                return scores
        except Exception as e:
            logger.warning(f"judge_reply attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    # Fallback neutral scores
    return {
        "helpfulness": 3.0, "empathy": 3.0, "accuracy": 3.0,
        "conciseness": 3.0, "brand_voice": 3.0,
        "overall": 3.0, "explanation": "Scoring failed (API error)"
    }


def batch_judge(
    rows: List[Dict],
    max_rows: Optional[int] = None,
    sleep_between: float = 0.5,
) -> List[Dict]:
    """
    Judge a batch of agent outputs.
    Each row must have: customer_text, intent, draft_reply
    Returns rows enriched with judge scores.
    """
    results = []
    n = min(len(rows), max_rows) if max_rows else len(rows)
    for i, row in enumerate(rows[:n]):
        logger.info(f"[judge] Scoring {i+1}/{n} …")
        scores = judge_reply(
            row["customer_text"],
            row["intent"],
            row["draft_reply"],
        )
        results.append({**row, **{f"judge_{k}": v for k, v in scores.items()}})
        time.sleep(sleep_between)
    return results


def human_agreement_analysis(
    judge_scores: List[float],
    human_scores: List[float],
) -> Dict[str, float]:
    """
    Measure how well the LLM judge agrees with human ratings.
    Returns Pearson r, Spearman rho, and MAE.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    j = np.array(judge_scores)
    h = np.array(human_scores)
    pearson_r, _ = pearsonr(j, h)
    spearman_rho, _ = spearmanr(j, h)
    mae = float(np.mean(np.abs(j - h)))
    return {
        "pearson_r":    round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "mae":          round(mae, 4),
        "n":            len(j),
    }


if __name__ == "__main__":
    # Smoke test
    score = judge_reply(
        customer_text="My order hasn't arrived and it's been a week!",
        intent="ORDER_STATUS",
        draft_reply=(
            "Hi! We're sorry for the delay. Please DM us your order number "
            "and we'll look into this right away."
        ),
    )
    print(json.dumps(score, indent=2))
