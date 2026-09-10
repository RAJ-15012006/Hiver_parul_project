"""
data_pipeline.py
----------------
Loads and preprocesses the Customer Support on Twitter dataset,
filters for AmazonHelp, reconstructs conversation threads, and
exports a clean DataFrame ready for downstream use.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── constants ────────────────────────────────────────────────────────────────
BRAND = "AmazonHelp"
RAW_CSV = Path(__file__).parent.parent.parent / "dataset" / "twcs" / "twcs.csv"
SAMPLE_CSV = Path(__file__).parent.parent.parent / "dataset" / "sample.csv"

# The subset size we train/evaluate on (full ≈ 500k rows; we use a manageable slice)
MAX_ROWS = 500_000   # scan this many rows from the full CSV


# ── helpers ──────────────────────────────────────────────────────────────────
def _strip_mention(text: str) -> str:
    """Remove leading @mentions (Twitter artifact) and clean whitespace."""
    text = re.sub(r"^(@\w+\s*)+", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_english(text: str) -> bool:
    """Very cheap English filter: at least 60 % ASCII chars."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= 0.60


# ── main loader ──────────────────────────────────────────────────────────────
def load_raw(csv_path: Path = RAW_CSV, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Load raw CSV with dtype coercion and row cap."""
    dtype_map = {
        "tweet_id": str,
        "author_id": str,
        "inbound": str,
        "created_at": str,
        "text": str,
        "response_tweet_id": str,
        "in_response_to_tweet_id": str,
    }
    chunks = []
    with pd.read_csv(
        csv_path,
        dtype=dtype_map,
        chunksize=50_000,
        on_bad_lines="skip",
    ) as reader:
        for i, chunk in enumerate(reader):
            chunks.append(chunk)
            if (i + 1) * 50_000 >= max_rows:
                break
    df = pd.concat(chunks, ignore_index=True)
    df["inbound"] = df["inbound"].str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    return df


def filter_amazon(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where either side is AmazonHelp."""
    mask_brand = df["author_id"] == BRAND
    mask_cust  = df["text"].str.contains(f"@{BRAND}", na=False)
    return df[mask_brand | mask_cust].copy()


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Strip mentions, URLs, repeated spaces; drop non-English."""
    df = df.copy()
    df["text_clean"] = df["text"].fillna("").apply(lambda t: re.sub(
        r"http\S+", "[URL]", t
    ))
    df["text_clean"] = df["text_clean"].apply(_strip_mention)
    df = df[df["text_clean"].str.len() > 5].copy()
    return df


def build_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct (customer_msg, brand_reply) pairs using the
    in_response_to_tweet_id / response_tweet_id linkage.

    Returns a DataFrame with columns:
        customer_tweet_id, customer_text, brand_tweet_id, brand_reply
    """
    # Index by tweet_id for O(1) lookup
    df = df.set_index("tweet_id")

    brand_tweets  = df[df["author_id"] == BRAND]
    customer_tweets = df[df["inbound"] == True]  # noqa: E712

    pairs = []
    for tid, row in tqdm(brand_tweets.iterrows(), total=len(brand_tweets),
                         desc="Building conversation pairs"):
        in_resp = row.get("in_response_to_tweet_id")
        if pd.isna(in_resp):
            continue
        in_resp = str(in_resp).strip()
        if in_resp not in df.index:
            continue
        cust = df.loc[in_resp]
        if cust["inbound"] != True:  # noqa: E712
            continue
        cust_text  = cust.get("text_clean", cust.get("text", ""))
        brand_text = row.get("text_clean", row.get("text", ""))
        if not _is_english(cust_text):
            continue
        pairs.append({
            "customer_tweet_id": in_resp,
            "customer_text":     cust_text,
            "brand_tweet_id":    tid,
            "brand_reply":       brand_text,
            "created_at":        row.get("created_at", ""),
        })

    convs = pd.DataFrame(pairs).drop_duplicates(subset="customer_tweet_id")
    return convs


def run(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """End-to-end data pipeline; returns clean conversation pairs."""
    print(f"[data_pipeline] Loading raw CSV from {csv_path} …")
    raw  = load_raw(csv_path)
    print(f"[data_pipeline] Raw rows: {len(raw):,}")
    amz  = filter_amazon(raw)
    print(f"[data_pipeline] Amazon rows: {len(amz):,}")
    amz  = clean_text(amz)
    convs = build_conversations(amz)
    print(f"[data_pipeline] Conversation pairs: {len(convs):,}")
    return convs


if __name__ == "__main__":
    df = run()
    out = Path(__file__).parent.parent / "data" / "conversations.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[data_pipeline] Saved → {out}")
