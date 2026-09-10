"""
eda.py
------
Exploratory Data Analysis on the AmazonHelp dataset.
Generates summary statistics and plots saved to outputs/.

Run: python eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid", palette="Set2")

CONVS_CSV = Path(__file__).parent / "data" / "conversations.csv"
OUTPUTS   = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def run_eda():
    print("Loading conversation pairs …")
    convs = pd.read_csv(CONVS_CSV)
    print(f"  Total pairs: {len(convs):,}")

    # ── 1. Text length distributions ──────────────────────────────────────────
    convs["cust_len"]  = convs["customer_text"].str.len()
    convs["reply_len"] = convs["brand_reply"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(convs["cust_len"].clip(upper=280), bins=40, color="#2196F3", alpha=0.8)
    axes[0].set_title("Customer Message Length (chars)")
    axes[0].set_xlabel("Characters")
    axes[1].hist(convs["reply_len"].clip(upper=280), bins=40, color="#4CAF50", alpha=0.8)
    axes[1].set_title("AmazonHelp Reply Length (chars)")
    axes[1].set_xlabel("Characters")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "text_length_dist.png", dpi=150)
    plt.close()
    print("  Saved: outputs/text_length_dist.png")

    # ── 2. Keyword-based intent distribution ──────────────────────────────────
    from intent_taxonomy import INTENTS
    def keyword_label(text):
        text_lower = str(text).lower()
        best, best_cnt = "GENERAL_INQUIRY", 0
        for intent in INTENTS:
            cnt = sum(1 for kw in intent.keywords if kw in text_lower)
            if cnt > best_cnt:
                best_cnt, best = cnt, intent.label
        return best

    print("  Applying keyword labels for EDA …")
    convs["intent_kw"] = convs["customer_text"].apply(keyword_label)
    intent_counts = convs["intent_kw"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        intent_counts.index, intent_counts.values,
        color=sns.color_palette("Set2", len(intent_counts))
    )
    ax.set_title("Keyword-labelled Intent Distribution (AmazonHelp corpus)", fontsize=14)
    ax.set_xlabel("Number of conversations")
    for bar, val in zip(bars, intent_counts.values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                f"{val:,} ({val/len(convs)*100:.1f}%)",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "intent_distribution.png", dpi=150)
    plt.close()
    print("  Saved: outputs/intent_distribution.png")

    # ── 3. Top word frequencies per intent ────────────────────────────────────
    from collections import Counter
    import re

    STOPWORDS = {"i", "my", "the", "a", "an", "and", "or", "to", "of", "in",
                 "is", "it", "you", "me", "we", "have", "are", "for", "on",
                 "at", "with", "was", "he", "she", "be", "this", "that",
                 "not", "from", "can", "do", "get", "got", "has", "had",
                 "but", "as", "so", "if", "been", "its", "they", "there",
                 "com", "https", "url", "please", "amazon", "amazonhelp"}

    top_words = {}
    for label in convs["intent_kw"].unique():
        texts = convs[convs["intent_kw"] == label]["customer_text"]
        words = []
        for t in texts:
            words.extend(re.findall(r"\b[a-z]{3,}\b", str(t).lower()))
        filtered = [w for w in words if w not in STOPWORDS]
        top_words[label] = Counter(filtered).most_common(8)

    print("\n  Top words per intent (after stopwords):")
    for label, words in top_words.items():
        print(f"    {label}: {[w for w,_ in words]}")

    # ── 4. Summary stats ──────────────────────────────────────────────────────
    print("\n  Customer message length stats:")
    print(f"    Mean:   {convs['cust_len'].mean():.1f} chars")
    print(f"    Median: {convs['cust_len'].median():.1f} chars")
    print(f"    Max:    {convs['cust_len'].max()} chars")
    print(f"    >280:   {(convs['cust_len']>280).sum()} ({(convs['cust_len']>280).mean()*100:.1f}%)")

    print("\n  AmazonHelp reply length stats:")
    print(f"    Mean:   {convs['reply_len'].mean():.1f} chars")
    print(f"    Median: {convs['reply_len'].median():.1f} chars")

    print("\n  Intent distribution:")
    for label, cnt in intent_counts.items():
        print(f"    {label:<25}: {cnt:>5} ({cnt/len(convs)*100:>5.1f}%)")

    print("\n✅  EDA complete. Plots saved to outputs/")


if __name__ == "__main__":
    run_eda()
