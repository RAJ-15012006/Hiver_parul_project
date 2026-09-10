"""
run_eval.py
-----------
Standalone evaluation script with robust rate limiting.
Runs classification + escalation + BLEU/ROUGE + LLM judge on golden set.
Saves results to outputs/evaluation_report.json
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_CSV = Path(__file__).parent / "data" / "golden_eval.csv"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

LABELS = [
    "ORDER_STATUS", "REFUND_RETURN", "PRODUCT_ISSUE", "ACCOUNT_ACCESS",
    "DELIVERY_PROBLEM", "BILLING_CHARGE", "PRIME_MEMBERSHIP", "GENERAL_INQUIRY",
]


def run_eval():
    from agent import run_agent, classify, _keyword_classify
    from embeddings import load_index
    from baselines import MajorityClassBaseline, TFIDFLogRegBaseline, TFIDFSVMBaseline
    from llm_judge import judge_reply

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        import nltk
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        smoothie = SmoothingFunction().method1
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        have_metrics = True
    except Exception as e:
        logger.warning(f"BLEU/ROUGE unavailable: {e}")
        have_metrics = False

    logger.info("Loading golden eval set …")
    golden = pd.read_csv(GOLDEN_CSV)
    logger.info(f"  {len(golden)} rows loaded")

    # Stratified 4-per-class sample = 32 rows total
    per_class = 4
    subset_idx = []
    for _, g in golden.groupby("intent"):
        subset_idx.extend(g.sample(n=min(len(g), per_class), random_state=42).index)
    subset = golden.loc[subset_idx].reset_index(drop=True)
    logger.info(f"  Evaluation subset: {len(subset)} rows ({per_class} per class)")

    # ── Load FAISS ─────────────────────────────────────────────────────────────
    try:
        faiss_index, faiss_meta = load_index()
        logger.info("  FAISS index loaded ✓")
    except Exception:
        faiss_index, faiss_meta = None, None
        logger.warning("  FAISS not found — running without RAG")

    # ── Run agent ──────────────────────────────────────────────────────────────
    logger.info("Running agent on evaluation subset …")
    results = []
    for i, (_, row) in enumerate(subset.iterrows()):
        res = run_agent(
            row["customer_text"],
            faiss_index=faiss_index,
            faiss_meta=faiss_meta,
            use_llm=True,
        )
        res["true_intent"] = row.get("intent", "")
        res["true_escalation"] = row.get("escalation", "AUTO")
        res["brand_reply"] = row.get("brand_reply", "")
        res["human_score"] = row.get("human_score", None)
        results.append(res)
        if (i + 1) % 5 == 0:
            logger.info(f"  {i+1}/{len(subset)} done …")
        time.sleep(0.8)  # ~75 requests/min

    # ── Classification metrics ─────────────────────────────────────────────────
    y_true = [r["true_intent"] for r in results]
    y_pred = [r["intent"] for r in results]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    clf_report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

    # ── Confusion Matrix Heatmap ──────────────────────────────────────────────
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[l.replace("_", "\n") for l in LABELS],
            yticklabels=LABELS,
            cbar=True,
        )
        plt.title("Confusion Matrix — AmazonHelp Intent Classification", fontsize=13, pad=12)
        plt.xlabel("Predicted Intent", fontsize=11)
        plt.ylabel("True Intent (Golden Set)", fontsize=11)
        plt.tight_layout()
        cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        logger.info(f"Confusion matrix plot saved → {cm_path}")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

    logger.info(f"Agent  →  accuracy={acc:.3f}  macro_F1={macro_f1:.3f}  weighted_F1={weighted_f1:.3f}")

    # ── Escalation metrics ─────────────────────────────────────────────────────
    y_esc_true = [r["true_escalation"] for r in results]
    y_esc_pred = [r["escalation_decision"] for r in results]

    esc_acc = accuracy_score(y_esc_true, y_esc_pred)
    esc_f1 = f1_score(y_esc_true, y_esc_pred, pos_label="ESCALATE",
                      average="binary", zero_division=0)
    logger.info(f"Escal  →  accuracy={esc_acc:.3f}  escalate_F1={esc_f1:.3f}")

    # ── BLEU / ROUGE ───────────────────────────────────────────────────────────
    bleus, rouges = [], []
    if have_metrics:
        for r in results:
            hyp = str(r.get("draft_reply", ""))
            ref = str(r.get("brand_reply", ""))
            if ref:
                tok_hyp = hyp.lower().split()
                tok_ref = [ref.lower().split()]
                bleus.append(sentence_bleu(tok_ref, tok_hyp, smoothing_function=smoothie))
                rouges.append(rouge.score(ref, hyp)["rougeL"].fmeasure)

    bleu_mean = float(np.mean(bleus)) if bleus else 0.0
    rouge_mean = float(np.mean(rouges)) if rouges else 0.0
    logger.info(f"Reply  →  BLEU={bleu_mean:.3f}  ROUGE-L={rouge_mean:.3f}")

    # ── LLM Judge on 8-sample (1 per class) ───────────────────────────────────
    logger.info("Running LLM judge on 8-sample …")
    judge_idx = []
    for _, g in subset.groupby("intent"):
        judge_idx.extend(g.sample(n=1, random_state=42).index)
    judge_sample = subset.loc[judge_idx].reset_index(drop=True)
    judge_scores = []
    for _, jrow in judge_sample.iterrows():
        # Find corresponding prediction
        matching = [r for r in results if r["true_intent"] == jrow["intent"]]
        if not matching:
            continue
        r = matching[0]
        try:
            score = judge_reply(
                customer_text=jrow["customer_text"],
                intent=jrow["intent"],
                draft_reply=r.get("draft_reply", ""),
            )
            judge_scores.append(score)
        except Exception as e:
            logger.warning(f"Judge error: {e}")
        time.sleep(1.2)

    judge_mean = float(np.mean([s.get("overall", 3.0) for s in judge_scores])) if judge_scores else None
    if judge_mean:
        logger.info(f"Judge  →  mean_overall={judge_mean:.2f}/5")

    # ── Baselines on FULL golden set ───────────────────────────────────────────
    logger.info("Training and evaluating baselines …")
    X = golden["customer_text"].tolist()
    y = golden["intent"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    baselines_out = {}
    for name, Cls in [("majority_class", MajorityClassBaseline),
                      ("tfidf_logreg", TFIDFLogRegBaseline),
                      ("tfidf_svm", TFIDFSVMBaseline)]:
        try:
            m = Cls()
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            ba = accuracy_score(y_te, preds)
            bf1 = f1_score(y_te, preds, average="macro", zero_division=0)
            bwf1 = f1_score(y_te, preds, average="weighted", zero_division=0)
            baselines_out[name] = {"accuracy": round(ba, 4), "macro_f1": round(bf1, 4), "weighted_f1": round(bwf1, 4)}
            logger.info(f"  {name:<20} acc={ba:.3f}  macro_F1={bf1:.3f}")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # ── Human score correlation ────────────────────────────────────────────────
    human_scores = [r.get("human_score") for r in results if r.get("human_score") is not None]
    judge_overlaps = [s.get("overall") for s in judge_scores[:len(human_scores)]]
    pearson_r = None
    if human_scores and judge_overlaps and len(human_scores) == len(judge_overlaps):
        try:
            from scipy.stats import pearsonr, spearmanr
            pearson_r = round(float(pearsonr(human_scores[:len(judge_overlaps)], judge_overlaps)[0]), 3)
            spearman_r = round(float(spearmanr(human_scores[:len(judge_overlaps)], judge_overlaps)[0]), 3)
        except Exception:
            spearman_r = None
    else:
        spearman_r = None

    # ── Compile report ─────────────────────────────────────────────────────────
    report = {
        "n_eval": len(subset),
        "n_golden": len(golden),
        "agent": {
            "classification": {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "per_class_report": clf_report,
            },
            "escalation": {
                "accuracy": round(esc_acc, 4),
                "escalate_f1": round(esc_f1, 4),
            },
            "reply_quality": {
                "bleu_mean": round(bleu_mean, 4),
                "rouge_l_mean": round(rouge_mean, 4),
            },
            "llm_judge_mean": round(judge_mean, 3) if judge_mean else None,
            "llm_judge_human_pearson_r": pearson_r,
            "llm_judge_human_spearman_r": spearman_r,
        },
        "baselines": baselines_out,
    }

    # ── Save outputs ───────────────────────────────────────────────────────────
    report_path = OUTPUTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nEvaluation report saved → {report_path}")

    pred_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "retrieved"}
        for r in results
    ])
    pred_df.to_csv(OUTPUTS_DIR / "agent_predictions.csv", index=False)
    logger.info(f"Predictions saved → outputs/agent_predictions.csv")

    if judge_scores:
        pd.DataFrame(judge_scores).to_csv(OUTPUTS_DIR / "judge_scores.csv", index=False)
        logger.info(f"Judge scores saved → outputs/judge_scores.csv")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  EVALUATION SUMMARY")
    print("="*65)
    print(f"  Eval sample:          {len(subset)} rows ({per_class} per class)")
    print(f"\n  [Intent Classification]")
    print(f"    Accuracy:           {acc:.4f}")
    print(f"    Macro F1:           {macro_f1:.4f}")
    print(f"    Weighted F1:        {weighted_f1:.4f}")
    print(f"\n  [Escalation Decision]")
    print(f"    Accuracy:           {esc_acc:.4f}")
    print(f"    Escalate F1:        {esc_f1:.4f}")
    print(f"\n  [Reply Quality]")
    print(f"    BLEU:               {bleu_mean:.4f}")
    print(f"    ROUGE-L:            {rouge_mean:.4f}")
    if judge_mean:
        print(f"\n  [LLM Judge Mean]:     {judge_mean:.2f}/5.0")
    print(f"\n  [Baselines]")
    for name, m in baselines_out.items():
        print(f"    {name:<20} acc={m['accuracy']:.4f}  macro_F1={m['macro_f1']:.4f}")
    print("="*65)

    return report


if __name__ == "__main__":
    run_eval()
