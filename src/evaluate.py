"""
evaluate.py
-----------
Full evaluation harness for the AmazonHelp AI agent.

Runs:
  1. Intent classification accuracy/F1 on the golden eval set
  2. Escalation decision accuracy on the golden eval set
  3. Reply quality: BLEU, ROUGE-L vs. ground-truth historical replies
  4. LLM-as-judge scores on a sample
  5. Baseline comparison table

Outputs a JSON + Markdown report in outputs/evaluation_report.*
"""

import json
import logging
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# NLTK for BLEU
import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
GOLDEN_CSV  = Path(__file__).parent.parent / "data" / "golden_eval.csv"


# ── Text quality metrics ───────────────────────────────────────────────────────
def bleu_score(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU-4 with add-1 smoothing."""
    smoothie = SmoothingFunction().method1
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


# ── Classification evaluation ─────────────────────────────────────────────────
def eval_classification(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return {
        "accuracy":     round(acc, 4),
        "macro_f1":     round(macro_f1, 4),
        "weighted_f1":  round(weighted_f1, 4),
        "per_class":    report,
    }


# ── Reply quality evaluation ──────────────────────────────────────────────────
def eval_reply_quality(
    predictions: List[Dict[str, str]],
    references: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE-L between draft replies and reference replies.
    If references is None, uses 'brand_reply' from each prediction dict.
    """
    bleus, rouges = [], []
    for i, pred in enumerate(predictions):
        hyp = pred.get("draft_reply", "")
        ref = (references[i] if references else pred.get("brand_reply", ""))
        if not ref:
            continue
        bleus.append(bleu_score(hyp, ref))
        rouges.append(rouge_l_score(hyp, ref))

    return {
        "bleu_mean":   round(float(np.mean(bleus)), 4) if bleus else 0.0,
        "bleu_std":    round(float(np.std(bleus)), 4) if bleus else 0.0,
        "rouge_l_mean": round(float(np.mean(rouges)), 4) if rouges else 0.0,
        "rouge_l_std":  round(float(np.std(rouges)), 4) if rouges else 0.0,
        "n":           len(bleus),
    }


# ── Escalation evaluation ─────────────────────────────────────────────────────
def eval_escalation(
    y_true_esc: List[str],
    y_pred_esc: List[str],
) -> Dict[str, float]:
    acc = accuracy_score(y_true_esc, y_pred_esc)
    f1 = f1_score(
        y_true_esc, y_pred_esc,
        pos_label="ESCALATE",
        average="binary",
        zero_division=0,
    )
    return {
        "escalation_accuracy": round(acc, 4),
        "escalation_f1_escalate": round(f1, 4),
    }


# ── Main harness ──────────────────────────────────────────────────────────────
def run_evaluation(
    golden_csv: Path = GOLDEN_CSV,
    use_llm: bool = True,
    judge_sample: int = 25,
    run_baselines: bool = True,
    eval_sample: Optional[int] = 50,
) -> Dict[str, Any]:
    """
    Full evaluation pipeline. Loads golden set, runs agent + baselines, scores all.
    """
    import time
    from intent_taxonomy import LABELS
    from agent import run_agent, classify
    from embeddings import load_index
    from baselines import (
        MajorityClassBaseline, TFIDFLogRegBaseline, TFIDFSVMBaseline,
        evaluate_baseline,
    )
    from llm_judge import batch_judge, human_agreement_analysis

    if not golden_csv.exists():
        raise FileNotFoundError(f"Golden eval set not found: {golden_csv}")

    golden = pd.read_csv(golden_csv)
    logger.info(f"Loaded golden eval set: {len(golden):,} rows")

    # ── Load FAISS index (optional) ───────────────────────────────────────────
    try:
        faiss_index, faiss_meta = load_index()
        logger.info("FAISS index loaded.")
    except FileNotFoundError:
        logger.warning("FAISS index not found — running without RAG.")
        faiss_index, faiss_meta = None, None

    # Determine subset for evaluation if specified
    if eval_sample is not None and eval_sample < len(golden):
        per_class = max(2, eval_sample // len(LABELS))
        eval_df = golden.groupby("intent", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), per_class), random_state=42)
        ).reset_index(drop=True)
        logger.info(f"Evaluating agent on stratified sample of {len(eval_df)} rows...")
    else:
        eval_df = golden
        logger.info(f"Evaluating agent on full set of {len(eval_df)} rows...")

    # ── Run agent on golden set ───────────────────────────────────────────────
    logger.info("Running agent on golden eval set …")
    agent_results = []
    for _, row in eval_df.iterrows():
        result = run_agent(
            row["customer_text"],
            faiss_index=faiss_index,
            faiss_meta=faiss_meta,
            use_llm=use_llm,
        )
        time.sleep(0.35)
        result["true_intent"] = row.get("intent", "")
        result["true_escalation"] = row.get("escalation", "AUTO")
        result["brand_reply"] = row.get("brand_reply", "")
        agent_results.append(result)

    # ── Classification metrics ────────────────────────────────────────────────
    y_true_intent = [r["true_intent"] for r in agent_results]
    y_pred_intent = [r["intent"] for r in agent_results]
    clf_metrics   = eval_classification(y_true_intent, y_pred_intent, labels=LABELS)
    logger.info(f"Agent accuracy: {clf_metrics['accuracy']:.3f} | macro_F1: {clf_metrics['macro_f1']:.3f}")

    # ── Escalation metrics ────────────────────────────────────────────────────
    y_true_esc = [r["true_escalation"] for r in agent_results]
    y_pred_esc = [r["escalation_decision"] for r in agent_results]
    esc_metrics = eval_escalation(y_true_esc, y_pred_esc)
    logger.info(f"Escalation accuracy: {esc_metrics['escalation_accuracy']:.3f}")

    # ── Reply quality ─────────────────────────────────────────────────────────
    reply_metrics = eval_reply_quality(agent_results)
    logger.info(f"BLEU: {reply_metrics['bleu_mean']:.3f} | ROUGE-L: {reply_metrics['rouge_l_mean']:.3f}")

    # ── LLM-as-judge ─────────────────────────────────────────────────────────
    judge_results = []
    judge_agreement = {}
    if use_llm and judge_sample > 0:
        logger.info(f"Running LLM judge on {judge_sample} samples …")
        judge_results = batch_judge(agent_results[:judge_sample], sleep_between=0.3)
        judge_scores = [r.get("judge_overall", 3.0) for r in judge_results]
        # Simulated human scores for agreement (from golden set column if available)
        if "human_score" in golden.columns:
            human_scores = golden["human_score"].dropna().tolist()[:judge_sample]
            if len(human_scores) == len(judge_scores):
                judge_agreement = human_agreement_analysis(judge_scores, human_scores)
        mean_judge = float(np.mean(judge_scores)) if judge_scores else 0.0
        logger.info(f"LLM judge mean overall score: {mean_judge:.2f}/5")
    else:
        mean_judge = None

    # ── Baselines ─────────────────────────────────────────────────────────────
    baseline_metrics = {}
    if run_baselines and len(golden) > 20:
        from sklearn.model_selection import train_test_split
        X_all = golden["customer_text"].tolist()
        y_all = golden["intent"].tolist()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
        )
        for Cls in [MajorityClassBaseline, TFIDFLogRegBaseline, TFIDFSVMBaseline]:
            model = Cls()
            model.fit(X_tr, y_tr)
            bm = evaluate_baseline(model, X_te, y_te, labels=LABELS)
            baseline_metrics[model.name] = {
                "accuracy":    round(bm["accuracy"], 4),
                "macro_f1":    round(bm["macro_f1"], 4),
                "weighted_f1": round(bm["weighted_f1"], 4),
            }
            logger.info(
                f"Baseline [{model.name}]: acc={bm['accuracy']:.3f} "
                f"macro_f1={bm['macro_f1']:.3f}"
            )

    # ── Assemble results ──────────────────────────────────────────────────────
    report = {
        "n_golden": len(golden),
        "agent": {
            "classification":  clf_metrics,
            "escalation":      esc_metrics,
            "reply_quality":   reply_metrics,
            "llm_judge_mean":  round(mean_judge, 4) if mean_judge else None,
            "llm_judge_agreement": judge_agreement,
        },
        "baselines": baseline_metrics,
    }

    # ── Save report ───────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save agent predictions
    pred_df = pd.DataFrame(agent_results)
    pred_df.to_csv(OUTPUTS_DIR / "agent_predictions.csv", index=False)

    if judge_results:
        pd.DataFrame(judge_results).to_csv(
            OUTPUTS_DIR / "judge_scores.csv", index=False
        )

    _print_summary(report)
    return report


def _print_summary(report: Dict) -> None:
    """Pretty-print evaluation summary."""
    ag = report["agent"]
    clf = ag["classification"]
    esc = ag["escalation"]
    rq  = ag["reply_quality"]

    print("\n" + "="*65)
    print("  EVALUATION SUMMARY")
    print("="*65)
    print(f"  Golden set size:      {report['n_golden']:,}")
    print(f"\n  [Intent Classification]")
    print(f"    Accuracy:           {clf['accuracy']:.4f}")
    print(f"    Macro F1:           {clf['macro_f1']:.4f}")
    print(f"    Weighted F1:        {clf['weighted_f1']:.4f}")
    print(f"\n  [Escalation Decision]")
    print(f"    Accuracy:           {esc['escalation_accuracy']:.4f}")
    print(f"    Escalate F1:        {esc['escalation_f1_escalate']:.4f}")
    print(f"\n  [Reply Quality]")
    print(f"    BLEU:               {rq['bleu_mean']:.4f} ± {rq['bleu_std']:.4f}")
    print(f"    ROUGE-L:            {rq['rouge_l_mean']:.4f} ± {rq['rouge_l_std']:.4f}")
    if ag.get("llm_judge_mean"):
        print(f"\n  [LLM Judge Mean]:     {ag['llm_judge_mean']:.2f}/5.0")
    if ag.get("llm_judge_agreement"):
        jag = ag["llm_judge_agreement"]
        print(f"  [Human Agreement]:    Pearson={jag['pearson_r']}, Spearman={jag['spearman_rho']}")
    if report["baselines"]:
        print(f"\n  [Baseline Comparison]")
        print(f"    {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
        for name, bm in report["baselines"].items():
            print(f"    {name:<25} {bm['accuracy']:>10.4f} {bm['macro_f1']:>10.4f}")
        print(f"    {'llm_agent (our)':<25} {clf['accuracy']:>10.4f} {clf['macro_f1']:>10.4f}")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_evaluation(use_llm=True, judge_sample=30, run_baselines=True)
