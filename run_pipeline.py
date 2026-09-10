#!/usr/bin/env python3
"""
run_pipeline.py
---------------
End-to-end reproducible pipeline for the Hiver AmazonHelp AI Agent.

Steps:
  1. Data pipeline   → data/conversations.csv
  2. FAISS index     → data/faiss_index.bin + faiss_meta.pkl
  3. Golden set      → data/golden_eval.csv  (if not already built)
  4. Evaluation      → outputs/evaluation_report.json + summaries

Usage:
    python run_pipeline.py [--skip-data] [--skip-index] [--skip-golden]
                           [--no-llm] [--judge-sample N]

Example (full run, ≤ 15 min on modern laptop):
    python run_pipeline.py

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
CONVS_CSV  = DATA_DIR / "conversations.csv"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
GOLDEN_CSV = DATA_DIR / "golden_eval.csv"

# Paths to upstream dataset
DATASET_PATH = Path(__file__).parent.parent / "dataset" / "twcs" / "twcs.csv"


def step1_data(force: bool = False) -> None:
    if CONVS_CSV.exists() and not force:
        logger.info(f"[Step 1] conversations.csv already exists ({CONVS_CSV}). Skipping.")
        return
    logger.info("[Step 1] Building conversation pairs from raw dataset …")
    t0 = time.time()
    from data_pipeline import run as build_convs
    convs = build_convs(DATASET_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    convs.to_csv(CONVS_CSV, index=False)
    logger.info(f"[Step 1] Done in {time.time()-t0:.1f}s — {len(convs):,} pairs")


def step2_index(force: bool = False) -> None:
    if INDEX_PATH.exists() and not force:
        logger.info(f"[Step 2] FAISS index already exists. Skipping.")
        return
    logger.info("[Step 2] Building FAISS embedding index …")
    t0 = time.time()
    import pandas as pd
    from embeddings import build_index
    convs = pd.read_csv(CONVS_CSV)
    build_index(convs)
    logger.info(f"[Step 2] Done in {time.time()-t0:.1f}s")


def step3_golden(force: bool = False, use_llm: bool = True) -> None:
    if GOLDEN_CSV.exists() and not force:
        logger.info(f"[Step 3] golden_eval.csv already exists. Skipping.")
        return
    logger.info("[Step 3] Building golden evaluation set …")
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).parent))
    from build_golden_set import build_golden_set
    build_golden_set(CONVS_CSV, use_llm_verify=use_llm)
    logger.info(f"[Step 3] Done in {time.time()-t0:.1f}s")


def step4_evaluate(use_llm: bool = True, judge_sample: int = 8) -> None:
    logger.info("[Step 4] Running evaluation harness …")
    t0 = time.time()
    from run_eval import run_eval
    report = run_eval()
    logger.info(f"[Step 4] Done in {time.time()-t0:.1f}s")
    return report


def main():
    parser = argparse.ArgumentParser(description="AmazonHelp AI Agent Pipeline")
    parser.add_argument("--skip-data",   action="store_true", help="Skip data pipeline step")
    parser.add_argument("--skip-index",  action="store_true", help="Skip FAISS index step")
    parser.add_argument("--skip-golden", action="store_true", help="Skip golden set step")
    parser.add_argument("--no-llm",      action="store_true", help="Run in offline mode (no LLM API)")
    parser.add_argument("--judge-sample", type=int, default=50, help="N examples for LLM judge (default 50)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild all steps")
    args = parser.parse_args()

    use_llm = not args.no_llm
    force   = args.force_rebuild

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  AmazonHelp AI Support Agent Pipeline    ║")
    logger.info("╚══════════════════════════════════════════╝")

    t_start = time.time()

    if not args.skip_data:
        step1_data(force=force)

    if not args.skip_index:
        step2_index(force=force)

    if not args.skip_golden:
        step3_golden(force=force, use_llm=use_llm)

    step4_evaluate(use_llm=use_llm, judge_sample=args.judge_sample)

    elapsed = time.time() - t_start
    logger.info(f"\n✅  Pipeline complete in {elapsed/60:.1f} minutes.")
    logger.info(f"   Results → outputs/evaluation_report.json")
    logger.info(f"   Predictions → outputs/agent_predictions.csv")


if __name__ == "__main__":
    main()
