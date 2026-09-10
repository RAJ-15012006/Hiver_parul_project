#!/usr/bin/env python3
"""
demo.py
-------
Interactive command-line demo of the AmazonHelp AI Support Agent.

Usage:
    python demo.py
    python demo.py --no-rag      # no retrieval augmentation
    python demo.py --no-llm      # keyword-only offline mode
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="AmazonHelp AI Agent Demo")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval")
    parser.add_argument("--no-llm", action="store_true", help="Offline keyword-only mode")
    args = parser.parse_args()

    from agent import run_agent
    from embeddings import load_index

    # Load FAISS index if RAG is enabled
    faiss_index, faiss_meta = None, None
    if not args.no_rag:
        try:
            faiss_index, faiss_meta = load_index()
            print("✅  FAISS RAG index loaded.\n")
        except FileNotFoundError:
            print("⚠️   FAISS index not found. Run: python src/embeddings.py\n"
                  "     Continuing without RAG.\n")

    use_llm = not args.no_llm

    print("╔══════════════════════════════════════════════════════╗")
    print("║   AmazonHelp AI Support Agent — Interactive Demo     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("Type a customer message (or 'quit' to exit).\n")

    while True:
        try:
            text = input("Customer > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = run_agent(text, faiss_index=faiss_index, faiss_meta=faiss_meta, use_llm=use_llm)

        decision_icon = "🔴 ESCALATE" if result["escalation_decision"] == "ESCALATE" else "🟢 AUTO"

        print(f"\n  Intent:      {result['intent']}")
        print(f"  Decision:    {decision_icon}")
        print(f"  Reason:      {result['escalation_reason']}")
        print(f"\n  Draft Reply:\n  {result['draft_reply']}")

        if result.get("retrieved"):
            print(f"\n  [RAG: top similar past case]")
            top = result["retrieved"][0]
            print(f"    Customer: {top['customer_text'][:90]}…")
            print(f"    Reply:    {top['brand_reply'][:90]}…")
            print(f"    Score:    {top['similarity']:.3f}")
        print()


if __name__ == "__main__":
    main()
