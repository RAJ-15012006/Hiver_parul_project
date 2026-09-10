# Decision Log

15 non-obvious decisions made during the AmazonHelp AI agent project.

---

1. **Chose AmazonHelp over all other brands.**  
   AmazonHelp has the most data by 3×: 42,944 brand responses vs. 15,694 for #2 (AppleSupport). More data = better RAG retrieval, more reliable baseline training, more diverse golden set sampling. The diversity of Amazon's support topics (orders, Prime, Alexa, Kindle, billing) also makes for a richer taxonomy.

2. **Used 8 intents, not fewer or more.**  
   4 intents would mix ORDER_STATUS with DELIVERY_PROBLEM (very different escalation treatment). 12+ intents would produce too few examples per class for the baselines and would fragment the golden set. 8 captures the major decision branches with enough within-class coherence.

3. **Used Groq (llama-3.3-70b) instead of OpenAI GPT-4.**  
   Groq's free tier offers sub-second inference for the 70B model, making the full evaluation loop (200+ LLM calls) feasible in minutes. GPT-4 would cost ~$3–5 for this evaluation, require billing setup, and be slower. The 70B model is competitive in quality.

4. **Keyword fallback classifier instead of pure LLM.**  
   When the Groq API rate-limits or errors, a pure LLM system fails silently. The keyword fallback ensures the agent degrades gracefully and remains testable offline. It also serves as a fast baseline sanity check.

5. **FAISS flat inner-product with L2-normalised embeddings.**  
   For 23K vectors this is exact (no approximation loss) and takes <1MB on disk. HNSW would be overkill at this scale and adds parameter tuning complexity. Cosine similarity via L2-normalised IP is mathematically equivalent and faster in FAISS.

6. **Filtered to English-only conversations (60% ASCII heuristic).**  
   AmazonHelp replies in Japanese, Spanish, Portuguese. Including multilingual text would make intent taxonomy meaningless (the taxonomy was defined in English) and would corrupt the RAG retrieval. English-only gives 23K clean pairs; multilingual might double the count but halve quality.

7. **Built conversation pairs (customer, reply) not individual tweets.**  
   The raw dataset has individual tweets. Reconstructing pairs is essential for: (a) RAG — we retrieve on customer text and return the brand reply; (b) golden set — we need the actual brand resolution, not just the customer complaint; (c) BLEU/ROUGE — we score against the reference reply.

8. **Sampled 28 per class for the golden set (not random sampling).**  
   Random sampling would give 0 examples of GENERAL_INQUIRY (4% of corpus) and 87 examples of ORDER_STATUS (31%). Stratified per-class sampling ensures every intent is evaluatable, and F1 per class is meaningful rather than dominated by the majority class.

9. **LLM-verified golden labels, not pure keyword labels.**  
   Pure keyword matching gives ~75% accuracy (as measured by our baselines). Using the LLM to verify/correct labels gives a cleaner ground truth that the agent's accuracy is measured against. Not doing this would produce a golden set with ~25% label noise, making all downstream metrics meaningless.

10. **Escalation via rules + LLM hybrid, not pure LLM.**  
    Pure LLM escalation would give inconsistent decisions on the same input (temperature variation). Pure rules miss nuanced signals ("I'm so frustrated I could scream" → AUTO, "I'm going to sue Amazon" → ESCALATE). The hybrid: hard rules catch legal/fraud language deterministically; LLM handles the grey zone for high-stakes intents.

11. **5-dimension LLM judge instead of single overall score.**  
    A single score hides what's wrong. Knowing that helpfulness=4.8 but conciseness=2.1 tells us the reply is detailed but too long for Twitter. This diagnostic power is crucial for iteration and justifying the agent to stakeholders.

12. **Used `sentence-transformers/all-MiniLM-L6-v2` not a larger model.**  
    Embedding 23K texts with `all-MiniLM-L6-v2` takes ~4 minutes on CPU. The same with `all-mpnet-base-v2` takes 15+ minutes and produces marginally better embeddings. For Twitter support retrieval (short, topically narrow texts) the quality difference is negligible.

13. **Reported BLEU/ROUGE even though they're misleading for this task.**  
    The spec asks for automated metrics. We compute them faithfully but explicitly call out in the report why they underestimate reply quality. The LLM judge is the primary quality signal; BLEU/ROUGE provide external verification and reproducibility.

14. **Did not use Banking77 secondary dataset.**  
    Banking77 is finance-domain specific and contains 77 very fine-grained intents like "card_about_to_expire", "direct_debit_payment_not_recognised". These don't map cleanly to Amazon's support taxonomy and would require remapping that introduces more noise than signal.

15. **Chose `llm_judge_mean` as the headline metric in the README, not accuracy.**  
    Intent classification accuracy is easy to inflate (by choosing an easy brand or simple intents). LLM judge mean measures what actually matters: does the customer get a helpful, empathetic, accurate reply? It's harder to game and closer to what Hiver's product would care about.

16. **Engineered Entity Slot-Guardrails to eliminate Failure Mode #1 (Order ID Fixation).**  
    Rather than merely cataloging that the agent asked for an Order ID on account lockouts or when a Tracking ID was already present, we implemented an extraction pipeline that injects hard negative constraints into the drafting prompt. This drove Order ID fixation on non-order queries from 37.5% to 0.0%.

17. **Pre-prompt RAG Context Sanitization to eliminate Failure Mode #3 (Monetary Context Leakage).**  
    Historical retrieved tweets occasionally contain past discretionary compensation claims (e.g., "$5 credit", "$10 gift card"). Injecting these into the prompt led the LLM to promise unauthorized financial credits. We regex-sanitize historical text to replace specific currency claims with neutral policy terms before LLM injection.

18. **Built an Interactive Streamlit Web Application (`app.py`) alongside CLI tools.**  
    Senior engineering requires making systems accessible to non-technical stakeholders and hiring managers. A full web UI allows instant 1-click verification of edge cases, live slot extraction inspection, RAG similarity exploration, and real-time LLM judge scoring without shell commands.
