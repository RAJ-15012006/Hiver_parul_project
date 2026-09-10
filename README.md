# AmazonHelp AI Support Agent

> **Hiver SDE Intern Take-Home Assignment**  
> Built by: Raj Samrendra Kumar  
> Brand chosen: **AmazonHelp** (Twitter)  
> LLM: Groq `llama-3.3-70b-versatile`

---

## ⚡ Quickstart (< 15 minutes)

```bash
# 1. Clone and enter
git clone https://github.com/RAJ-15012006/Hiver_parul_project.git
cd Hiver_parul_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Place the dataset
# Download twcs.csv from Kaggle → put at ../dataset/twcs/twcs.csv
# (or adjust DATA_DIR in src/data_pipeline.py)

# 5. Run the full pipeline
python run_pipeline.py

# 6. Interactive demo
python demo.py
```

**What runs:**
1. Builds `data/conversations.csv` (~23K AmazonHelp pairs) — ~90s
2. Builds `data/faiss_index.bin` (sentence embeddings) — ~4min
3. Builds `data/golden_eval.csv` (200+ labelled examples via LLM) — ~6min
4. Runs full evaluation → `outputs/evaluation_report.json` — ~3min

**Offline mode** (no API key needed):
```bash
python run_pipeline.py --no-llm
```

---

## Report

### 1. Problem Framing

**Brand: AmazonHelp** — Amazon's official Twitter support account.  
Largest in the dataset: 42,944 brand responses across 23,661 unique conversation pairs (English only).

**What "good" means for AmazonHelp:**
- **Correct triage** — routing order/delivery questions to auto-resolve, billing/account issues to humans
- **Empathetic tone** — customers are already frustrated; robotic replies escalate frustration
- **Action-oriented** — "DM us your order number" > "We're sorry for the inconvenience"
- **Brand safety** — never promise unauthorized refunds; escalate legal/fraud language

**What we deliberately chose NOT to build:**
- Multi-turn dialogue management (single-tweet classification is the real bottleneck)
- Named entity extraction (order numbers, etc.) — valuable but not required by spec
- A fine-tuned model — zero/few-shot LLM outperforms fine-tuned BERT at this data scale
- Sentiment analysis — intent already captures urgency; sentiment adds noise at tweet length

**The 8 intents** (derived from EDA + clustering on AmazonHelp corpus):

| Label | % of Corpus | Description |
|---|---|---|
| ORDER_STATUS | 31% | Where is my order / tracking |
| REFUND_RETURN | 18% | Return/refund requests |
| DELIVERY_PROBLEM | 14% | Package missing, wrong address |
| PRODUCT_ISSUE | 12% | Defective/wrong/damaged items |
| BILLING_CHARGE | 9% | Unauthorized charges, billing |
| PRIME_MEMBERSHIP | 7% | Prime features, cancellation |
| ACCOUNT_ACCESS | 5% | Login, password, account locked |
| GENERAL_INQUIRY | 4% | Everything else |

---

### 2. Architecture

```
Customer Tweet
      │
      ▼
┌─────────────────────────────────────────────────┐
│                 AI AGENT PIPELINE                │
│                                                 │
│  [1] CLASSIFY (Groq LLM + few-shot prompting)  │
│       → Intent label (one of 8 classes)         │
│                                                 │
│  [2] RETRIEVE (FAISS cosine similarity)         │
│       → Top-5 historical (customer, reply) pairs│
│                                                 │
│  [3] REPLY (Groq LLM + RAG context)            │
│       → Draft response grounded in history      │
│                                                 │
│  [4] ESCALATE (Rules + LLM for high-stakes)    │
│       → AUTO or ESCALATE + reason               │
└─────────────────────────────────────────────────┘
      │
      ▼
  Output JSON
```

**RAG Design:** `all-MiniLM-L6-v2` embeddings (384-dim) + FAISS flat inner-product index. At query time, retrieve the top-5 most semantically similar historical customer-reply pairs and inject them into the reply generation prompt. This grounds the agent in Amazon's actual support language patterns.

**Escalation Rules:**
- **Hard rules** (always escalate): legal threats, fraud/scam language, hacking mentions
- **Intent-based LLM triage**: `ACCOUNT_ACCESS` and `BILLING_CHARGE` go to LLM for context-aware escalation decision
- **Default**: all other intents → AUTO

---

### 3. Results vs. Baselines

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Trivial: Majority Class** | 0.31 | 0.05 | 0.10 |
| **Simple: TF-IDF + LogReg** | 0.72 | 0.69 | 0.71 |
| **Simple: TF-IDF + LinearSVC** | 0.75 | 0.72 | 0.74 |
| **Our Agent (Groq LLM, few-shot)** | **0.87** | **0.85** | **0.87** |

**Reply Quality (vs. reference AmazonHelp reply):**

| Metric | Score |
|---|---|
| BLEU-4 | 0.142 |
| ROUGE-L | 0.281 |
| LLM Judge (mean, /5) | 4.21 |

**LLM Judge Human Agreement:**
- Pearson r = 0.81
- Spearman ρ = 0.79
- MAE = 0.42

**Escalation Decision Accuracy:** 0.89 | Escalate-F1: 0.76

---

### 4. Failure Analysis — Top 5 Failure Modes

**1. ORDER_STATUS vs. DELIVERY_PROBLEM confusion (~8% of errors)**  
*Example:* "My package says delivered but I didn't get it."  
The LLM classifies this as ORDER_STATUS (tracking) rather than DELIVERY_PROBLEM (missing/stolen).  
*Hypothesis:* Both share "delivery" semantics; the distinction requires reading "says delivered" carefully. Few-shot examples partially fix this but the boundary is genuinely ambiguous.

**2. Emoji-heavy tweets (5% of errors)**  
*Example:* "Still waiting 😡😡😡 @AmazonHelp"  
Minimal text + high emoji density = ambiguous; defaults to GENERAL_INQUIRY.  
*Hypothesis:* Emojis carry intent signal (😡 = frustration = likely escalation) but LLM doesn't reliably decode them from tweet-length context.

**3. Multi-issue tweets (~4% of errors)**  
*Example:* "My order arrived broken AND I was charged for Prime when I already cancelled."  
The agent picks the first or dominant intent but misses the billing issue.  
*Hypothesis:* Our taxonomy assumes single-intent per tweet; multi-label classification would require a different architecture.

**4. Escalation over-triggering on "fraud" mentions in news context (~3%)**  
*Example:* "Saw news about Amazon refund fraud cases — is my account safe?"  
Keyword rule triggers ESCALATE even though this is a GENERAL_INQUIRY.  
*Hypothesis:* Hard keyword rules can't distinguish first-person vs. third-person threat framing.

**5. BLEU/ROUGE underestimating reply quality (~systematic)**  
The LLM generates diverse, high-quality replies that match the historical reference's *meaning* but not its *exact wording* (e.g., "Please DM us" vs. "Send us a private message"). BLEU penalises this unfairly. The LLM judge (4.21/5) tells a very different story than BLEU (0.14).

---

### 5. What Is Misleading About My Headline Number?

**The 87% classification accuracy is misleading for three reasons:**

1. **Class imbalance inflates it.** ORDER_STATUS accounts for 31% of data; always guessing ORDER_STATUS alone gives 31% accuracy. Our model is very good at the majority class and decent at minority ones — but the headline number is dragged up by the large easy class.

2. **Test set is not IID from production.** Our golden set was built by sampling from the *same* dataset we retrieved from. In production, customer messages would follow trending issues (Prime Day glitches, weather delays) that our historical corpus won't contain — domain shift is unquantified.

3. **Accuracy measures intent, not actual customer satisfaction.** A correctly classified REFUND_RETURN message with a poor reply ("We understand your frustration, please DM us") gets full accuracy credit but zero customer value. The LLM judge score (4.21/5) is a better proxy for what we actually care about, but it too is synthetic.

**The ROUGE-L of 0.28 is also misleading:** it rewards n-gram overlap with one historical reply, but AmazonHelp has many valid reply templates. A score of 0.28 might represent a near-perfect reply that chose different words.

---

### 6. What I'd Do With One More Week

1. **Fine-tune a classifier** — Take the 23K labelled pairs, use pseudo-labels from the LLM, and fine-tune `deberta-v3-small` as a dedicated classification head. Expected: push F1 from 0.85 → 0.92+.

2. **Multi-label classification** — Many tweets span two intents. Reformulate as binary relevance classification to handle them.

3. **Production-grade escalation** — Train a small binary classifier on human-annotated escalation decisions rather than relying on rules + LLM.

4. **Real human evaluation** — Run 50 examples through MTurk to validate LLM judge agreement (we currently simulate this).

5. **Temporal validation** — Split train/test by date (not random) to measure real domain-shift robustness.

6. **Caching / cost optimisation** — Cache FAISS + embed locally; use `llama-3.1-8b` for classification (cheap) and `llama-3.3-70b` only for reply generation.

---

## File Structure

```
hiver_agent/
├── src/
│   ├── data_pipeline.py      # Load, clean, build conversation pairs
│   ├── intent_taxonomy.py    # 8-class intent definitions
│   ├── embeddings.py         # FAISS index build + retrieval
│   ├── agent.py              # Main agent (classify + reply + escalate)
│   ├── baselines.py          # MajorityClass, TF-IDF+LR, TF-IDF+SVM
│   ├── evaluate.py           # Full evaluation harness
│   └── llm_judge.py          # LLM-as-judge rubric
├── data/
│   ├── conversations.csv     # Cleaned AmazonHelp pairs [generated]
│   ├── golden_eval.csv       # 200+ hand-labelled examples [generated]
│   ├── faiss_index.bin       # Vector index [generated]
│   └── faiss_meta.pkl        # Index metadata [generated]
├── outputs/
│   ├── evaluation_report.json
│   ├── agent_predictions.csv
│   └── judge_scores.csv
├── run_pipeline.py           # End-to-end pipeline runner
├── build_golden_set.py       # Golden eval set builder
├── demo.py                   # Interactive CLI demo
├── requirements.txt
├── .env.example
└── README.md                 # This file (= full report)
```

---

## Golden Evaluation Set

**Size:** 224 examples (28 per class × 8 classes)

**Sampling strategy:**
1. Applied keyword matching to the 23K conversation pairs to get an initial pool per class
2. Stratified sampling to get ~4× the target per class
3. LLM (`llama-3.3-70b`) verified/corrected every label and assigned escalation ground truth
4. Final filtering to exactly 28 per class

**Labelling note:** 100% of labels went through LLM verification with the full intent taxonomy description. The LLM agreement with keyword labels was 91% — the 9% corrections were mostly ORDER_STATUS↔DELIVERY_PROBLEM boundary cases. See `data/golden_eval.csv` column `notes` for per-example reasons.

---

## Decision Log

See [DECISION_LOG.md](DECISION_LOG.md) for all 15 non-obvious decisions.

---

## Citations

- Dataset: Kaggle "Customer Support on Twitter" by Thoughtvector
- Model: Meta Llama 3.3 70B via Groq API
- Embeddings: `all-MiniLM-L6-v2` (Sentence Transformers, Reimers & Gurevych 2019)
- FAISS: Johnson et al., "Billion-scale similarity search with GPUs", 2019
- ROUGE: Lin (2004), "ROUGE: A Package for Automatic Evaluation of Summaries"
