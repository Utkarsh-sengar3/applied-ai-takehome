# Applied AI / ML Engineering — Take-Home Assignment Submission

Candidate: **Utkarsh Sengar** (`utkarshsengar3@gmail.com`)  
Date Submitted: **November 25, 2026**  
Submission Document: [`submission_template.docx`](./submission_template.docx)  

This monorepo contains the end-to-end production code, test suites, evaluation harnesses, cost analysis models, and verification artifacts for both problems in the Applied AI / ML Engineering assignment:
1. **Problem 1 — Cost-Efficient RAG Application** (`problem1_cost_rag/`)
2. **Problem 2 — LLM-as-Judge Evaluation Pipeline** (`problem2_llm_judge/`)

---

## 📁 Repository Structure

```
applied-ai-takehome/
├── problem1_cost_rag/                     # Problem 1: Cost-Efficient RAG System
│   ├── src/
│   │   ├── config.py                     # Pydantic settings & environment configuration
│   │   ├── ingest.py                     # Multi-format parser (PDF/HTML/MD) + deterministic SHA256 chunking
│   │   ├── retriever.py                  # Top-k vector retrieval with metadata filtering & relevance threshold
│   │   ├── generator.py                  # Grounded generation with strict [Chunk X] citations & refusal gate
│   │   ├── service.py                    # Production FastAPI HTTP service (/query, /ingest, /stats, /health)
│   │   └── cli.py                        # Command-line interface for ingestion, querying, and stats
│   ├── data/                             # Multi-format corpus (PDF, HTML, Markdown)
│   ├── evaluation/
│   │   ├── eval_dataset.json             # 20 curated test queries (16 in-domain + 4 unanswerable)
│   │   ├── harness.py                    # 3-layer automated evaluation harness (IR, generation, latency)
│   │   ├── cost_model.py                 # TCO cost calculator across scale (100K, 1M, 10M vectors)
│   │   └── results.json                  # Empirical evaluation results
│   ├── artifacts/                        # High-resolution diagrams and terminal execution screenshots
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
├── problem2_llm_judge/                    # Problem 2: LLM-as-Judge Evaluation Pipeline
│   ├── src/
│   │   ├── config.py                     # Decoupled judge and generator configurations
│   │   ├── rubric.py                     # Explicit 5-dimension rubric with calibrated score anchors
│   │   ├── parser.py                     # 5-tier resilient structured JSON verdict parser with fallback
│   │   ├── judge.py                      # Pointwise and pairwise judge with position-bias order-swap
│   │   ├── pipeline.py                   # Suite aggregation, pass rate, win rate, and reporting
│   │   └── cli.py                        # CLI runner for suites, A/B comparisons, and validation
│   ├── suites/
│   │   ├── test_suite.json               # 20 pointwise test cases
│   │   ├── ab_comparison_suite.json      # 20 pairwise comparison pairs (Prompt v1 vs v2)
│   │   ├── adversarial_probe_suite.json  # Verbose-but-wrong and terse-but-correct probes
│   │   └── validation_suite.json         # 25 benchmark cases with human consensus gold labels
│   ├── reports/                          # Generated suite reports, A/B reports, and validation summaries
│   ├── artifacts/                        # Architecture diagrams and terminal execution screenshots
│   ├── validate_pipeline.py              # Automated validation runner (Cohen's kappa, flip rates, probes)
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
├── submission_template.docx              # Fully populated, grader-ready Word submission template
├── .gitignore
└── README.md                             # Monorepo master documentation
```

---

## ⚡ Problem 1 Summary — Cost-Efficient RAG Application

- **Store Chosen**: **ChromaDB (Embedded Persistent Mode on SQLite + C++ HNSW)**.
  - Zero idle compute cost ($0.00/mo) co-located in application runtime.
  - In-process sub-5ms retrieval latency on commodity CPU without network serialization overhead.
- **Strict Idempotency**: Chunk IDs are computed as `SHA256(source + chunk_idx + text)`. Re-running ingestion is 100% idempotent and produces zero duplicate vectors (`Delta: +0`).
- **Hallucination Prevention**: Pre-generation distance gate (`dist <= 1.15`, `sim >= 0.35`). Queries without relevant context short-circuit with standardized refusal, eliminating hallucinations.
- **Evaluation Results (20 questions, $k=5$)**:
  - **Recall@k / Hit Rate**: **90.0%**
  - **MRR (Mean Reciprocal Rank)**: **0.850**
  - **nDCG@5**: **0.884**
  - **Context Precision**: **0.829**
  - **Faithfulness / Groundedness**: **0.950 (95.0%)**
  - **Retrieval P50 / P95**: **3.8 ms / 8.6 ms** (warm in-memory)
- **Cost Comparison Across Scale**:
  - **100K vectors**: $0.00/mo vs $70.00/mo managed (100% savings)
  - **1M vectors**: $0.96/mo vs $280.00/mo managed (99.6% savings)
  - **10M vectors**: $9.60/mo vs $2,100.00/mo managed (99.5% savings)

---

## ⚖️ Problem 2 Summary — LLM-as-Judge Evaluation Pipeline

- **Judging Architecture**:
  - Pointwise 5-dimension rubric scoring for absolute regression release gating.
  - Pairwise A-vs-B comparison with symmetrical order-swapping for model/prompt upgrade decisions.
- **Explicit 5-Dimension Rubric**: Correctness (0.30), Faithfulness (0.25), Completeness (0.20), Instruction-Following (0.15), and Tone & Safety (0.10) with score anchors (1.0 to 5.0).
- **Bias Mitigations Implemented in Code & Measured**:
  - **Position Bias**: Symmetrical order-swap evaluates `(A, B)` and `(B, A)`; agreement enforced. **Flip rate reduced from 25.0% to 0.0%**.
  - **Verbosity Bias**: Length-normalized scoring penalizes responses $>2.5\times$ longer than reference without new facts.
  - **Self-Enhancement Bias**: Strictly decoupled cross-family judge (`meta-llama/llama-3.3-70b-instruct`) and generator (`mistralai/mistral-7b-instruct`).
  - **Sycophancy / Style**: Mandatory per-criterion evidence citation.
- **Judge Validation**:
  - **Human Consensus Agreement**: **92.0% (23 / 25 cases)**
  - **Cohen's Kappa ($\kappa$)**: **0.828** (Near-perfect inter-annotator agreement)
  - **Test-Retest Consistency**: **100.0% (0.0% flip rate)** at $T=0.0$.
  - **Adversarial Probes**:
    - *Verbose-but-Wrong Probe*: Scored **1.88 / 5.0 (FAIL)** — Judge was **NOT fooled**.
    - *Terse-but-Correct Probe*: Scored **4.47 / 5.0 (PASS)** — Judge was **NOT fooled**.
- **A/B Comparison (Prompt v1 vs Prompt v2)**:
  - **Config B (Prompt v2 - Structured CoT)** won with a **100.0% win rate (20/20 wins)** over Config A (0.0% win rate, 3.20 mean score).

---

## 🚀 Quick Reproduction Commands

### Running Problem 1 (Cost-Efficient RAG):
```bash
cd problem1_cost_rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Ingest corpus (idempotent)
python -m src.cli ingest --data ./data

# Run query
python -m src.cli query "What is the monthly cost comparison for 1M vectors on ChromaDB vs Managed DB?" --k 5

# Run 3-layer evaluation harness
python -m evaluation.harness
```

### Running Problem 2 (LLM-as-Judge):
```bash
cd problem2_llm_judge
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Run full evaluation & validation suite
python validate_pipeline.py

# Run pairwise A/B comparison CLI
python -m src.cli compare-ab --suite suites/ab_comparison_suite.json
```
