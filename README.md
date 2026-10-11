# AI Team Intern Assignment — The Audit
## Multilingual Tokenizer Audit, Serving Capacity Reconciliation & AI Strategy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, reproduction-verified audit of the previous intern's tokenizer fertility analysis and GPU serving capacity report (`REPORT_v0.md`).

---

## Repository Structure

```text
my-submission/
├── NOTEBOOK.md                        # Chronological research lab notebook (Hypothesis → Experiment → Result → Revision)
├── AI_USAGE.md                        # Transparent record of AI usage and verified hallucinations
├── partA/
│   ├── prepare_corpus.py              # Multilingual parallel corpus builder (1,012 FLORES-200 sentences)
│   ├── run_audit_experiments.py       # Minimal reproducible experiments isolating all bugs & verifying features
│   ├── fertility_audited.py           # Production-ready, multi-tokenizer & multi-denominator benchmark tool
│   ├── run_cross_lang_bench.py        # Benchmark runner evaluating 3 tokenizers across 4 denominators
│   ├── corpus/                        # Clean parallel eval corpora (eng, hin, kan, tam, tel)
│   ├── results/
│   │   └── cross_language_benchmark.csv # Full experimental benchmark matrix
│   └── recommendation_memo.md         # ≤1-page routing & budgeting recommendation memo
├── partB/
│   ├── capacity_calculations.py       # Exact formula-based arithmetic verifying serving log reconciliation
│   └── analysis.md                    # Deep-dive capacity reconciliation, anomaly mechanism, & goodput derivations
├── partC/
│   └── memo.md                        # ≤1-page decision memo on Path (c) under strict constraints
└── starter_kit/                       # Preserved, untouched original starter files for baseline reference
```

---

## Quickstart & Reproducibility

### 1. Setup Environment
```bash
git clone <your-repo-url>
cd my-submission
pip install tiktoken transformers
```

### 2. Run Audit Experiments (Part A)
Isolate and measure the whitespace bug, lowercasing distortion, macro/micro aggregation trap, and Unicode NFC normalization:
```bash
python partA/run_audit_experiments.py
```

### 3. Run Multilingual Tokenizer Benchmark (Part A)
Evaluate English, Hindi, Kannada, Tamil, and Telugu on 1,012 parallel sentences across `gpt2`, `xlm-roberta-base`, and `Qwen/Qwen2.5-0.5B`:
```bash
python partA/run_cross_lang_bench.py
```

Or run the audited fertility tool directly:
```bash
python partA/fertility_audited.py \
  --corpus eng=partA/corpus/eng.txt \
  --corpus hin=partA/corpus/hin.txt \
  --corpus kan=partA/corpus/kan.txt \
  --tokenizer hf:xlm-roberta-base
```

### 4. Verify GPU Capacity & Goodput Calculations (Part B)
Verify theoretical KV-cache sizing, concurrency limits, and honest goodput:
```bash
python partB/capacity_calculations.py
```

---

## Summary of Findings

### Part A: Tokenizer Problem
* **Flaws in `fertility.py`:**
  * `line.split(" ")` creates empty strings on consecutive spaces, depressing fertility by up to 2%.
  * `line.lower()` selectively penalized English by +3.12% token inflation in GPT-2 BPE while doing 0% to uncased Indic scripts.
  * Macro-averaging introduces ratio-estimator bias; serving requires aggregate micro-averages.
* **The Conceptual Fallacy:** The claim that Hindi token explosion is "a property of the script, not the tokenizer" is false. When evaluated on 1,012 parallel sentences using `xlm-roberta-base`, Hindi token expansion is only **1.25×** English, and Dravidian languages are **1.32×–1.35×** (not 6×–7×).
* **Metric Choice:** **Tokens per Parallel Sentence (Semantic Task)** is the single metric that must drive routing decisions because word-based metrics heavily penalize agglutinative languages.

### Part B: Serving Capacity Reconciliation
* **KV Cache Per Token:** Exactly **114,688 bytes (112 KiB)** in fp16.
* **Concurrency Ceiling:** Max concurrent 4096-token sequences on 1× NVIDIA L4 (24 GB) is **~25 sequences**, consistent with `bench_log.csv` (cache saturates at 0.97 with 7 and 23 sequences preempted).
* **Throughput Anomaly:** Batches 32 and 48 suffer preemption thrashing (re-evaluating 3,584 prompt tokens from scratch) due to KV memory exhaustion. Capping concurrency at `--max-num-seqs 24` eliminates preemptions and preserves peak throughput (~1,607 tok/s).
* **Column Misread:** `reported_tok_s` measured total throughput (87.5% was prompt prefill). Honest generation goodput for batch 24 long-prompt is **~200.92 output tok/s** (derived via two independent agreeing methods), roughly 8× lower than reported. At matched batch 16, longer prompts reduce output goodput by **44.3%** (from 294.46 to 163.94 tok/s).

### Part C: Strategic Decision Memo
* **Selected Path:** **Path (c): Prompt-Engineering Only**.
* **Why SFT & Rewriter Fail:** SFT suffers from a fatal review blindspot (0 native reviewers for Tamil, Telugu, Bengali, Marathi) and $0 API budget. Modifying weights risks unmonitored catastrophic degradation. Rewriters introduce serialized execution overhead and extra memory allocation. Prompt engineering keeps base representations frozen, incurs ~3.83 MiB KV cache per sequence, and enables immediate iteration within our 480-review human budget.
