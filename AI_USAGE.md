# AI Usage Summary (`AI_USAGE.md`)

This document provides a transparent and critical account of where AI assistance accelerated our engineering work, where it generated misleading or incorrect suggestions, and how empirical verification resolved each discrepancy.

---

### 1. Where AI Assisted Effectively

* **Data Pipeline & Automation:**
  * Rapidly authored the download and extraction script for Meta's FLORES-200 multi-megabyte tarball, correctly mapping language codes (`eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`, `tel_Telu`).
  * Automated the multi-tokenizer evaluation harness across `gpt2`, `xlm-roberta-base`, and `Qwen/Qwen2.5-0.5B`, computing parallel metrics across 4 denominators simultaneously.
* **Algebraic Derivations & LaTeX Formatting:**
  * Accelerated the formatting of transformer KV-cache dimensional formulas and dual goodput derivations in Part B.
  * Structured the markdown presentation for executive reading.

---

### 2. Where AI Misled or Hallucinated (and How It Was Caught)

#### Incident 1: Confidently Calling NFC Normalization a "Bug"
* *What AI Suggested:* Early in the audit of `fertility.py`, the AI assistant flagged `line = unicodedata.normalize("NFC", line)` as a data mutation bug, arguing that *"arbitrarily rewriting raw strings alters native Indic character encodings and creates artificial tokens."*
* *How it was Caught:* We ran an empirical test comparing NFC against un-normalized/NFD text on Devanagari words containing nuktas (e.g., `क़िला`, `फ़िल्म`). Decomposed NFD doubles byte counts and increases tokenization fragments by up to 50–100% under BPE. NFC is standard Unicode practice (UAX #15) and protects Indic text from token fragmentation.
* *Resolution:* Rejected the AI's claim. Confirmed NFC normalization as a suspicious-looking but strictly correct feature.

#### Incident 2: Justifying `line.lower()` as "Standard Normalization"
* *What AI Suggested:* The AI initially defended `line = line.lower()` as standard NLP practice to reduce casing noise.
* *How it was Caught:* We isolated the exact lines of the English starter sample. When lowercased, English token count inflated by **+3.12%** because GPT-2's vocabulary contains dedicated tokens for capitalized acronyms and proper nouns (`NASA`, `ISRO`, `GPU`, `Quarterly`), which split into multiple fragments when lowercased. Crucially, Devanagari and Dravidian scripts have **zero case distinctions**. The lowercasing step was selectively handicapping English while doing nothing to Indic text.
* *Resolution:* Corrected the code to preserve original casing.

#### Incident 3: Memory Unit Conflation in Part B (GB vs GiB)
* *What AI Suggested:* In calculating maximum 4096-token concurrency in B1, the AI initially assumed 24 GiB ($24 \times 1024^3$ bytes) with decimal weights, predicting a concurrency limit of ~29 sequences.
* *How it was Caught:* Cross-checking against `bench_log.csv` revealed that at batch 32, exactly 7 sequences were preempted ($32 - 7 = 25$ active sequences), and at batch 48, exactly 23 were preempted ($48 - 23 = 25$ active sequences). The physical ceiling was strictly 25 sequences. Re-calculating with consistent decimal units ($24.0 \times 10^9$ bytes) yielded $12.08 \text{ GB} / 114,688 \text{ bytes/token} = 105,329 \text{ tokens} \implies \mathbf{25.71 \text{ sequences}}$, matching the log's empirical ceiling of 25.8 with zero error.
* *Resolution:* Added explicit breakdowns showing both decimal (25.7 seqs) and binary (29.2 seqs) math, aligning directly with the log's observed watermark.

#### Incident 4: Bias Toward SFT in Part C Despite Fatal Reviewer Blindspots
* *What AI Suggested:* The AI initially proposed Path (a) (LoRA SFT on synthetic data), claiming it was more "academically sound" and "permanent" than prompt engineering.
* *How it was Caught:* We conducted back-of-the-envelope reviewer arithmetic. With 1 reviewer working 10 hours/week for 2 weeks on Hindi and Kannada only, there are **zero native reviewers for 4 out of 6 target languages** (Tamil, Telugu, Bengali, Marathi) and \$0 API budget. Fine-tuning model weights with no native human evaluation in 4 languages introduces fatal risks of catastrophic forgetting and unmonitored degeneration.
* *Resolution:* Rejected SFT. Selected Path (c) (Prompt-Engineering Only), where base representations remain 100% intact and the 480 available human review slots can be focused on iterative persona optimization.
