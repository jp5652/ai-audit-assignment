# Research & Audit Lab Notebook: Chronological Investigation Log

**Project:** AI Team Intern Assignment — The Audit  
**Auditor:** AI Systems & Infrastructure Audit Team  
**Date:** October 7, 2026  
**Format:** Chronological Log (Hypothesis $\to$ Experiment $\to$ Result $\to$ Revision)

---

## Entry 0: Baseline Verification & Starter Corpus Inspection

* **Hypothesis:** The figures reported in `REPORT_v0.md` (English fertility 1.27, tok/char 0.226; Hindi fertility 7.45, tok/char 1.579) can be directly reproduced from `starter_kit/fertility.py` using `gpt2`.
* **Experiment:** Ran the unmodified starter script on the starter samples:
  ```bash
  python starter_kit/fertility.py \
    --corpus eng=starter_kit/corpus_sample/eng_sample.txt \
    --corpus hin=starter_kit/corpus_sample/hin_sample.txt \
    --tokenizer gpt2
  ```
* **Result:**
  * English: `fertility = 1.27`, `tok/char = 0.226`
  * Hindi: `fertility = 7.45`, `tok/char = 1.579`
  * Reported conclusion: *"hin is 5.89x the fertility of eng (worse tokenization)"*
  * Matches `REPORT_v0.md` down to the exact hundredth decimal.
* **Surprising Discovery (Corpus Sanity Check):**
  Inspecting `eng_sample.txt` and `hin_sample.txt` revealed two major defects:
  1. *Not parallel line-by-line:* Contrary to the intern's claim, Line 1 in English is about Bengaluru airport (*"Bengaluru International Airport handled record traffic in March."*), whereas Line 1 in Hindi is about tea (*"मुझे सुबह की चाय बहुत पसंद है।"*). Sentence 3 in Hindi corresponds to Sentence 3 in English, but Sentence 4 in Hindi corresponds to Sentence 8 in English, and Sentence 1 in English has no Hindi counterpart at all.
  2. *Corrupted spacing:* Line 7 in `eng_sample.txt` contains a double space (*"Please keep the books  in the cupboard."*), and Line 10 in `hin_sample.txt` contains a double space (*"किताबें  अलमारी में रखी हैं।"*).

---

## Entry 1: Code Audit & Bug Isolation in `fertility.py`

### 1.1 The Whitespace Splitting Bug (`line.split(" ")` vs `line.split()`)
* **Hypothesis:** Splitting text with `line.split(" ")` (line 62) creates empty strings `""` whenever text contains consecutive whitespace, artificially inflating the word count and depressing fertility.
* **Experiment:** Isolated line 7 in English and line 10 in Hindi. Compared `line.split(" ")` against standard `line.split()`.
  ```python
  "Please keep the books  in the cupboard.".split(" ")
  # Output: ['Please', 'keep', 'the', 'books', '', 'in', 'the', 'cupboard.'] (len: 8)
  "Please keep the books  in the cupboard.".split()
  # Output: ['Please', 'keep', 'the', 'books', 'in', 'the', 'cupboard.'] (len: 7)
  ```
* **Result & Evidence:**
  * English sample: Total words changed from 79 to 78 (-1). Macro fertility increased from 1.2652 to 1.2831 (+1.41%).
  * Hindi sample: Total words changed from 62 to 61 (-1). Macro fertility increased from 7.4485 to 7.5985 (+2.01%).
* **Conclusion:** `line.split(" ")` is an active code bug. It must be replaced with `line.split()`.

### 1.2 Asymmetric Lowercasing Distortion (`line.lower()`)
* **Hypothesis:** Lowercasing text before tokenization (`line = line.lower()`, line 60) was intended by the intern to "remove casing noise", but BPE tokenizers like GPT-2 are trained on cased text with specific uppercase merge operations.
* **Dead End / Early Misconception:** We initially wondered if lowercasing might help compression by reducing vocabulary sparsity.
* **Experiment:** Tokenized original cased text vs lowercased text with `gpt2`:
  ```bash
  python partA/run_audit_experiments.py
  ```
* **Result & Evidence:**
  * English sample: Cased tokens = **96**, Lowercased tokens = **99** (+3 tokens, **+3.12% inflation**).
    * Line 2: *"The Quarterly Review..."* (`Quarterly` is 1 cased token $\to$ `quarterly` splits into 2 tokens).
    * Line 6: *"NASA and ISRO..."* (`NASA` is 1 cased token $\to$ `nasa` splits into 2 tokens).
    * Line 10: *"The GPU cluster..."* (`GPU` is 1 cased token $\to$ `gpu` splits into 2 tokens).
  * Hindi sample: Cased tokens = **459**, Lowercased tokens = **459** (**0.00% delta**).
* **Conclusion:** Devanagari and Dravidian scripts have no casing distinctions. Lowercasing selectively penalized English by +3.12% while leaving Indic text untouched. It introduces asymmetric cross-lingual distortion and must be removed.

### 1.3 Aggregation Flaw: Macro-Average vs Micro-Average
* **Hypothesis:** Computing $\frac{1}{N}\sum \frac{\text{tokens}_i}{\text{words}_i}$ (macro-average) is mathematically biased compared to $\frac{\sum \text{tokens}}{\sum \text{words}}$ (micro-average), giving excessive weight to short outlier lines.
* **Experiment:** Measured macro-average vs micro-average on both sample corpora.
* **Result & Evidence:**
  * English: Macro = 1.2472 vs Micro = 1.2308 (-1.32% delta).
  * Hindi: Macro = 7.5985 vs Micro = 7.5246 (-0.97% delta).
* **Conclusion:** Serving infrastructure bills on total tokens across all traffic ($\sum T / \sum W$), not the mean of per-line ratios. Micro-average is the correct capacity planning metric.

### 1.4 Suspicious Feature Audit: `unicodedata.normalize("NFC", line)`
* **Suspicion:** A junior reviewer might suspect that normalizing lines to NFC alters native Indic characters or mutates data.
* **Experiment:** Analyzed Indic words containing combining nuktas and matras (e.g. `क़िला`, `फ़िल्म`, `ज़रूरत`, `गाड़ी`) in both NFC and decomposed NFD.
* **Result & Evidence:** 
  * Under Python's `unicodedata`, Devanagari words maintain **identical UTF-8 byte lengths** (e.g., 18 bytes vs 18 bytes) and **identical token counts** (12 tokens vs 12 tokens in GPT-2) across NFC and NFD (delta = 0).
  * NFC normalization is standard defensive hygiene (UAX #15) ensuring canonical representation across heterogeneous inputs, but it does not alter token counts for these words.
* **Conclusion:** `unicodedata.normalize("NFC", line)` is **harmless and strictly correct**. Flagging it as a bug would be an error. (Note: Our primary defensible proof of a "suspicious but correct" behavior is Experiment 1.5, `add_special_tokens=False`).

### 1.5 Suspicious Feature Audit: `add_special_tokens=False`
* **Suspicion:** Does omitting BOS/EOS undercount tokens?
* **Experiment:** Tested `xlm-roberta-base` with `add_special_tokens=True` vs `False`.
* **Result & Evidence:** Adding special tokens increased fertility from 1.2633 to 1.5338 (**+21.4% skew**) due to a constant +2 offset on 10-word sentences.
* **Conclusion:** `add_special_tokens=False` is **harmless and strictly correct** for measuring intrinsic tokenization fertility.

---

## Entry 2: Evaluation Corpus Assembly (FLORES-200)

* **Objective:** Replace the 10-sentence unaligned toy sample with a rigorous, standard multilingual evaluation corpus.
* **Corpus Selection:** Downloaded official Meta NLLB FLORES-200 `devtest` split (1,012 sentence-aligned parallel sentences):
  * English (`eng_Latn`)
  * Hindi (`hin_Deva`)
  * Kannada (`kan_Knda`)
  * Tamil (`tam_Taml`)
  * Telugu (`tel_Telu`)
* **Corpus Statistics (1,012 Parallel Sentences):**
  * `eng`: 21,901 words, 131,966 characters, 132,096 UTF-8 bytes
  * `hin`: 25,643 words, 131,180 characters, 337,439 UTF-8 bytes
  * `kan`: 16,100 words, 138,027 characters, 375,341 UTF-8 bytes
  * `tam`: 16,775 words, 154,131 characters, 421,635 UTF-8 bytes
  * `tel`: 16,938 words, 132,465 characters, 353,661 UTF-8 bytes
* **Crucial Linguistic Finding:**
  Notice that Kannada has **26.5% fewer words** than English, Tamil has **23.4% fewer words**, and Telugu has **22.7% fewer words** for the exact same semantic sentences. Dravidian languages are agglutinative: case markers and prepositions attach directly to words.
  Therefore, dividing tokens by words artificially inflates Dravidian fertility numbers.
* **What the Corpus Cannot Tell Us:**
  FLORES-200 consists of formal, professionally translated news and encyclopedic sentences. It cannot evaluate performance on colloquial conversational dialogue, noisy user chat, or Latin-transliterated code-mixed text (Hinglish/Kanglish).

---

## Entry 3: Corrected Multilingual Tokenizer Benchmark

* **Experiment:** Ran `partA/run_cross_lang_bench.py` evaluating 3 tokenizers across all 1,012 sentences and 4 denominators:
  1. `gpt2` (tiktoken, 50k vocab, English-only baseline)
  2. `xlm-roberta-base` (SentencePiece, 250k vocab, multilingual baseline)
  3. `Qwen/Qwen2.5-0.5B` (BPE, 152k vocab, modern LLM tokenizer)
* **Results Table:**

| Tokenizer | Language | Total Tokens | Tok / Sentence (Semantic) | Ratio vs Eng (Semantic) | Tok / Word | Tok / Char | Tok / Byte |
|---|---|---|---|---|---|---|---|
| **gpt2** | eng | 27,044 | 26.72 | 1.00× | 1.23 | 0.205 | 0.205 |
| | hin | 200,688 | 198.31 | **7.42×** | 7.83 | 1.530 | 0.595 |
| | kan | 367,366 | 363.01 | **13.58×** | 22.82 | 2.662 | 0.979 |
| | tam | 420,171 | 415.19 | **15.54×** | 25.05 | 2.726 | 0.997 |
| | tel | 350,748 | 346.59 | **12.97×** | 20.71 | 2.648 | 0.992 |
| **xlm-roberta** | eng | 30,661 | 30.30 | 1.00× | 1.40 | 0.232 | 0.232 |
| | hin | 38,221 | 37.77 | **1.25×** | 1.49 | 0.291 | 0.113 |
| | kan | 41,459 | 40.97 | **1.35×** | 2.58 | 0.300 | 0.110 |
| | tam | 41,354 | 40.86 | **1.35×** | 2.47 | 0.268 | 0.098 |
| | tel | 40,372 | 39.89 | **1.32×** | 2.38 | 0.305 | 0.114 |
| **Qwen2.5** | eng | 27,621 | 27.29 | 1.00× | 1.26 | 0.209 | 0.209 |
| | hin | 121,957 | 120.51 | **4.42×** | 4.76 | 0.930 | 0.361 |
| | kan | 191,127 | 188.86 | **6.92×** | 11.87 | 1.385 | 0.509 |
| | tam | 168,828 | 166.83 | **6.11×** | 10.06 | 1.095 | 0.400 |
| | tel | 193,169 | 190.88 | **6.99×** | 11.40 | 1.458 | 0.546 |

* **Key Takeaway:** Under `xlm-roberta-base`, Hindi is only **1.25×** English tokens, and Kannada/Tamil/Telugu are **1.32× to 1.35×**. The previous intern's 6× finding was entirely an artifact of testing `gpt2`. Furthermore, **Tokens per Parallel Sentence** is the single metric that should drive routing decisions because it holds user utility and semantic information constant.

---

## Entry 4: Serving Capacity Reconciliation (Part B)

### 4.1 KV-Cache Sizing Arithmetic (B1)
* **Equation:**
  $$\text{KV bytes/token} = 2 \times 28 \text{ layers} \times 8 \text{ heads} \times 128 \text{ dim} \times 2 \text{ bytes} = \mathbf{114,688 \text{ bytes}} = \mathbf{112 \text{ KiB}}$$
* **Max Concurrency:**
  $$\text{Usable RAM} = 24.0 \text{ GB} \times 0.92 = 22.08 \text{ GB}$$
  $$\text{KV Pool} = 22.08 - 8.40 \text{ (weights)} - 1.60 \text{ (overhead)} = 12.08 \text{ GB}$$
  $$\text{Token Capacity} = \frac{12.08 \times 10^9}{114,688} = 105,329 \text{ tokens} \implies \frac{105,329}{4096} = \mathbf{25.71 \approx 25 \text{ sequences}}$$
* **Check against `bench_log.csv`:**
  * Batch 16: $16 / 0.62 = \mathbf{25.8}$
  * Batch 24: $24 / 0.93 = \mathbf{25.8}$
  * Batch 32: $32 - 7 \text{ preemptions} = \mathbf{25.0}$
  * Batch 48: $48 - 23 \text{ preemptions} = \mathbf{25.0}$
  Exact agreement between theory and log.

### 4.2 Anomaly Mechanism (B2)
* At batch 32 and 48, throughput collapses from 1607 tok/s to 1384 and 1298 tok/s because the engine exhausts KV cache blocks (`kv_cache_util = 0.97`).
* Preemptions occur (7 at batch 32, 23 at batch 48), triggering re-prefill thrashing.
* Proposed fix: Cap `max_num_seqs = 24`. Eliminates preemptions, prevents thrashing, preserves 1607 tok/s throughput, and finishes 48 requests in 122.3s instead of 151.4s (19.2% speedup).

### 4.3 Column Misread & Honest Goodput (B3)
* `reported_tok_s` measured total throughput (prefill + decode tokens/sec).
* Prompt tokens constituted 87.5% of total tokens processed.
* Honest goodput of batch 24 derived via two agreeing whole-run methods:
  * Method 1 (Gen tokens / Wall-clock): $(24 \times 512) / 61.16\text{s} = \mathbf{200.916 \text{ output tok/s}}$
  * Method 2 (Reported throughput × Output fraction): $1607.4 \times (512 / 4096) = \mathbf{200.925 \text{ output tok/s}}$
  * Both methods agree on **~200.92 output tok/s**, roughly 8× lower than the reported 1607.4 tok/s!

### 4.4 Production Metric (B4)
* Pull `vllm:num_preemptions_total`. Expected values: counter increase ($\Delta$) is 0 for batch $\le 24$, $\ge 7$ at batch 32, $\ge 23$ at batch 48.

---

## Entry 5: Decision Memo Formulation (Part C)

* **Constraint Analysis:** 1 A100 for 2 weeks, 1 reviewer for 10 h/week in Hindi + Kannada only, 3 weeks to launch, \$0 API budget.
* **The Reviewer Blindspot:** Only 480 total human reviews possible ($20 \text{ hrs} \times 60 / 2.5 \text{ min}$). Tamil, Telugu, Bengali, and Marathi have ZERO native speakers available.
* **Path Elimination:**
  * Path (b) (Rewriter): Doubles latency, doubles memory overhead, compound errors. Disqualified.
  * Path (a) (SFT): Fine-tuning weights risks unmonitored catastrophic forgetting in 4 of 6 languages with no reviewer to catch failures. Disqualified.
  * Path (c) (Prompt Engineering): Zero training risk, zero catastrophic degradation, enables daily iteration starting Day 1. Selected.
* **Framework Defined:** Assumptions, 480-review arithmetic, 3.5/5.0 casualness threshold, 98% factual preservation guardrail, Day-6 kill criterion, and Day-1 experiment with 4 prompt arms.

---

## Entry 6 (October 11, 2026): Peer Review, Evidence Corrections & Metric Reconciliation

Following internal review and rigorous verification against the evidence rule, several analytical and arithmetic discrepancies were identified and corrected:

### 6.1 NFC Evidence Reconciliation
* **Finding:** In `partA/run_audit_experiments.py`, our initial explanation asserted that decomposed NFD increases token counts by 50–100%. However, our printed experiment table on sample Devanagari words (`फ़िल्म`, `ज़रूरत`, `क़िला`, `गाड़ी`) showed **identical bytes (18 vs 18) and identical tokens (12 vs 12)** across NFC and NFD.
* **Correction:** We removed the contradictory narrative. In standard Unicode (UAX #15), Devanagari nukta characters possess composition exclusions, so NFC and NFD maintain identical decomposition under Python's `unicodedata`. Normalizing to NFC is benign (delta = 0).
* **Primary Defensible Finding:** Our primary empirically verified "suspicious but fine" example is `add_special_tokens=False` (Experiment 5), which demonstrated a real, measured **+21.42% artificial inflation** when special tokens were added.

### 6.2 Serving Goodput Derivation (B3)
* **Finding:** Our previous write-up presented a goodput range of 201–250 tok/s by dividing batch size by median ITL ($24 / 0.09607\text{s} = 249.8$). However, dividing by median ITL measures instantaneous decode-step rate, not whole-run throughput.
* **Correction:** We derived goodput using the two agreeing whole-run methods:
  1. *Generated tokens / wall-clock time:* $(24 \times 512) / 61.16 = \mathbf{200.916 \text{ output tok/s}}$
  2. *Reported throughput × output token fraction:* $1607.4 \times (512 / 4096) = \mathbf{200.925 \text{ output tok/s}}$
  Both independent methods agree on **~200.92 output tok/s** (the 0.009 difference is consistent with rounding in recorded log fields). Output goodput is ~8× lower than reported total throughput.
* **Corrected Degradation:** Comparing short vs. long prompt at matched batch size 16: short prompt delivered 294.46 output tok/s; long prompt delivered 163.94 output tok/s. This represents a **44.3% reduction** (correcting our prior "over 70%" estimate). Both prompt length and generation length change, so bandwidth pressure and prefill overhead are plausible mechanisms to verify with telemetry.

### 6.3 Serving Preemption Semantics & Latency Discrepancy (B1, B2)
* **Preempted Sequences:** `preempted_seqs` records sequences preempted at least once, not static simultaneous occupancy. The log is consistent with an approximate ~25-sequence physical capacity, and two-wave completion (122.32s) is a model prediction with batching assumptions.
* **Latency vs Wall Clock:** Twelve of thirteen rows have p95 latency exceeding the recorded run duration. The supplied files do not explain the measurement boundaries. Client/server timing differences are one hypothesis; harness instrumentation is needed to resolve this. We avoid presenting untested hypotheses as proven facts.

### 6.4 Conditional Production Cost Recommendation (A4)
* **Finding:** XLM-R's 1.25–1.35× ratio disproves a universal 6× script penalty, but cannot be transferred directly to FLM-4B. Our own Qwen2.5 test showed 4.42–6.99× ratios despite a 152k vocabulary.
* **Correction:** Recommendation memo updated to make capacity planning conditional: leadership must profile FLM-4B's actual tokenizer on representative tasks before setting production multipliers.

### 6.5 Memory Math & Reviewer Budget Corrections (Part C)
* **KV Memory Math:** Adding 35 prompt tokens per sequence adds $35 \times 112 \text{ KiB} = \mathbf{3.83 \text{ MiB per sequence}}$, totaling $\mathbf{91.88 \text{ MiB}}$ across 24 sequences (correcting the erroneous $0.96\text{ KiB}$ calculation).
* **Context Limit Tradeoff:** Adding 35 tokens to a 3,584-token prompt generating 512 tokens yields 4,131 tokens, exceeding `max_model_len` (4096). Context must be managed by trimming prompt history or capping generation.
* **Reviewer Plan & Feasible Fallback:** Day-1 screening requires 120 paired comparisons ($40 \text{ queries} \times 3 \text{ treatment arms} = 120$ pairs = 5.0 hours). The Day-7 fallback to an unbudgeted SFT training run was replaced with a feasible kill action: retain the formal baseline for launch and gate casualization to validated languages. Both memos condensed to $\le 1$ page.

### 6.6 Production Alert Baseline & Operational Threshold
* **Finding:** Alerting at a rigid 1.50× expansion threshold was unjustified when tokenizers like Qwen2.5 operate normally at 4.42–6.99×.
* **Correction:** Replaced the hardcoded threshold with an operational relative rule: alert if token expansion drifts **>20% above the validated task baseline**, explicitly labeled as a provisional operational choice.

### 6.7 Reviewer Budget Reconciliation & Separate Language Thresholds
* **Finding:** Stating "240 paired evaluations + edge cases" in Week 2 exceeded the 10-hour reviewer budget, and asserting baseline casualness as $\le 2.2/5$ lacked empirical support.
* **Correction:** Reconciled Week 2 budget as 200 held-out evaluations + 40 safety/edge-case checks = exactly 240 evaluations (10.0 hours). Clarified that baseline casualness is to be measured empirically on Day 1, and that the $\ge 3.5/5.0$ threshold must pass separately for Hindi and Kannada.
