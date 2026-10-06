# Strategic Decision Memo: Multilingual Conversational Tone Adaptation

**To:** VP of Product & AI Leadership  
**From:** AI Engineering Team  
**Date:** October 7, 2026  
**Subject:** Strategy & Execution Plan for Casualizing Indic Assistant Responses  
**Selected Strategy:** **Path (c) — Prompt-Engineering Only**

---

### 1. Executive Summary & Path Selection

We recommend **Path (c): Prompt-Engineering Only** (system prompt persona conditioning + language-specific few-shot conversational exemplars). 

**Why Paths (a) and (b) are Disqualified:**
* **Path (b) (Rewriter Model):** Adds a secondary $\le 1$B model in series, doubling inference latency, consuming scarce L4 VRAM, and creating two potential points of hallucination.
* **Path (a) (SFT on Synthetic Pairs):** SFT is fatally bottlenecked by the **review capacity blindspot**. We have zero external API budget to generate teacher data and **zero native speakers for 4 of the 6 languages** (Tamil, Telugu, Bengali, Marathi). Weight updates risk catastrophic forgetting, safety regression, and unmonitored linguistic degradation across the unreviewed languages with no recovery runway before the 3-week launch.
* **Path (c) (Prompt Engineering):** Zero training risk, zero catastrophic forgetting, zero infrastructure additions, and enables daily feedback cycles starting Day 1 while keeping the A100 free for automated evaluation sweeps.

---

### 2. Explicit Assumptions

1. **Reviewer Capacity:** 1 native reviewer (Hindi + Kannada only) working 10 hours/week for 2 weeks = 20 total reviewer hours. Reviewing one paired response (rating tone, grammar, and factual fidelity) takes 2.5 minutes.
2. **Serving & Hardware:** Main model is FLM-4B running on NVIDIA L4 (24 GB) with a tight 24-concurrency KV budget. Added tokens must not degrade generation goodput.
3. **Budget & Horizon:** \$0 external API budget (no GPT-4/Claude calls for data synthesis or judging). Hard deadline for launch review in exactly 3 weeks (21 days).

---

### 3. Back-of-the-Envelope Arithmetic

* **Reviewer Throughput Budget:**
  $$\text{Total Review Minutes} = 20 \text{ hours} \times 60 \text{ min/hr} = 1,200 \text{ minutes}$$
  $$\text{Total Human Evaluations} = \frac{1,200 \text{ min}}{2.5 \text{ min/eval}} = \mathbf{480 \text{ response evaluations}}$$
  *Allocation:* 240 evaluations in Hindi and 240 in Kannada across the 2-week period (120 evals/week/language).
* **Serving Overhead:**
  A language-specific casual persona prompt + 1 conversational few-shot exemplar adds **$\approx 35$ prompt tokens**.
  * Latency impact: On NVIDIA L4, 35 prefill tokens process in **$<5 \text{ ms}$** (virtually zero impact on TTFT).
  * KV-cache impact: $35 \text{ tokens} \times 112 \text{ KiB} / 4096 \approx 0.96 \text{ KiB}$ per sequence ($<0.01\%$ of GPU memory; concurrency ceiling of 24 is entirely unaffected).
* **Training & Infrastructure Cost:**
  * Training cost: **\$0** (0 GPU training hours required).
  * Compute allocation: The assigned A100-80GB is deployed as an offline testbed for batch inference across all 6 languages and automated heuristic filtering.

---

### 4. Success Metric & Numerical Threshold

* **Primary Tone Metric (Human Review):** Mean Casualness Score on a 5-point Likert scale (1 = stiff/formal textbook, 5 = natural spoken conversational).
  * **Threshold:** **$\ge 3.8 / 5.0$** on Hindi and Kannada evaluation sets (compared to the baseline formal output benchmark of $\le 2.2 / 5.0$).
* **Fidelity & Safety Guardrail:**
  * **Factual & Information Preservation:** **$\ge 98.0\%$** (no loss of factual content or hallucination introduced by colloquial phrasing).
  * **Safety / Toxicity Pass Rate:** **$100\%$** (colloquial phrasing must not trigger slang-associated abuse or safety filter violations).
* **Surrogate Automated Metric (Unreviewed Languages):** Ratio of formal grammatical markers (e.g., textbook verb endings) to conversational markers drops by **$\ge 65\%$** measured by linguistic heuristics across Tamil, Telugu, Bengali, and Marathi.

---

### 5. Kill Criterion & Pivot Trigger

* **Kill Criterion:** If by **Day 6 (End of Week 1)**, after testing 4 distinct prompt architectures (direct persona instructions, register-specific rules, and few-shot exemplars) over 120 Hindi and 120 Kannada reviewer evaluations:
  1. The mean casualness score remains **$< 3.2 / 5.0$**, OR
  2. The prompt causes formatting degradation, refusal rate $> 3\%$, or factual drift $> 2\%$.
* **Action Upon Trigger:** Immediately terminate the prompt-only path on **Day 7**. Pivot to a tightly constrained parameter-efficient LoRA adapter (rank 8, targeting only attention projection layers) trained on the A100 using a high-precision, human-curated 300-example seed dataset in Hindi and Kannada, freezing all MLP and base representations to protect unreviewed languages.

---

### 6. Day-1 Experiment

* **Objective:** Establish the empirical headroom between zero-shot base output and prompt-conditioned outputs.
* **Test Dataset:** Construct 40 parallel test queries (20 Hindi, 20 Kannada) spanning common user scenarios (product questions, casual advice, greetings, technical explanation).
* **Experimental Arms:**
  1. *Arm 0 (Baseline):* Default system prompt ("You are a helpful assistant").
  2. *Arm 1 (Negative Constraint Prompt):* Explicit negative instructions ("Do not use textbook formal language. Avoid literary verb endings. Write like a friendly peer on WhatsApp").
  3. *Arm 2 (Linguistic Register Guidance):* Script-specific register instructions (e.g., Hindi: natural 'tum'/'aap' balance, spoken postpositions; Kannada: colloquial spoken verb suffixes).
  4. *Arm 3 (Few-Shot Exemplar):* 1 high-quality, authentic two-turn conversational exemplar per language.
* **Deliverable by Day 1 End:** Run batch inference on the A100; deliver the 40 paired outputs to the reviewer for first-pass scoring within the first 4-hour review block on Day 2.
