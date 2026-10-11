# Strategic Decision Memo: Indic Conversational Tone Adaptation

**To:** VP of Product & AI Leadership | **From:** AI Engineering Team | **Date:** October 11, 2026  
**Subject:** Conversational Tone Strategy for Indic Models | **Strategy Selected:** **Path (c) — Prompt-Engineering Only**

### 1. Executive Recommendation & Tradeoffs
We recommend **Path (c): Prompt Engineering** (persona conditioning + language-specific few-shot exemplars).
* **Path (a) (SFT) Disqualified:** Fatal reviewer blindspot. With $0 external API budget and **zero native reviewers for 4 of 6 languages** (Tamil, Telugu, Bengali, Marathi), modifying weights risks unmonitored catastrophic degradation and safety drift before the 3-week launch.
* **Path (b) (Rewriter Model) Disqualified:** Adds serialized inference latency, secondary VRAM footprint for weights/KV cache, and compound hallucination risk.
* **Path (c) (Prompt Engineering) Selected:** Freezes base weights, incurs zero training costs, enables immediate iteration on Day 1, and focuses 100% of human review bandwidth on prompt evaluation.

### 2. Core Assumptions & Serving Arithmetic
* **Human Review Budget:** 1 reviewer (Hindi + Kannada only) @ 10 h/wk × 2 weeks = 20 hours (1,200 min). At ~2.5 min per paired comparison $\implies$ **480 total evaluation slots**.
* **Serving & Context Budget:** Assumed prompt overhead is **~35 tokens** (<5 ms prefill latency, *assumed until profiled*).
  * *KV Memory:* $35 \text{ tokens} \times 112 \text{ KiB/token} = \mathbf{3.83 \text{ MiB/seq}}$ ($91.88 \text{ MiB}$ across batch 24, ~0.76% of 12.08 GB KV pool).
  * *Context Limit:* At maximum 4096-token sequences (3584 in + 512 out), adding 35 tokens ($4,131$) exceeds `max_model_len`; context must be managed by trimming 35 prompt history tokens or capping generation at 477.
* **Constraints:** $0 external API budget; launch review in 3 weeks.

### 3. Reviewer Allocation (480 slots total)
* **Days 1–2 (Screen 3 arms vs baseline, 40 queries):** $40 \times 3 = \mathbf{120 \text{ comparisons}} = \mathbf{5.0 \text{ hours}}$.
* **Days 3–5 (Refine top 2 variants, 20 queries):** $20 \times 2 = \mathbf{40 \text{ comparisons}} = \mathbf{1.7 \text{ hours}}$.
* **Day 6 (Milestone checkpoint):** $\mathbf{60 \text{ comparisons}} = \mathbf{2.5 \text{ hours}}$ (Week 1 Total = 9.2 hrs).
* **Week 2 (Held-Out Validation):** 200 held-out evaluations + 40 safety/edge-case checks = $\mathbf{240 \text{ comparisons}} = \mathbf{10.0 \text{ hours}}$.

### 4. Success Thresholds, Guardrails & Kill Criterion
* **Primary Metric:** Mean Casualness Score on 5-point Likert scale (1=stiff, 5=natural).
  * **Threshold:** **$\ge 3.5 / 5.0$**, which must pass **separately for Hindi and Kannada** (formal baseline to be measured Day 1).
* **Fidelity & Safety:** Factual preservation **$\ge 98.0\%$**; Safety pass rate **$100\%$**.
* **Unreviewed Languages Policy:** Without native reviewers, Tamil, Telugu, Bengali, and Marathi **cannot be validated**; gate behind an experimental flag or defer until native reviewers are funded.
* **Feasible Kill Criterion:** If by **Day 6**, reviewer casualness remains **$< 3.2 / 5.0$** for either Hindi or Kannada, or prompt persona induces $>3\%$ safety refusals:
  * **Kill Action:** **Retain the existing formal baseline for launch.** Do not attempt an unbudgeted emergency fine-tuning run. Restrict conversational tone rollout only to validated languages.

### 5. Day-1 Experiment
* **Queries:** 40 prompts (20 Hindi, 20 Kannada) across greetings, advice, and product inquiries.
* **4-Arm Screening:** Arm 0 (Baseline) vs Arm 1 (Negative constraint), Arm 2 (Linguistic register rules), Arm 3 (Few-shot conversational exemplars).
* **Deliverable:** Generate 160 responses on Day 1; execute 120 paired comparisons with the reviewer on Day 2.
