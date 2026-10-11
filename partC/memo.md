# Strategic Decision Memo: Indic Conversational Tone Adaptation

**To:** VP of Product & AI Leadership  
**From:** AI Engineering Team  
**Date:** October 11, 2026  
**Subject:** Strategy & Recommendation for Casualizing Indic Assistant Outputs  
**Selected Strategy:** **Path (c) — Prompt-Engineering Only**

---

### 1. Executive Recommendation & Path Tradeoffs

We recommend **Path (c): Prompt-Engineering Only** (system prompt persona conditioning + language-specific few-shot exemplars).

**Why Alternative Paths are Disqualified:**
* **Path (b) (Rewriter Model):** Adds serialized model execution overhead, consumes additional VRAM for second-model weights and KV cache, and creates dual failure points for hallucinations.
* **Path (a) (SFT on Synthetic Pairs):** SFT is disqualified by the **reviewer blindspot**. We have zero external API budget and **zero native reviewers for 4 of the 6 languages** (Tamil, Telugu, Bengali, Marathi). Modifying base model weights risks unmonitored catastrophic forgetting and safety regression across the unreviewed languages before the 3-week launch.
* **Path (c) (Prompt Engineering):** Leaves base representations intact, incurs zero training costs, enables daily iteration starting on Day 1, and allows us to focus 100% of available reviewer capacity on iterative prompt evaluation.

---

### 2. Explicit Assumptions

1. **Reviewer Capacity:** 1 native reviewer (Hindi + Kannada only) for 10 h/week over 2 weeks = 20 total hours (1,200 minutes). Rating one paired response (tone Likert 1–5, factual preservation) takes ~2.5 minutes $\implies$ **480 total evaluation slots**.
2. **Serving & Context Budget:** Assumed prompt overhead is **~35 tokens** with **<5 ms** prefill latency (*assumptions until measured*). If applied to maximum 4096-token sequences (3584 prompt + 512 gen), adding 35 tokens ($4,131$) exceeds `max_model_len`; context must be managed by trimming user history by 35 tokens or capping generation at 477 tokens.
3. **Budget & Horizon:** \$0 external API budget. Launch review in 3 weeks.

---

### 3. Back-of-the-Envelope Arithmetic

* **KV Memory Overhead:**
  $$35 \text{ tokens} \times 112 \text{ KiB/token} = 3,920 \text{ KiB} = \mathbf{3.83 \text{ MiB per sequence}}$$
  Across a batch of 24 concurrent sequences: $24 \times 3.83 \text{ MiB} = \mathbf{91.88 \text{ MiB}}$ (~0.76% of the 12.08 GB KV cache pool).
* **Reviewer Time Allocation:**
  * *Day 1–2 (Screening 3 prompt arms vs baseline on 40 queries):* $40 \times 3 = \mathbf{120 \text{ paired comparisons}} = 300 \text{ min} = \mathbf{5.0 \text{ hours}}$.
  * *Day 3–5 (Refining top 2 variants on 20 queries):* $20 \times 2 = \mathbf{40 \text{ comparisons}} = 100 \text{ min} = \mathbf{1.7 \text{ hours}}$.
  * *Day 6 (Week 1 Milestone Check):* $60 \text{ comparisons} = 150 \text{ min} = \mathbf{2.5 \text{ hours}}$ (Total Week 1 = 9.2 hrs).
  * *Week 2 (Held-Out Validation):* 240 paired evaluations + edge-case checks = **10.0 hours**.

---

### 4. Success Metric & Numeric Thresholds

* **Primary Tone Metric:** Mean Casualness Score on 5-point Likert scale (1=stiff/formal, 5=natural spoken).
  * **Threshold:** **$\ge 3.5 / 5.0$** on Hindi and Kannada evaluation sets (baseline formal output $\le 2.2 / 5.0$).
* **Fidelity Guardrail:** Factual preservation **$\ge 98.0\%$**; Safety pass rate **$100\%$**.
* **Unreviewed Languages Policy:** Without native speakers, conversational quality in Tamil, Telugu, Bengali, and Marathi **cannot be validated**. They will be gated behind an experimental flag or deferred until native reviewers are funded.

---

### 5. Feasible Kill Criterion

If by **Day 6**, reviewer casualness remains **$< 3.2 / 5.0$**, or if prompt persona induces safety refusals $>3\%$:
* **Kill Action:** **Retain the existing formal baseline for launch.** Do not attempt an unbudgeted emergency fine-tuning pipeline. Restrict conversational tone rollout only to validated languages.

---

### 6. Day-1 Experiment

* **Test Queries:** 40 prompts (20 Hindi, 20 Kannada) across greetings, advice, and product inquiries.
* **4-Arm Evaluation:** Arm 0 (Baseline) vs Arm 1 (Negative constraint: "Avoid textbook endings"), Arm 2 (Linguistic register: pronoun/ending rules), Arm 3 (Few-shot conversational exemplar).
* **Deliverable:** Generate 160 responses on Day 1; execute 120 paired comparisons with the reviewer on Day 2.
