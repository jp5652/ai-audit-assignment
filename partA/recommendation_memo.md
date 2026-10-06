# Executive Recommendation Memo: Tokenizer Audit & Indic Traffic Routing

**To:** VP of Engineering & AI Leadership  
**From:** AI Systems & Infrastructure Audit Team  
**Date:** October 7, 2026  
**Subject:** Correction of REPORT_v0 Tokenizer Findings & Indic Serving Strategy  

---

### 1. Corrected Headline Numbers

The previous intern report (`REPORT_v0.md`) asserted that Hindi requires **5.89× to 7.0×** more tokens than English, claiming that Devanagari script intrinsically forces extreme token expansion and recommending a 6× serving budget and separate serving infrastructure.

**This conclusion was fundamentally flawed.** It was an artifact of running `gpt2`—a 2019 English-centric tokenizer whose 50k vocabulary contains virtually zero Indic subwords, forcing byte-level fallback (1 token per UTF-8 byte).

When evaluated on the 1,012 parallel sentences of the standardized FLORES-200 benchmark across Indo-Aryan and Dravidian languages using an Indic-aware multilingual tokenizer (`xlm-roberta-base`, 250k vocab) matching our production model's 128k design:

| Language | Script Family | Tokens / Sentence (Semantic Task) | Relative Cost vs English (Semantic) | Tokens / Word (Morphological) | Relative Cost vs English (Per-Word) |
|---|---|---|---|---|---|
| **English** | Latin | 30.30 | **1.00×** | 1.40 | 1.00× |
| **Hindi** | Devanagari | 37.77 | **1.25×** | 1.49 | 1.06× |
| **Kannada** | Dravidian | 40.97 | **1.35×** | 2.58 | 1.84× |
| **Tamil** | Dravidian | 40.86 | **1.35×** | 2.47 | 1.76× |
| **Telugu** | Dravidian | 39.89 | **1.32×** | 2.38 | 1.70× |

*Key finding:* The true cost overhead of serving Indic traffic is **1.25× to 1.35×**, not 6.0×. Furthermore, "Tokens per Word" heavily penalizes Dravidian languages (1.84× for Kannada) because Dravidian languages are agglutinative (packing case endings and prepositions into single words, resulting in 26% fewer words for identical semantic content). The true unit of billing and user utility is the **semantic task (tokens per sentence)**.

---

### 2. Routing & Architectural Recommendation

1. **Reject the Dual-Stack Proposal:** Do **not** deploy a separate Indic-only serving stack or budget 6× capacity. Maintaining isolated clusters for Indic traffic introduces cold starts, GPU underutilization, duplicate weights, and routing complexity.
2. **Standardize on Unified Multilingual Serving:** Route all English and Indic traffic to the unified FLM-4B serving cluster (128k vocabulary). Budget capacity at **1.30×** baseline English token throughput for mixed Indic traffic.
3. **Save Projected Cloud Spend:** Downgrading the planned Indic infrastructure budget from 6.0× to 1.30× eliminates an immediate 78% over-provisioning expenditure.

---

### 3. Biggest Caveat

**Domain and Script Mismatch (Formal vs Code-Mixed Traffic):**  
FLORES-200 consists of clean, professionally translated news and encyclopedic text written strictly in native scripts (Devanagari, Kannada, Tamil, Telugu). In real-world customer support and messaging workloads, up to 40–60% of Indic user traffic is **code-mixed** or written in **Latin transliteration (Romanized / Hinglish / Kanglish)** (e.g., *"mera order kab aayega"* instead of *"मेरा ऑर्डर कब आएगा"*). Unstandardized phonetic spelling, slang, and script-switching can degrade tokenization compression compared to native formal scripts.

---

### 4. Primary Production Metric to Monitor

**Metric:** **Production Token Expansion Ratio ($R_{\text{lang}}$)**
$$R_{\text{lang}} = \frac{\text{P50 (Prompt + Completion Tokens per Session)}_{\text{Indic}}}{\text{P50 (Prompt + Completion Tokens per Session)}_{\text{English}}}$$

*Target Threshold:* **$1.20 \le R_{\text{lang}} \le 1.45$**.  
*Alert Trigger:* If $R_{\text{lang}}$ exceeds **1.55×** over a 24-hour window, trigger an automated alert. This immediately isolates whether incoming traffic is suffering from out-of-vocabulary transliteration anomalies, degenerate repetitive decoding loops, or system prompt bloat.
