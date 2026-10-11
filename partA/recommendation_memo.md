# Executive Recommendation Memo: Multilingual Tokenizer Audit

**To:** VP of Engineering & AI Leadership | **From:** AI Systems & Infrastructure Audit Team | **Date:** October 11, 2026  
**Subject:** Audit of REPORT_v0 Tokenizer Findings & Indic Serving Strategy

### 1. Corrected Headline Numbers
The previous report (`REPORT_v0.md`) asserted that Hindi requires **5.89× to 7.0×** more tokens than English, claiming that Devanagari script intrinsically forces extreme token expansion, and recommended budgeting 6× serving costs.

**This conclusion is invalid.** It was an artifact of evaluating `gpt2`—an English-centric tokenizer with virtually zero Indic subwords, forcing byte fallback. When evaluated on 1,012 parallel sentences from the FLORES-200 benchmark across Indo-Aryan and Dravidian languages, token expansion depends directly on tokenizer architecture:

| Tokenizer | Vocab Size | English | Hindi | Kannada | Tamil | Telugu |
|---|---|---|---|---|---|---|
| **gpt2** (English baseline) | 50k | 1.00× | 7.42× | 13.58× | 15.54× | 12.97× |
| **Qwen2.5** (Modern BPE) | 152k | 1.00× | 4.42× | 6.92× | 6.11× | 6.99× |
| **xlm-roberta-base** (Multilingual) | 250k | 1.00× | **1.25×** | **1.35×** | **1.35×** | **1.32×** |

*(Ratios computed on Tokens per Parallel Sentence, holding semantic content constant).*

*Key Finding on Denominators:* "Tokens per Word" heavily distorts Dravidian languages (e.g. Kannada appears 1.84× under XLM-R) because Dravidian languages are agglutinative—they attach case markers and prepositions directly onto words, resulting in 26.5% fewer words for identical semantic meaning. The proper metric holding meaning constant across languages is **Tokens per Parallel Sentence**.

### 2. Strategic Routing Recommendation
1. **Reject Universal 6× Provisioning:** The claim that Indic text intrinsically demands 6× serving capacity is disproven by XLM-R's 1.25–1.35× ratio.
2. **Conditional Capacity Planning:** Multilingual vocabulary allocation dictates token expansion. While XLM-R achieves 1.25–1.35×, Qwen2.5 remains at 4.42–6.99×. Because FLM-4B's proprietary 128k tokenizer was not provided in this audit kit, leadership must **directly benchmark FLM-4B's actual tokenizer on matched tasks before finalizing production capacity**.
3. **Architecture:** Avoid maintaining separate, fragmented single-language clusters unless FLM-4B exhibits severe token fragmentation.

### 3. Primary Caveat
**Domain & Register Distribution Mismatch:** FLORES-200 comprises clean, professionally translated formal news text in standard native orthography. It does not reflect colloquial conversational chat, user typos, or Latin-transliterated text (e.g., Romanized Hinglish).

### 4. Production Metric to Monitor
**Metric:** **Task-Conditioned Token Expansion Ratio ($R_{\text{task, lang}}$)**
$$R_{\text{task, lang}} = \frac{\text{P50 (Prompt + Completion Tokens)}_{\text{Indic, task}}}{\text{P50 (Prompt + Completion Tokens)}_{\text{English, task}}}$$
Tracked for fixed model/tokenizer versions, partitioned by intent (e.g., customer support, summarization). Establish an empirical baseline ($R_{\text{baseline}}$) during offline tokenizer profiling. Alert if $R_{\text{task, lang}}$ drifts **>20% above the validated baseline** for matched tasks—a provisional operational choice to catch vocabulary mismatches, repetitive decoding loops, or system prompt bloat without assuming an ungrounded global multiplier.
