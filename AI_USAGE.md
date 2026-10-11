# AI Usage Summary (`AI_USAGE.md`)

This document provides a transparent, unsparing account of where AI tools assisted our engineering work, where they misled us with confident hallucinations, and how empirical verification caught and corrected each flaw.

---

### 1. Where AI Assisted Effectively

* **Data Engineering & Benchmark Automation:**
  * Accelerated the authoring of `partA/prepare_corpus.py` to extract and verify 1,012 parallel sentences from Meta's FLORES-200 archive across 5 languages.
  * Rapidly wrote the multi-tokenizer evaluation matrix (`run_cross_lang_bench.py`) across `gpt2`, `xlm-roberta-base`, and `Qwen/Qwen2.5-0.5B` across 4 denominators.
* **Algebraic Scaffolding:**
  * Drafted transformer KV-cache dimensional formulas and LaTeX formatting in Part B.

---

### 2. Where AI Misled or Hallucinated (and How Verification Caught It)

#### Incident 1: Hallucinating a 50–100% Token Explosion on NFC Normalization
* *What AI Suggested:* The AI drafted an explanation claiming that decomposed NFD text *"increases tokenization fragments by up to 50–100% in Indic NLP"*.
* *How it was Caught:* When looking directly at the printed experiment output on Devanagari words (`फ़िल्म`, `ज़रूरत`, `क़िला`, `गाड़ी`), the numbers showed **identical bytes (18 vs 18) and identical tokens (12 vs 12)** across NFC and NFD (delta = 0). The AI had confidently stated a theoretical trope that directly contradicted the measured output. Under standard Unicode, Devanagari nukta consonants have composition exclusions; Python's `unicodedata` maintains the decomposed representation under both NFC and NFD.
* *Correction:* Removed the fabricated 50–100% claim. Reported the zero delta honestly, confirming NFC normalization as a benign defensive feature, and relied on `add_special_tokens=False` (+21.4% measured distortion) as our primary defensible "suspicious but fine" finding.

#### Incident 2: Erroneous KV Memory Math (Dividing by 4096)
* *What AI Suggested:* In Part C, the AI calculated prompt-engineering KV cache memory overhead as `35 tokens * 112 KiB / 4096 = 0.96 KiB per sequence`.
* *How it was Caught:* Dividing by 4096 was mathematically nonsensical—KV cache is 112 KiB *per token*, not per 4096-token sequence! The true overhead is $35 \times 112 \text{ KiB} = \mathbf{3,920 \text{ KiB} = 3.83 \text{ MiB per sequence}}$, or $\mathbf{91.88 \text{ MiB}}$ across 24 concurrent requests. Furthermore, adding 35 prompt tokens to a 3,584-token prompt generating 512 tokens totals 4,131 tokens, exceeding the 4,096-token context limit unless prompt history or generation length is trimmed.
* *Correction:* Corrected the arithmetic and documented the context budget tradeoff.

#### Incident 3: Conflating Median ITL with Whole-Run Goodput
* *What AI Suggested:* The AI proposed deriving goodput by dividing batch size by median inter-token latency ($24 / 0.09607\text{s} = 249.8 \text{ tok/s}$), reporting goodput as a wide range of "201 to 250 tok/s".
* *How it was Caught:* Reviewing benchmark mechanics revealed that `batch_size / itl_ms_p50` measures instantaneous step rate during the decode phase, not whole-run throughput. The two agreeing whole-run methods are:
  1. $N_{\text{gen}} / T_{\text{wall}} = (24 \times 512) / 61.16 = \mathbf{200.916 \text{ output tok/s}}$
  2. $\text{reported\_tok\_s} \times (\text{gen\_len} / \text{total\_len}) = 1607.4 \times (512 / 4096) = \mathbf{200.925 \text{ output tok/s}}$
* *Correction:* Unified the derivation around the two agreeing whole-run methods (~200.92 output tok/s).

#### Incident 4: Proposing an Unbudgeted SFT Fallback in Part C
* *What AI Suggested:* The AI proposed a Day-7 fallback to train a LoRA adapter on 300 human-curated examples.
* *How it was Caught:* Arithmetic check on reviewer capacity: 1 reviewer working 10 h/week has only 240 review slots in Week 2. Curating 300 examples from scratch and validating them exceeds total remaining hours.
* *Correction:* Replaced the unrealistic fallback with a feasible operational kill criterion: retain the formal baseline for launch and gate casual tone to validated languages.

#### Incident 5: Over-Generalizing XLM-R Ratios to FLM-4B Production Costs
* *What AI Suggested:* The AI initially asserted that XLM-R's 1.25–1.35× ratio proved that FLM-4B serving costs could be budgeted at 1.30× with 78% infrastructure savings.
* *How it was Caught:* Our own benchmark on Qwen2.5 showed token ratios of 4.42–6.99× despite having a 152k vocabulary. Because FLM-4B's proprietary tokenizer was not provided in the kit, assuming XLM-R's efficiency was an unjustified transfer.
* *Correction:* Reframed the recommendation as conditional on direct tokenizer profiling of FLM-4B.
