# Part B: Serving Capacity Reconciliation & Anomaly Analysis

This document reconciles the serving hardware specifications from `bench/model_spec.md` against empirical load-test data in `bench/bench_log.csv`, identifies critical analytical flaws in `REPORT_v0.md`, and provides corrected mathematical derivations.

---

## B1. Theoretical KV-Cache Sizing & Concurrency Limit

### (a) Exact KV-Cache Bytes Per Token

From `bench/model_spec.md`:
* Model architecture: **FLM-4B-Instruct** (dense transformer)
* Layers ($L$): **28**
* Attention heads (Q): **24**
* Key-Value heads (GQA, $n_{kv}$): **8**
* Head dimension ($d_{head}$): **128**
* KV Cache Precision: **fp16** (2 bytes per scalar)

In a transformer with Grouped-Query Attention (GQA), each layer stores Key and Value tensors of size $n_{kv} \times d_{head}$ elements per token:
$$\text{KV bytes per layer per token} = 2 \times n_{kv} \times d_{head} \times \text{bytes\_per\_elem}$$
$$\text{KV bytes per layer per token} = 2 \times 8 \times 128 \times 2 = 4,096 \text{ bytes (4 KiB)}$$

Summing across all $L = 28$ layers:
$$\text{KV-cache bytes per token} = 28 \times 4,096 = \mathbf{114,688 \text{ bytes}} = \mathbf{112 \text{ KiB}}$$

---

### (b) Approximate Maximum Concurrent 4096-Token Sequences

From hardware and serving constraints:
* **GPU RAM**: 1× NVIDIA L4 = **24 GB** ($24.0 \times 10^9$ bytes decimal) or **24 GiB** ($25.77 \times 10^9$ bytes binary).
* **Usable Memory Ceiling**: `gpu_memory_utilization = 0.92`.
  $$\text{Memory}_{\text{usable}} = 0.92 \times 24.0 \text{ GB} = \mathbf{22.08 \text{ GB}}$$
* **Static Deductions**:
  1. *Model Weights*: 4.2B parameters in fp16 = $4.2 \times 10^9 \times 2 = \mathbf{8.40 \text{ GB}}$.
  2. *Runtime Overhead* (activations, CUDA graphs, workspace): $\mathbf{1.60 \text{ GB}}$.
* **Available Memory for KV Cache Pool**:
  $$\text{Memory}_{\text{KV}} = 22.08 \text{ GB} - 8.40 \text{ GB} - 1.60 \text{ GB} = \mathbf{12.08 \text{ GB}} = 12,080,000,000 \text{ bytes}$$
  *(On a binary 24 GiB / 25.77 GB physical basis: $\text{Memory}_{\text{KV}} = 23.708 - 8.40 - 1.60 = \mathbf{13.71 \text{ GB}}$)*

**Total Token Capacity in KV Pool**:
$$N_{\text{max\_tokens}} = \frac{12,080,000,000 \text{ bytes}}{114,688 \text{ bytes/token}} = \mathbf{105,329 \text{ tokens}} \quad (\text{or } 119,526 \text{ tokens on binary basis})$$

**Maximum Concurrent 4096-Token Sequences**:
$$\text{Max Concurrent Sequences} = \frac{105,329}{4096} = \mathbf{25.71 \approx 25 \text{ sequences}}$$
*(Binary basis: $119,526 / 4096 = 29.18 \approx 29$ sequences)*

### Verification Against `bench_log.csv`

Inspecting the long-context test sweep (`prompt_len = 3584`, `gen_len = 512`, Total = 4096 tokens per request):

| Batch Size | KV Cache Util | Preempted Seqs | Log Interpretation & Consistency |
|---|---|---|---|
| **4** | 0.16 | 0 | Un-preempted execution ($4 / 0.16 \approx 25$ capacity) |
| **8** | 0.31 | 0 | Un-preempted execution ($8 / 0.31 \approx 25.8$ capacity) |
| **16** | 0.62 | 0 | Un-preempted execution ($16 / 0.62 \approx 25.8$ capacity) |
| **24** | 0.93 | 0 | Un-preempted execution ($24 / 0.93 \approx 25.8$ capacity) |
| **32** | 0.97 | 7 | Cache saturated; 7 sequences preempted at least once |
| **48** | 0.97 | 23 | Cache saturated; 23 sequences preempted at least once |

**Conclusion:** The benchmark log is consistent with an approximate physical capacity ceiling of **~25 concurrent sequences**. Note that `preempted_seqs` records sequences preempted at least once during the run; it does not prove that exactly 25 sequences were simultaneously resident at every instant.

---

## B2. Long-Context Sweep Throughput Anomaly & Root Mechanism

### 1. Empirical Anomaly

In naive scaling models, throughput is expected to increase with batch size until saturating compute or bandwidth. In `bench_log.csv`:
* **Batch 4 $\to$ 24**: `reported_tok_s` scales smoothly from 565.4 tok/s up to **1607.4 tok/s** at batch 24.
* **Batch 24 $\to$ 32**: Throughput suddenly **collapses by 13.9%** from 1607.4 down to **1384.0 tok/s**, while wall-clock time balloons from 61.16s to 94.71s (+54.9%).
* **Batch 32 $\to$ 48**: Throughput degrades further to **1298.5 tok/s** (-19.2% vs batch 24), and wall-clock time surges to 151.41s (+147.6%).
* **Latency explosion**: `e2e_ms_p95` surges from 69.2s (batch 24) to 97.5s (batch 32) and 105.4s (batch 48). Median TTFT nearly doubles from 500.5ms to 955.4ms.

### 2. Physical Mechanism

* **Row 12 (`batch_size=24`)**: Peak `kv_cache_util` reaches **0.93**, with `preempted_seqs = 0`. All 24 sequences fit within the available physical memory budget.
* **Row 13 (`batch_size=32`) & Row 14 (`batch_size=48`)**: `kv_cache_util` pegs at **0.97**, exhausting the block allocator watermark.
* When 32 requests are admitted simultaneously, the physical KV cache runs out of free memory blocks during iterative decoding. The scheduler is forced to preempt sequences: 7 sequences at batch 32, and 23 sequences at batch 48.
* In vLLM/PagedAttention recomputation mode, preempted sequences are discarded from the GPU. When memory frees up, the preempted sequences must undergo **re-prefill**—recomputing all 3,584 prompt tokens from scratch!
* This induces **severe memory thrashing**: GPU FLOPs are wasted repeatedly re-evaluating prompt tokens rather than generating new tokens, causing throughput collapse and catastrophic tail latency.

### 3. Proposed Fix & Predicted Quantitative Effect

* **Deployment Change**: Configure the serving engine with `--max-num-seqs 24` (or enforce request admission concurrency limits of 24 in-flight requests per L4 replica at the API gateway).
* **Predicted Quantitative Effect (Model-Based Prediction):**
  1. *Preemption Elimination*: `preempted_seqs = 0` at all load levels.
  2. *Throughput Preservation*: High-load throughput remains at the optimal **~1,607 tok/s** instead of collapsing to 1,298 tok/s (**+23.8% throughput improvement**).
  3. *Estimated Wall-Clock Time for Batch 48*: Under the assumption of two successive non-preempted waves of 24:
     $$\text{Predicted Wall-Clock Time} = 2 \times 61.16\text{ s} = \mathbf{122.32 \text{ s}}$$
     Compared to the observed 151.41s, this represents a predicted **29.09s (19.2%) reduction** in total run duration. *(Note: This is an idealized two-wave prediction with assumptions, not an unconditional guarantee).*

---

## B3. Flawed Extrapolation in `REPORT_v0.md` & Honest Goodput Derivation

### 1. The Column Misread

Section 2 of `REPORT_v0.md` claims:
> *"at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization... assume ~1600 tok/s per L4 and scale linearly with batch size, so batch 48 should give us ~3200 tok/s."*

The author misread the column **`reported_tok_s`** as **generation goodput** (output tokens produced per second).

`reported_tok_s` measures **Total Throughput** $(\text{Prompt Tokens} + \text{Generation Tokens}) / \text{wall\_clock\_s}$:
* For batch 24 long-prompt:
  $$\text{Total Tokens} = 24 \times (3584 + 512) = 98,304 \text{ tokens}$$
  $$\frac{98,304 \text{ tokens}}{61.16 \text{ s}} = \mathbf{1607.33 \text{ tok/s}} \quad (\text{recorded as } 1607.4)$$
* Prompt tokens account for **87.5%** ($\frac{3584}{4096}$) of this count. Autoregressive generation represents only **12.5%** ($\frac{512}{4096}$) of the token count.
* Linear extrapolation to batch 48 was impossible because KV cache memory was already near saturation at batch 24 (0.93 utilization).

---

### 2. Honest Goodput Derivation (Batch 24, Long-Prompt Row)

Total generated tokens: $N_{\text{gen}} = 24 \times 512 = \mathbf{12,288 \text{ tokens}}$.

#### Method 1: Generated Tokens Divided by Wall-Clock Time
$$\text{Goodput}_{\text{Method 1}} = \frac{24 \times 512}{61.16 \text{ s}} = \frac{12,288}{61.16} = \mathbf{200.916 \text{ output tok/s}}$$

#### Method 2: Reported Throughput Multiplied by Output Token Fraction
$$\text{Goodput}_{\text{Method 2}} = \text{reported\_tok\_s} \times \frac{\text{gen\_len}}{\text{prompt\_len} + \text{gen\_len}} = 1607.4 \times \frac{512}{4096} = 1607.4 \times 0.125 = \mathbf{200.925 \text{ output tok/s}}$$

**Agreement:** Both independent derivations agree that honest output goodput is **~200.92 output tok/s**. The minor 0.009 tok/s difference is consistent with rounding of recorded fields in the harness log (such as duration or reported throughput). Real output goodput is **~8× lower** than the reported 1607.4 tok/s!

---

### 3. Matched-Batch Comparison: Do Longer Prompts Give Better Throughput?

Comparing short vs long prompts at the **same batch size of 16**:
* **Short Prompt (Row 6, batch 16, prompt 512, gen 256, wall 13.91s):**
  $$\text{Output Goodput} = \frac{16 \times 256}{13.91} = \mathbf{294.46 \text{ output tok/s}}$$
* **Long Prompt (Row 11, batch 16, prompt 3584, gen 512, wall 49.97s):**
  $$\text{Output Goodput} = \frac{16 \times 512}{49.97} = \mathbf{163.94 \text{ output tok/s}}$$
* **Observed Difference:**
  $$\frac{294.46 - 163.94}{294.46} = \mathbf{44.3\% \text{ reduction in output goodput}}$$

Comparing these rows shows a **44.3% drop in output goodput** at batch 16. Note that between Row 6 and Row 11, both prompt length (512 to 3584) and generation length (256 to 512) change. Bandwidth pressure from scanning larger KV caches during decode and prefill overhead are plausible mechanisms, which serving telemetry would verify.

---

### 4. Important Caveat: End-to-End Latency vs Wall-Clock Time

Twelve of thirteen rows have p95 latency exceeding the recorded run duration. The supplied files do not explain the measurement boundaries. Client/server timing differences are one hypothesis; harness instrumentation is needed to resolve this. Avoid treating client latency percentiles and server wall-clock as perfectly reconciled.

---

## B4. Production Verification Metric

To confirm the memory exhaustion and preemption mechanism in production, pull the single Prometheus metric:
$$\mathbf{vllm:num\_preemptions\_total}$$

**Expected behavior:**
* In vLLM, `num_preemptions_total` is a cumulative counter of **preemption events** (increments each time an active sequence is preempted).
* For batch sizes **$\le 24$**: The counter's increase ($\Delta$) during the run should be **zero**.
* At batch **32**: Preemptions begin occurring (with the counter increasing by at least 7 under the assumed serving implementation).
* At batch **48**: The counter increases further (by at least 23 events).
