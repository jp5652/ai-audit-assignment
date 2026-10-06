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

In a transformer with Grouped-Query Attention (GQA), each layer stores Key and Value vectors of size $n_{kv} \times d_{head}$ elements per token:
$$\text{KV bytes per layer per token} = 2 \times n_{kv} \times d_{head} \times \text{bytes\_per\_elem}$$
$$\text{KV bytes per layer per token} = 2 \times 8 \times 128 \times 2 = 4,096 \text{ bytes (4 KiB)}$$

Summing across all $L = 28$ layers:
$$\text{KV-cache bytes per token} = L \times 4,096 = 28 \times 4,096 = \mathbf{114,688 \text{ bytes}} = \mathbf{112 \text{ KiB}}$$

---

### (b) Approximate Maximum Concurrent 4096-Token Sequences

From hardware and serving constraints:
* **GPU RAM**: 1× NVIDIA L4 = **24 GB** ($24 \times 10^9$ bytes) or **24 GiB** ($25.77 \times 10^9$ bytes).
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
*(Binary upper bound: $119,526 / 4096 = 29.18 \approx 29$ sequences)*

### Verification Against `bench_log.csv`

Inspecting the long-context test sweep (`prompt_len = 3584`, `gen_len = 512`, Total = 4096 tokens per request):

| Batch Size | KV Cache Util | Preempted Seqs | Inferred Maximum Concurrent Capacity |
|---|---|---|---|
| **4** | 0.16 | 0 | $4 / 0.16 = \mathbf{25.0}$ |
| **8** | 0.31 | 0 | $8 / 0.31 = \mathbf{25.8}$ |
| **16** | 0.62 | 0 | $16 / 0.62 = \mathbf{25.8}$ |
| **24** | 0.93 | 0 | $24 / 0.93 = \mathbf{25.8}$ |
| **32** | 0.97 | 7 | Active = $32 - 7 = \mathbf{25.0}$ |
| **48** | 0.97 | 23 | Active = $48 - 23 = \mathbf{25.0}$ |

**Conclusion:** The mathematical model predicting a hard ceiling of **25 to 26 concurrent sequences** is perfectly validated by empirical server metrics.

---

## B2. Long-Context Sweep Throughput Anomaly & Root Mechanism

### 1. Empirical Anomaly

In naive scaling models, system throughput increases monotonically with batch size until saturating compute or memory bandwidth. In `bench_log.csv`:
* **Batch 4 $\to$ 24**: `reported_tok_s` scales smoothly from 565.4 tok/s up to **1607.4 tok/s** at batch 24.
* **Batch 24 $\to$ 32**: Throughput suddenly **collapses by 13.9%** from 1607.4 down to **1384.0 tok/s**, while wall-clock time balloons from 61.16s to 94.71s (+54.9%).
* **Batch 32 $\to$ 48**: Throughput degrades further to **1298.5 tok/s** (-19.2% vs batch 24), and wall-clock time surges to 151.41s (+147.6%).
* **Latency explosion**: `e2e_ms_p95` surges from 69.2s (batch 24) to 97.5s (batch 32) and 105.4s (batch 48). Median TTFT nearly doubles from 500.5ms to 955.4ms.

### 2. Physical Mechanism

* **Row 12 (`batch_size=24`)**: Peak `kv_cache_util` reaches **0.93**, with `preempted_seqs = 0`. All 24 sequences fit within the available physical memory budget.
* **Row 13 (`batch_size=32`) & Row 14 (`batch_size=48`)**: `kv_cache_util` pegs at **0.97**, exhausting the block allocator watermark.
* When 32 requests are admitted simultaneously, the physical KV cache runs out of free memory blocks during iterative decoding. The scheduler is forced to **preempt active sequences**:
  * At batch 32: `preempted_seqs = 7` ($32 - 7 = 25$ held).
  * At batch 48: `preempted_seqs = 23` ($48 - 23 = 25$ held).
* Under vLLM/PagedAttention recomputation mode, preempted sequences are discarded from the GPU. Once active sequences finish, the preempted sequences must undergo **re-prefill**—recomputing all 3,584 prompt tokens from scratch!
* This induces **severe memory thrashing**: GPU FLOPs are wasted repeatedly re-evaluating prompt tokens rather than generating new tokens, causing throughput collapse and catastrophic tail latency.

### 3. Proposed Fix & Predicted Quantitative Effect

* **Deployment Change**: Configure the serving engine with `--max-num-seqs 24` (or set the API gateway reverse proxy connection limit to a maximum of 24 concurrent in-flight requests per L4 instance).
* **Predicted Quantitative Effect**:
  1. *Preemption Elimination*: `preempted_seqs = 0` at all load levels.
  2. *Throughput Preservation*: High-load throughput remains at the optimal **~1,607 tok/s** instead of collapsing to 1,298 tok/s (**+23.8% throughput improvement**).
  3. *Wall-Clock Speedup*: A batch of 48 requests will be processed cleanly as two consecutive waves of 24:
     $$\text{Predicted Wall-Clock Time} = 2 \times 61.16\text{ s} = \mathbf{122.32 \text{ s}}$$
     Compared to the unconstrained 151.41s, this reduces total execution time by **29.09 seconds (19.2% faster)** while capping request p95 latency at 69.2s (first wave) and ~122s (second wave) instead of 105.4s across all requests.

---

## B3. Flawed Extrapolation in `REPORT_v0.md` & Honest Goodput Derivation

### 1. The Column Misread

Section 2 of `REPORT_v0.md` claims:
> *"at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization... assume ~1600 tok/s per L4 and scale linearly with batch size, so batch 48 should give us ~3200 tok/s."*

The author misread the column **`reported_tok_s`** as **generation goodput** (output tokens produced per second).

`reported_tok_s` measures **Total Throughput** $(\text{Prompt Tokens} + \text{Generation Tokens}) / \text{wall\_clock\_s}$:
* For batch 24 long-prompt:
  $$\text{Total Tokens} = 24 \times (3584 + 512) = 98,304 \text{ tokens}$$
  $$\frac{98,304 \text{ tokens}}{61.16 \text{ s}} = \mathbf{1607.32 \text{ tok/s}} \quad (\text{exactly matches } 1607.4)$$
* Prompt tokens account for **87.5%** ($\frac{3584}{4096}$) of this count. They were processed in parallel during the compute-bound prefill phase in the first 500 ms!
* Generation tokens account for only **12.5%** ($\frac{512}{4096}$), emitted autoregressively over 60.6 seconds.
* Linear extrapolation to batch 48 was impossible because KV cache memory was already exhausted at batch 24.

---

### 2. Honest Goodput Derivation (Batch 24, Long-Prompt Row)

Total generated tokens: $N_{\text{gen}} = 24 \times 512 = \mathbf{12,288 \text{ tokens}}$.

#### Method 1: Generation Tokens over Wall-Clock / Decode Time
* End-to-end wall clock:
  $$\text{Goodput}_{\text{wall}} = \frac{12,288 \text{ tokens}}{61.16 \text{ s}} = \mathbf{200.92 \text{ tok/s}}$$
* Excluding initial prefill phase ($T_{\text{decode}} = \text{wall\_clock} - \text{TTFT} = 61.16\text{s} - 0.5005\text{s} = 60.6595\text{s}$):
  $$\text{Goodput}_{\text{decode}} = \frac{12,288 \text{ tokens}}{60.6595 \text{ s}} = \mathbf{202.57 \text{ tok/s}}$$

#### Method 2: From Median Inter-Token Latency (`itl_ms_p50`)
During decode, the GPU runs 24 parallel streams in lockstep, emitting 24 tokens per decode step. Each step takes `itl_ms_p50` = $96.07 \text{ ms} = 0.09607 \text{ s}$:
$$\text{Goodput}_{\text{ITL}} = \frac{\text{batch\_size}}{\text{itl\_ms\_p50} / 1000} = \frac{24 \text{ tokens}}{0.09607 \text{ s}} = \mathbf{249.82 \text{ tok/s}}$$
*(The slight variance between ~202 tok/s and ~250 tok/s reflects ITL growth across sequence generation as KV cache lengths expand).*

**Summary:** Honest generation goodput is **~201 to 250 tok/s**—roughly **8× lower** than the reported 1607.4 tok/s!

### 3. What the Report Should Have Said

> *"For long-context requests (prompt 3584, gen 512), sustained generation goodput is only **~201–250 output tok/s**, not 1600 tok/s. Long prompts reduce generation throughput by over 70% compared to short prompts because reading large KV caches at each decode step saturates GPU memory bandwidth (300 GB/s on L4). Furthermore, memory capacity hard-caps concurrent batch size at 24 sequences (93% KV utilization). Under no circumstances will batch 48 deliver 3200 tok/s; without concurrency controls, batch 48 causes severe preemption thrashing and reduces throughput to 1298 total tok/s."*

---

## B4. Production Verification Metric

To confirm the memory exhaustion and preemption thrashing mechanism in production, pull the Prometheus metric:
$$\mathbf{vllm:num\_preemptions\_total}$$
*(alongside `vllm:iteration_tokens_total{status="recompute"}`).*

**Expected values across load sweeps:**
* At batch sizes **$\le 24$**: The counter remains at **0**, while `vllm:gpu_cache_usage_factor` scales linearly up to **0.93**.
* At batch **32**: `vllm:num_preemptions_total` increments by **7**, and `gpu_cache_usage_factor` pegs at **0.97**.
* At batch **48**: `vllm:num_preemptions_total` increments by **23**.

Observing non-zero values on this counter simultaneously with a plateau at 0.97 in GPU cache utilization confirms that memory exhaustion induced scheduler preemption and re-prefill thrashing.
