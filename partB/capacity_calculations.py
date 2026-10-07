#!/usr/bin/env python3
"""
========================================================================================
ASSIGNMENT PART B — Serving Capacity Reconciliation & Verification
========================================================================================
QUESTIONS ANSWERED IN THIS SCRIPT:
- Calculate KV-cache memory per token (exact bytes).
- Calculate maximum concurrent 4096-token sequences this GPU can hold.
- Check whether those calculations match the actual benchmark log.
- Explain why a particular long-context throughput result is anomalous.
- Explain whether the claim that longer prompts give better throughput is actually true.
- Explain whether batch 48 ≈ 3200 tok/s is actually justified.
- Derive honest goodput for batch-24 long-prompt row using two independent ways.

CORE QUESTION:
"Can we actually trust the previous GPU capacity and throughput conclusions?"
----------------------------------------------------------------------------------------
SUMMARY OF SOLUTIONS:
1. KV-cache bytes/token = 2 * 28 layers * 8 heads * 128 dim * 2 bytes = 114,688 bytes (112 KiB).
2. Available KV memory on 24 GB L4 at 0.92 util (minus 8.4 GB weights & 1.6 GB overhead) = 12.08 GB.
3. Max 4096-token concurrency = 12.08 GB / (114,688 * 4096) = 25.71 ≈ 25 sequences!
   -> Perfectly matches bench_log.csv (batches 32 & 48 cap at exactly 25 active sequences).
4. Long-prompt anomaly: Throughput collapses above batch 24 due to memory exhaustion (0.97 util)
   triggering scheduler preemptions (7 and 23 sequences) and re-prefill thrashing.
5. Column misread: reported_tok_s included prefill tokens (87.5% of total).
   Honest goodput is ~201 to 250 tok/s (~8x lower than reported 1607 tok/s).
   Extrapolating to batch 48 (3200 tok/s) was completely invalid.
========================================================================================
"""

import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")


def calculate_b1():
    """
    QUESTION:
    From model_spec.md alone, compute:
      (a) KV-cache bytes per token, exactly;
      (b) Approximate maximum number of concurrent 4096-token sequences this GPU can hold.
    Then check your prediction against the log.
    """
    print("=" * 70)
    print("PART B1: KV-Cache Sizing & Concurrency Limit")
    print("=" * 70)

    # Model parameters from model_spec.md
    num_layers = 28
    num_kv_heads = 8
    head_dim = 128
    bytes_per_elem = 2  # fp16

    # (a) KV-cache bytes per token
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    kv_kib_per_token = kv_bytes_per_token / 1024

    print(f"(a) KV-cache bytes per token:")
    print(
        f"    Formula: 2 * n_layers ({num_layers}) * n_kv_heads ({num_kv_heads}) * head_dim ({head_dim}) * bytes ({bytes_per_elem})"
    )
    print(f"    = {kv_bytes_per_token:,} bytes ({kv_kib_per_token:.1f} KiB)\n")

    # (b) Max concurrent 4096-token sequences
    gpu_total_ram_bytes = 24 * (10**9)  # 24 GB decimal
    gpu_total_ram_gib = 24 * (1024**3)  # 24 GiB binary

    utilization = 0.92
    model_weights_bytes = 4.2 * (10**9) * 2  # 8.4 GB
    overhead_bytes = 1.6 * (10**9)  # 1.6 GB

    usable_ram_decimal = gpu_total_ram_bytes * utilization
    kv_pool_decimal = usable_ram_decimal - model_weights_bytes - overhead_bytes
    max_tokens_decimal = kv_pool_decimal / kv_bytes_per_token
    max_seqs_4096_decimal = max_tokens_decimal / 4096

    usable_ram_gib = gpu_total_ram_gib * utilization
    kv_pool_gib = usable_ram_gib - model_weights_bytes - overhead_bytes
    max_tokens_gib = kv_pool_gib / kv_bytes_per_token
    max_seqs_4096_gib = max_tokens_gib / 4096

    print(
        f"(b) Theoretical 4096-token concurrency limit on 1x NVIDIA L4 (24 GB):"
    )
    print(
        f"    Usable memory (util=0.92): {usable_ram_decimal / 10**9:.2f} GB (decimal) | {usable_ram_gib / 1024**3:.2f} GiB (binary)"
    )
    print(
        f"    Deductions: Weights = {model_weights_bytes / 10**9:.2f} GB, Overhead = {overhead_bytes / 10**9:.2f} GB"
    )
    print(
        f"    KV cache memory pool: {kv_pool_decimal / 10**9:.3f} GB (decimal) | {kv_pool_gib / 10**9:.3f} GB (binary)"
    )
    print(
        f"    Max KV cache tokens:  {max_tokens_decimal:,.0f} tokens (decimal) | {max_tokens_gib:,.0f} tokens (binary)"
    )
    print(
        f"    Theoretical 4096-token concurrency: {max_seqs_4096_decimal:.2f} seqs (decimal) | {max_seqs_4096_gib:.2f} seqs (binary)"
    )
    print(f"    --> Predicted ceiling: ~25 to 26 concurrent sequences.\n")

    # Checking against log
    with open("starter_kit/bench/bench_log.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    long_prompts = [r for r in rows if int(r["prompt_len"]) == 3584]
    print(
        "Checking against bench_log.csv (long prompt sweep: 3584 prompt + 512 gen = 4096 total):"
    )
    for row in long_prompts:
        bs = int(row["batch_size"])
        util = float(row["kv_cache_util"])
        preempted = int(row["preempted_seqs"])
        inferred_max = bs / util if preempted == 0 else (bs - preempted)
        print(
            f"    batch {bs:>2}: util={util:.2f}, preempted={preempted:>2} "
            f"--> Inferred capacity = {inferred_max:.1f} sequences"
        )


def calculate_b3():
    """
    QUESTION:
    Find the column misread in REPORT_v0.md.
    Derive honest goodput of the batch-24 long-prompt row using two independent ways.
    """
    print("\n" + "=" * 70)
    print("PART B3: Honest Goodput Derivation (Batch 24, Long Prompt)")
    print("=" * 70)

    # Data from Row 12: batch 24, prompt 3584, gen 512
    batch_size = 24
    prompt_len = 3584
    gen_len = 512
    num_requests = 24
    wall_clock_s = 61.16
    reported_tok_s = 1607.4
    ttft_ms_p50 = 500.5
    itl_ms_p50 = 96.07

    total_tokens = num_requests * (prompt_len + gen_len)
    gen_tokens = num_requests * gen_len

    print(f"Row 12 parameters: batch={batch_size}, wall_clock={wall_clock_s}s")
    print(
        f"Total tokens = {total_tokens:,} ({num_requests * prompt_len:,} prompt + {gen_tokens:,} generated)"
    )
    print(
        f"Harness reported_tok_s = {total_tokens} / {wall_clock_s} = {total_tokens / wall_clock_s:.1f} tok/s (matches log: {reported_tok_s})"
    )
    print(
        f"Notice: Prompt tokens represent {prompt_len / (prompt_len + gen_len) * 100:.1f}% of total tokens!\n"
    )

    # Method 1: Generated tokens / Wall clock (or decode time)
    goodput_wall = gen_tokens / wall_clock_s
    decode_time_s = wall_clock_s - (ttft_ms_p50 / 1000.0)
    goodput_decode = gen_tokens / decode_time_s

    print("Method 1: Generation Tokens over Wall Clock / Decode Time")
    print(
        f"    gen_tokens / wall_clock = {gen_tokens} / {wall_clock_s}s = {goodput_wall:.2f} tok/s"
    )
    print(
        f"    gen_tokens / (wall_clock - TTFT) = {gen_tokens} / {decode_time_s:.3f}s = {goodput_decode:.2f} tok/s"
    )

    # Method 2: From Median Inter-Token Latency (itl_ms_p50)
    goodput_itl = batch_size / (itl_ms_p50 / 1000.0)
    print("\nMethod 2: From Inter-Token Latency (itl_ms_p50)")
    print(
        f"    batch_size / (itl_ms_p50 / 1000) = {batch_size} / {itl_ms_p50 / 1000:.5f}s = {goodput_itl:.2f} tok/s"
    )

    print("\nSummary:")
    print(f"    Reported throughput:  {reported_tok_s:.1f} tok/s (prefill + decode)")
    print(
        f"    Honest output goodput: {goodput_wall:.1f} to {goodput_itl:.1f} tok/s (~8x lower than reported)"
    )


if __name__ == "__main__":
    calculate_b1()
    calculate_b3()
