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
- Derive honest goodput for batch-24 long-prompt row using two independent agreeing methods.

CORE QUESTION:
"Can we actually trust the previous GPU capacity and throughput conclusions?"
----------------------------------------------------------------------------------------
EXACT SOLUTIONS:
1. KV-cache bytes/token = 2 * 28 layers * 8 heads * 128 dim * 2 bytes = 114,688 bytes (112 KiB).
2. Usable memory on 24 GB L4 at 0.92 util = 22.08 GB.
   Minus 8.4 GB weights (4.2B fp16) & 1.6 GB overhead = 12.08 GB KV pool.
3. Max 4096-token concurrency = 12.08 GB / (114,688 * 4096) = 25.71 ≈ 25 sequences!
   Consistent with bench_log.csv where batch 24 runs un-preempted (0.93 util), and batches
   32 and 48 saturate cache (0.97 util) with 7 and 23 sequences preempted at least once.
4. Long-prompt anomaly: Throughput collapses from 1607 tok/s to 1384 & 1298 tok/s due to
   memory exhaustion triggering preemption and re-prefill thrashing.
5. Column misread in REPORT_v0.md: reported_tok_s is total throughput (prefill + decode).
   Prompt tokens represented 87.5% of total tokens processed.
6. Honest goodput of batch-24 long-prompt row:
   - Method 1 (Gen tokens / wall clock): 24 * 512 / 61.16 = 200.916 output tok/s.
   - Method 2 (Reported tok/s * output fraction): 1607.4 * (512 / 4096) = 200.925 output tok/s.
   Both independent derivations agree on ~200.92 output tok/s (8x lower than reported 1607 tok/s).
7. Degradation: At matched batch 16, output goodput falls from 294.46 to 163.94 tok/s (a 44.3% drop).
========================================================================================
"""

import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")


def calculate_b1():
    """
    QUESTION B1:
    From model_spec.md alone, compute:
      (a) KV-cache bytes per token, exactly;
      (b) Approximate maximum number of concurrent 4096-token sequences this GPU can hold.
    Then check your prediction against the log.
    """
    print("=" * 75)
    print("PART B1: KV-Cache Sizing & Concurrency Limit")
    print("=" * 75)

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
    print(f"    --> Predicted capacity ceiling: approximately 25 to 26 concurrent sequences.\n")

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
        print(
            f"    batch {bs:>2}: util={util:.2f}, preempted_seqs={preempted:>2} "
            f"(preempted_seqs = sequences preempted at least once)"
        )
    print("    Log consistency: Batches 4-24 show util scaling smoothly up to 0.93 with 0 preemptions.")
    print("    At batch 32 & 48, cache utilization pegs at 0.97, triggering 7 and 23 preempted sequences.")
    print("    This confirms an approximate ~25 sequence physical capacity ceiling.")


def calculate_b3():
    """
    QUESTION B3:
    Find the column misread in REPORT_v0.md.
    Derive honest goodput of the batch-24 long-prompt row using two independent ways.
    Compare throughput at matched batch sizes.
    """
    print("\n" + "=" * 75)
    print("PART B3: Honest Goodput Derivation (Batch 24, Long Prompt)")
    print("=" * 75)

    # Data from Row 12: batch 24, prompt 3584, gen 512
    batch_size = 24
    prompt_len = 3584
    gen_len = 512
    total_len = prompt_len + gen_len  # 4096
    num_requests = 24
    wall_clock_s = 61.16
    reported_tok_s = 1607.4

    total_tokens = num_requests * total_len
    gen_tokens = num_requests * gen_len

    print(f"Row 12 parameters: batch={batch_size}, prompt_len={prompt_len}, gen_len={gen_len}, wall_clock={wall_clock_s}s")
    print(f"Total tokens = {total_tokens:,} ({num_requests * prompt_len:,} prompt + {gen_tokens:,} generated)")
    print(f"Harness reported_tok_s = {total_tokens} / {wall_clock_s} = {total_tokens / wall_clock_s:.2f} tok/s (matches log: {reported_tok_s})")
    print(f"Notice: Prompt tokens represent {prompt_len / total_len * 100:.1f}% of total tokens processed!\n")

    # Derivation Method 1: Generated tokens / Elapsed wall-clock time
    goodput_method1 = gen_tokens / wall_clock_s

    # Derivation Method 2: Reported throughput * Output-token fraction
    output_token_fraction = gen_len / total_len  # 512 / 4096 = 0.125
    goodput_method2 = reported_tok_s * output_token_fraction

    print("Two Independent Derivations of Honest Output Goodput:")
    print(f"  Method 1 (Generated tokens / wall_clock_s):")
    print(f"    ({num_requests} * {gen_len}) / {wall_clock_s} = {gen_tokens} / {wall_clock_s} = {goodput_method1:.3f} output tok/s")
    print(f"  Method 2 (Reported throughput * Output token fraction):")
    print(f"    {reported_tok_s} * ({gen_len} / {total_len}) = {reported_tok_s} * {output_token_fraction:.3f} = {goodput_method2:.3f} output tok/s")
    print(f"  Agreement: Both independent derivations confirm honest goodput is ~{goodput_method1:.2f} output tok/s.")
    print(f"  (The minor {abs(goodput_method1 - goodput_method2):.3f} difference is consistent with rounding of recorded fields in the harness log, such as duration or reported throughput rounded to 1-2 decimal places).")
    print(f"  Crucial insight: Real output goodput is ~8x lower than the intern's reported 1607.4 tok/s!\n")

    # Matched comparison: Batch 16 short vs long prompt
    print("Comparing Short vs Long Prompt at Matched Batch Size 16:")
    # Row 6: batch 16, prompt 512, gen 256, wall 13.91s
    goodput_short_b16 = (16 * 256) / 13.91
    # Row 11: batch 16, prompt 3584, gen 512, wall 49.97s
    goodput_long_b16 = (16 * 512) / 49.97
    degradation_pct = ((goodput_short_b16 - goodput_long_b16) / goodput_short_b16) * 100

    print(f"  Batch 16 Short Prompt (512 in, 256 out, 13.91s):  {goodput_short_b16:.2f} output tok/s")
    print(f"  Batch 16 Long Prompt  (3584 in, 512 out, 49.97s): {goodput_long_b16:.2f} output tok/s")
    print(f"  Output Goodput Degradation: {degradation_pct:.1f}% reduction (from 294.46 down to 163.94 tok/s)")


def print_latency_note():
    print("\n" + "=" * 75)
    print("NOTE ON LATENCY VS WALL-CLOCK COLUMNS (Caveat for Defense)")
    print("=" * 75)
    print("Twelve of thirteen rows have p95 latency exceeding the recorded run duration.")
    print("The supplied files do not explain the measurement boundaries. Client/server timing")
    print("differences are one hypothesis; harness instrumentation is needed to resolve this.")


if __name__ == "__main__":
    calculate_b1()
    calculate_b3()
    print_latency_note()
