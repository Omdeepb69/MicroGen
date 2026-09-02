#!/usr/bin/env python3
"""Benchmark script comparing autoregressive generation performance: cache-on vs cache-off."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minigen.generator import generate

MODEL_NAME = "sshleifer/tiny-gpt2"
PROMPT_TEXT = "The quick brown fox jumps over the lazy dog"
MAX_NEW_TOKENS = 40
NUM_WARMUP = 3
NUM_RUNS = 10


def run_benchmark():
    print("==================================================")
    print("       miniGen KV Cache Performance Benchmark     ")
    print("==================================================")
    print(f"Model:           {MODEL_NAME}")
    print(f"Prompt:          '{PROMPT_TEXT}'")
    print(f"New Tokens:      {MAX_NEW_TOKENS}")
    print(f"Warmup Runs:     {NUM_WARMUP}")
    print(f"Measured Runs:   {NUM_RUNS}")
    print("--------------------------------------------------\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(PROMPT_TEXT, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 1. Warmup runs
    for _ in range(NUM_WARMUP):
        _ = generate(model, input_ids, max_new_tokens=5, use_cache=True)
        _ = generate(model, input_ids, max_new_tokens=5, use_cache=False)

    # 2. Benchmark Cache-On (SimpleKVCache)
    print("Running Cache-ON benchmark...")
    cached_times = []
    cached_output = None
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        cached_output = generate(
            model, input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True
        )
        end = time.perf_counter()
        cached_times.append(end - start)

    avg_cached_sec = sum(cached_times) / len(cached_times)
    cached_tps = MAX_NEW_TOKENS / avg_cached_sec
    cached_ms_per_token = (avg_cached_sec * 1000) / MAX_NEW_TOKENS

    # 3. Benchmark Cache-Off (Full Sequence Recomputation)
    print("Running Cache-OFF benchmark...")
    uncached_times = []
    uncached_output = None
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        uncached_output = generate(
            model, input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=False
        )
        end = time.perf_counter()
        uncached_times.append(end - start)

    avg_uncached_sec = sum(uncached_times) / len(uncached_times)
    uncached_tps = MAX_NEW_TOKENS / avg_uncached_sec
    uncached_ms_per_token = (avg_uncached_sec * 1000) / MAX_NEW_TOKENS

    # 4. Correctness Check
    is_identical = torch.equal(cached_output, uncached_output)
    speedup = avg_uncached_sec / avg_cached_sec if avg_cached_sec > 0 else 0.0

    print("\n==================================================")
    print("                  BENCHMARK RESULTS               ")
    print("==================================================")
    print(f"Cache-ON  Avg Latency:  {avg_cached_sec * 1000:.2f} ms ({cached_ms_per_token:.2f} ms/token, {cached_tps:.1f} tok/s)")
    print(f"Cache-OFF Avg Latency:  {avg_uncached_sec * 1000:.2f} ms ({uncached_ms_per_token:.2f} ms/token, {uncached_tps:.1f} tok/s)")
    print("--------------------------------------------------")
    print(f"Speedup Ratio (OFF/ON): {speedup:.2f}x")
    print(f"Token Output Exact Match: {is_identical}")
    print("==================================================")

    generated_text = tokenizer.decode(cached_output[0], skip_special_tokens=True)
    print(f"\nGenerated Output Text:\n\"{generated_text}\"\n")

    assert is_identical, "Error: Cached and uncached outputs do not match!"
    print("Benchmark successfully verified!")


if __name__ == "__main__":
    run_benchmark()
