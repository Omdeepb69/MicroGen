#!/usr/bin/env python3
# ============================================================
#  MicroGen Decision Engine — Task 7.1
#  Batched Trie Latency Scaling Curve
# ============================================================
#
# Plots decision latency against candidate count (2 to 100)
# to evaluate the parallel batched BFS Trie optimization.
#
# Run this via Kaggle T4 or locally.

import time
import json
import platform
import numpy as np
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.engine import DecisionEngine
from microgen.decision.schema import Choice, ChoiceSchema
from microgen.decision.tokenizer import tokenize_candidate

def latency_pct(latencies_s):
    arr = np.array(latencies_s) * 1000.0
    return float(np.median(arr)), float(np.percentile(arr, 95))

def main():
    # ── 1. Environment & Setup ──────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE.type == "cuda" else "HuggingFaceTB/SmolLM-135M"
    
    print("=" * 64)
    print("  Task 7.1: Candidate-Count Latency Scaling Curve")
    print("=" * 64)
    print(f"  OS:       {platform.system()} {platform.release()}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Device:   {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model:    {MODEL_ID}")
    
    # ── 2. Load Model ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto" if DEVICE.type == "cuda" else None,
    )
    if DEVICE.type == "cpu":
        model.to(DEVICE)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    # ── 3. Configuration ────────────────────────────────────────
    N_CANDIDATES = [2, 4, 6, 10, 20, 50, 77, 100]
    WARMUP_ITERS = 10
    EVAL_ITERS = 50
    
    print(f"\n  Warmup Iters: {WARMUP_ITERS}")
    print(f"  Eval Iters:   {EVAL_ITERS}")
    print("-" * 64)

    # ── 4. Fixed Context Setup ──────────────────────────────────
    prompt_text = (
        "Classify the text into exactly one category.\n\n"
        "Text: The quick brown fox jumps over the lazy dog.\n\n"
        "Category:"
    )
    prompt_tokens = len(tokenizer(prompt_text).input_ids)

    # Generate synthetic long labels to force multi-token trie evaluation
    # E.g., "Category_Alpha_1", "Category_Alpha_2"
    all_labels = [f"Category_Alpha_{i}" for i in range(1, 101)]

    # We want to measure the number of forward passes.
    # The depth of the Trie determines the number of forward passes.
    # We will log the max tokens for the given candidate set.
    
    results = []

    for count in N_CANDIDATES:
        labels = all_labels[:count]
        schema = ChoiceSchema(name=f"trie_{count}", options=[Choice(n) for n in labels])
        
        # Determine candidate token lengths
        cand_stats = [tokenize_candidate(tokenizer, choice.name) for choice in schema.options]
        max_tokens = max(s.token_count for s in cand_stats)
        avg_tokens = np.mean([s.token_count for s in cand_stats])
        
        # The number of model forwards is 1 (prefill) + (max_tokens - 1) (decode)
        # assuming at least 1 multi-token candidate. Since all our labels are multi-token,
        # forwards = max_tokens.
        model_forwards = max_tokens
        
        print(f"  Evaluating {count:>3} candidates (max depth {max_tokens}) ... ", end="", flush=True)

        # Warmup
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                engine.choose(context=prompt_text, schema=schema)
        
        # Eval
        latencies = []
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        
        for _ in range(EVAL_ITERS):
            t0 = time.perf_counter()
            with torch.no_grad():
                engine.choose(context=prompt_text, schema=schema)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            
        p50, p95 = latency_pct(latencies)
        
        print(f"p50: {p50:4.0f} ms | p95: {p95:4.0f} ms")
        
        results.append([
            count,
            f"{p50:.0f}",
            f"{p95:.0f}",
            prompt_tokens,
            f"{avg_tokens:.1f} / {max_tokens}",
            model_forwards
        ])

    print("\n" + "=" * 70)
    print("  FINAL LATENCY SCALING")
    print("=" * 70)
    headers = ["Candidates", "p50 (ms)", "p95 (ms)", "Prompt Toks", "Cand Toks (Avg/Max)", "Model Forwards"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    print("=" * 70)
    
if __name__ == "__main__":
    main()
