"""Performance Benchmark: Decode-Free Inference vs Standard Generation.

Simulates a real-world workload (e.g., drone state evaluations) to quantify
the latency speedups of the MicroGen decision module against standard LLM generation.
"""

import time
import json
import argparse
from typing import Callable, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.decision.huggingface import TransformersDecisionModel
from microgen.decision.schema import Choice, ChoiceSchema, CandidateStats
from microgen.decision.engine import DecisionEngine
from microgen.decision.tokenizer import tokenize_candidate
from microgen.decision.scorer import _extract_last_token_logits, score_sequence_candidates
from microgen.decision.constrained import score_trie_candidates
from microgen.runtime.kv_cache import KVCacheState


def simulate_generation(model: Any, tokenizer: Any, prompt: str, schema: ChoiceSchema, device: torch.device) -> str:
    """Baseline A: Normal generation (LLM -> generate -> parse)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate up to 5 tokens to capture short commands
    outputs = model.generate(
        input_ids,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Simple parse: pick first option found in generated text (case insensitive)
    gen_upper = generated_text.upper()
    for opt in schema.options:
        if opt.name.upper() in gen_upper:
            return opt.name
    return "UNKNOWN"


def run_benchmark(num_evals: int = 100, model_name: str = "HuggingFaceTB/SmolLM-135M") -> None:
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    adapter = TransformersDecisionModel(model, tokenizer)
    engine = DecisionEngine(adapter)

    # We use a multi-token candidate schema to demonstrate sequence/trie advantages
    schema = ChoiceSchema(
        name="drone_action",
        options=[
            Choice("TURN LEFT"),
            Choice("TURN RIGHT"),
            Choice("PULL UP"),
            Choice("BRAKE NOW"),
            Choice("CONTINUE FLIGHT"),
        ]
    )

    base_context = "The drone is flying towards a tall building. It is 5 meters away.\nWhat should the drone do next? Action:"

    print(f"\nStarting benchmark with {num_evals} evaluations on {device}...")
    results = {}

    # --- Baseline A: Generation ---
    print("Running Baseline A: Normal Generation...")
    start = time.perf_counter()
    for _ in range(num_evals):
        simulate_generation(model, tokenizer, base_context, schema, device)
    time_a = time.perf_counter() - start
    results["A_Generation"] = time_a

    # --- Prepare stats for internal baselines ---
    prompt_encoding = tokenizer(base_context, return_tensors="pt")
    input_ids = prompt_encoding.input_ids.to(device)
    stats_list = [tokenize_candidate(tokenizer, c.name) for c in schema.options]
    
    # --- Baseline C: Sequence Scoring (Linear) ---
    print("Running Baseline C: Sequence Scoring (Linear)...")
    start = time.perf_counter()
    for _ in range(num_evals):
        cache = KVCacheState()
        logits, _ = adapter.prefill(input_ids, cache=cache)
        last_logits = _extract_last_token_logits(logits)
        
        _ = score_sequence_candidates(
            last_logits=last_logits,
            candidates=stats_list,
            decode_fn=adapter.decode,
            prefill_cache=cache,
            device=device,
            alpha=1.0,
        )
    time_c = time.perf_counter() - start
    results["C_Sequence_Scoring"] = time_c

    # --- Baseline D: Constrained Candidate Scoring (Trie) ---
    print("Running Baseline D: Constrained Candidate Scoring (Trie)...")
    start = time.perf_counter()
    for _ in range(num_evals):
        cache = KVCacheState()
        logits, _ = adapter.prefill(input_ids, cache=cache)
        last_logits = _extract_last_token_logits(logits)
        
        _ = score_trie_candidates(
            last_logits=last_logits,
            candidates=stats_list,
            decode_fn=adapter.decode,
            prefill_cache=cache,
            device=device,
            alpha=1.0,
        )
    time_d = time.perf_counter() - start
    results["D_Constrained_Trie"] = time_d

    # --- Baseline E: MicroGen Full API ---
    print("Running Baseline E: MicroGen Full API (choose)...")
    start = time.perf_counter()
    for _ in range(num_evals):
        engine.choose(base_context, schema)
    time_e = time.perf_counter() - start
    results["E_MicroGen_API"] = time_e

    # --- Print Markdown Table ---
    print("\n## Performance Benchmark Results\n")
    print("| Baseline | Total Time (s) | Latency (ms/eval) | Speedup vs Gen |")
    print("|----------|----------------|-------------------|----------------|")
    
    gen_time = results["A_Generation"]
    
    for name, total_time in results.items():
        latency_ms = (total_time / num_evals) * 1000
        speedup = gen_time / total_time
        print(f"| {name:<20} | {total_time:>14.2f} | {latency_ms:>17.2f} | {speedup:>13.2f}x |")
        
    # Save to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run decision engine performance benchmarks.")
    parser.add_argument("--evals", type=int, default=10, help="Number of evaluations to run.")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="HuggingFace model ID.")
    args = parser.parse_args()
    
    run_benchmark(num_evals=args.evals, model_name=args.model)
