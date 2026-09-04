"""
Combinatorial Optimization Interaction Matrix Experiment (RQ3).

Evaluates individual vs combined inference optimizations (+Paged KV, +Prefix Caching,
+INT8 Quantization, +Paged+Prefix+INT8) across an 8-combination permutation matrix
to measure non-linear performance interactions, latency overhead, and throughput synergies.
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.caching.prefix_cache import PrefixKVCache
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice
from microgen.runtime.kv_cache import KVCacheManager
from microgen.runtime.paged_kv import PagedKVCacheAllocator


def evaluate_interaction_configuration(
    model_name: str,
    workload: WorkloadSuite,
    use_int8: bool = False,
    use_paged: bool = False,
    use_prefix: bool = False,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates a specific combination of optimizations from the 8-state interaction matrix."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()

    if use_int8:
        backend = QuantizedPyTorchBackend(device=device_obj, quant_type="int8")
    else:
        backend = PyTorchBackend(device=device_obj)

    backend.load_model(model_name)

    prefix_cache = PrefixKVCache(max_capacity=100) if use_prefix else None
    paged_allocator = PagedKVCacheAllocator(num_blocks=128, block_size=16) if use_paged else None
    kv_manager = KVCacheManager() if not use_paged else None

    total_ttft_ms = 0.0
    total_tpot_ms = 0.0
    total_generated_tokens = 0
    t0_start = time.perf_counter()

    for idx, req in enumerate(workload.requests):
        prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

        t0_prefill = time.perf_counter()

        # Check prefix cache if enabled
        cached_prefix_match = None
        if prefix_cache is not None:
            cached_prefix_match = prefix_cache.match_prefix(req.prompt_ids)

        if cached_prefix_match is not None:
            # Prefix hit - skip full prefill
            match_len, kv_cache = cached_prefix_match
            logits = backend.model(prompt_tensor[:, -1:]).logits
        else:
            if use_paged and paged_allocator is not None:
                block_table = paged_allocator.allocate_sequence(req.request_id, req.prompt_len)
                logits, kv_cache = backend.prefill(prompt_tensor)
            else:
                logits, kv_cache = backend.prefill(prompt_tensor)

            if prefix_cache is not None and kv_cache is not None:
                prefix_cache.insert(req.prompt_ids, kv_cache)

        device_obj.synchronize()
        t1_prefill = time.perf_counter()
        total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

        curr_token = torch.argmax(logits, dim=-1, keepdim=True)
        decode_times_ms = []

        for step in range(req.max_new_tokens - 1):
            t0_dec = time.perf_counter()
            if use_paged and paged_allocator is not None and 'block_table' in locals():
                paged_allocator.append_token(block_table)

            logits, kv_cache = backend.decode(curr_token, cache=kv_cache)
            device_obj.synchronize()
            t1_dec = time.perf_counter()

            decode_times_ms.append((t1_dec - t0_dec) * 1000.0)
            curr_token = torch.argmax(logits, dim=-1, keepdim=True)

        total_generated_tokens += req.max_new_tokens
        if decode_times_ms:
            total_tpot_ms += sum(decode_times_ms) / len(decode_times_ms)

        if use_paged and paged_allocator is not None and 'block_table' in locals():
            paged_allocator.free_sequence(block_table)

    t1_end = time.perf_counter()
    num_reqs = len(workload.requests)
    tot_latency_ms = (t1_end - t0_start) * 1000.0

    return {
        "ttft_ms": total_ttft_ms / max(1, num_reqs),
        "tpot_ms": total_tpot_ms / max(1, num_reqs),
        "generated_tokens": total_generated_tokens,
        "total_latency_ms": tot_latency_ms,
        "use_int8": use_int8,
        "use_paged": use_paged,
        "use_prefix": use_prefix,
    }


def run_combined_interactions_matrix(
    model_name: str = "sshleifer/tiny-gpt2",
    num_requests: int = 4,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: str = "cpu",
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes the full 8-configuration combinatorial optimization interaction matrix."""
    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_shared_prefix_workload(
        num_requests=num_requests,
        total_prompt_len=128,
        prefix_ratio=0.5,
        max_new_tokens=16,
        seed=42,
    )

    # 8 Combinatorial Permutations: (use_int8, use_paged, use_prefix)
    permutations = [
        ("baseline_fp32", False, False, False, "microgen_unoptimized"),
        ("opt_int8", True, False, False, "microgen_optimized"),
        ("opt_paged", False, True, False, "microgen_optimized"),
        ("opt_prefix", False, False, True, "microgen_optimized"),
        ("opt_int8_paged", True, True, False, "microgen_optimized"),
        ("opt_int8_prefix", True, False, True, "microgen_optimized"),
        ("opt_paged_prefix", False, True, True, "microgen_optimized"),
        ("opt_all_combined", True, True, True, "microgen_optimized"),
    ]

    results: List[ExperimentResult] = []

    for name, use_int8, use_paged, use_prefix, baseline_type in permutations:
        config = ExperimentConfig(
            model_name=model_name,
            optimization_name=name,
            baseline_type=baseline_type,
            n_trials=n_trials,
            warmup_trials=warmup_trials,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness = ExperimentHarness(config)
        fn = lambda i8=use_int8, pg=use_paged, px=use_prefix: evaluate_interaction_configuration(
            model_name=model_name,
            workload=workload,
            use_int8=i8,
            use_paged=pg,
            use_prefix=px,
            device_str=device,
        )
        res = harness.run_experiment(name, len(workload.requests), fn)
        results.append(res)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    trials = args.n_trials if args.n_trials is not None else (3 if args.quick else 30)
    warmups = 1 if args.quick else 5

    print(f"Executing Combinatorial Optimization Interaction Matrix Experiment (N={trials} trials)...")
    results = run_combined_interactions_matrix(n_trials=trials, warmup_trials=warmups, device="cpu")
    print(f"Interaction matrix complete! Total experiments recorded: {len(results)}")
