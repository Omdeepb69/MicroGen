"""
Popular Architecture Generalization Experiment (RQ4).

Evaluates whether observed inference optimization trade-offs (+INT8, +Paged KV, +Prefix Caching)
hold across popular modern open-weights model architectures (e.g. Qwen2.5-0.5B, TinyLlama-1.1B, GPT-2).
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


def evaluate_model_optimization(
    model_name: str,
    workload: WorkloadSuite,
    optimization: str = "baseline_fp32",
    device_str: str = "cuda",
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluates a single model under a specified optimization configuration."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()

    use_int8 = "int8" in optimization or "all_combined" in optimization
    use_paged = "paged" in optimization or "all_combined" in optimization
    use_prefix = "prefix" in optimization or "all_combined" in optimization

    if backend is None:
        if use_int8:
            backend = QuantizedPyTorchBackend(device=device_obj, quant_type="int8")
        else:
            backend = PyTorchBackend(device=device_obj)
        backend.load_model(model_name)

    prefix_cache = PrefixKVCache(max_capacity=100) if use_prefix else None
    paged_allocator = PagedKVCacheAllocator(num_blocks=128, block_size=16) if use_paged else None

    total_ttft_ms = 0.0
    total_tpot_ms = 0.0
    total_tokens = 0
    t0_start = time.perf_counter()

    for idx, req in enumerate(workload.requests):
        prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

        t0_prefill = time.perf_counter()

        cached_prefix_match = None
        if prefix_cache is not None:
            cached_prefix_match = prefix_cache.match_prefix(req.prompt_ids)

        if cached_prefix_match is not None:
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

        total_tokens += req.max_new_tokens
        if decode_times_ms:
            total_tpot_ms += sum(decode_times_ms) / len(decode_times_ms)

        if use_paged and paged_allocator is not None and 'block_table' in locals():
            paged_allocator.free_sequence(block_table)

    t1_end = time.perf_counter()
    num_reqs = len(workload.requests)
    arch_type = "unknown"
    if backend.model is not None and hasattr(backend.model, "config"):
        arch_type = getattr(backend.model.config, "model_type", "unknown")

    return {
        "model_name": model_name,
        "architecture_type": arch_type,
        "optimization": optimization,
        "ttft_ms": total_ttft_ms / max(1, num_reqs),
        "tpot_ms": total_tpot_ms / max(1, num_reqs),
        "total_tokens": total_tokens,
        "total_latency_ms": (t1_end - t0_start) * 1000.0,
    }


def run_model_generalization_experiment(
    models: Optional[List[str]] = None,
    optimizations: Optional[List[str]] = None,
    num_requests: int = 4,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes model architecture generalization evaluation across popular models."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if models is None:
        models = [
            "sshleifer/tiny-gpt2",
            "gpt2",
            "Qwen/Qwen2.5-0.5B",
            "meta-llama/Llama-3.2-1B",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]

    if optimizations is None:
        optimizations = ["baseline_fp32", "opt_int8", "opt_paged", "opt_prefix", "opt_all_combined"]

    results: List[ExperimentResult] = []

    for model_name in models:
        try:
            generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
            workload = generator.generate_shared_prefix_workload(
                num_requests=num_requests,
                total_prompt_len=128,
                prefix_ratio=0.5,
                max_new_tokens=16,
                seed=42,
            )

            for opt in optimizations:
                baseline_type = "microgen_unoptimized" if opt == "baseline_fp32" else "microgen_optimized"
                config = ExperimentConfig(
                    model_name=model_name,
                    optimization_name=f"gen_{opt}",
                    baseline_type=baseline_type,
                    n_trials=n_trials,
                    warmup_trials=warmup_trials,
                    device=device,
                    output_dir=output_dir,
                    jsonl_filename=jsonl_filename,
                )
                harness = ExperimentHarness(config)
                fn = lambda m=model_name, o=opt: evaluate_model_optimization(
                    model_name=m,
                    workload=workload,
                    optimization=o,
                    device_str=device,
                )
                res = harness.run_experiment(f"generalization_{opt}", len(workload.requests), fn)
                results.append(res)
        except Exception as err:
            print(f"[Warning] Model generalization sweep skipped '{model_name}': {err}")
            continue

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    trials = args.n_trials if args.n_trials is not None else (3 if args.quick else 30)
    warmups = 1 if args.quick else 5
    target_device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Executing Popular Architecture Generalization Experiment (N={trials} trials, device={target_device})...")
    results = run_model_generalization_experiment(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Generalization experiment complete! Total experiments recorded: {len(results)}")
