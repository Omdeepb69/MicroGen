"""
Shared-Prefix Ratio Sweep Experiment (RQ1 & RQ2).

Evaluates the impact of shared prompt prefix ratios (alpha in [0.0, 0.25, 0.50, 0.75, 0.90, 1.00])
on Time To First Token (TTFT) reduction, prefix KV cache hit rate, and throughput using
the PrefixKVCache manager and PyTorchBackend.
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.caching.prefix_cache import PrefixKVCache
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice


def create_uncached_prefix_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    device_str: str = "cpu",
) -> Any:
    """Execution function measuring uncached baseline (full prefill for every request)."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(model_name)

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        t0_start = time.perf_counter()

        for req in workload.requests:
            prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

            # Full Prefill
            t0_prefill = time.perf_counter()
            logits, kv_cache = backend.prefill(prompt_tensor)
            device_obj.synchronize()
            t1_prefill = time.perf_counter()
            total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

            # Decode loop
            curr_token = torch.argmax(logits, dim=-1, keepdim=True)
            decode_times_ms = []

            for step in range(req.max_new_tokens - 1):
                t0_dec = time.perf_counter()
                logits, kv_cache = backend.decode(curr_token, cache=kv_cache)
                device_obj.synchronize()
                t1_dec = time.perf_counter()

                decode_times_ms.append((t1_dec - t0_dec) * 1000.0)
                curr_token = torch.argmax(logits, dim=-1, keepdim=True)

            total_generated_tokens += req.max_new_tokens
            if decode_times_ms:
                total_tpot_ms += sum(decode_times_ms) / len(decode_times_ms)

        t1_end = time.perf_counter()
        num_reqs = len(workload.requests)
        tot_latency_ms = (t1_end - t0_start) * 1000.0

        return {
            "ttft_ms": total_ttft_ms / num_reqs,
            "tpot_ms": total_tpot_ms / num_reqs if num_reqs > 0 else 0.0,
            "generated_tokens": total_generated_tokens,
            "total_latency_ms": tot_latency_ms,
            "cache_hit_rate": 0.0,
        }

    return execution_fn


def create_cached_prefix_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    prefix_ratio: float,
    device_str: str = "cpu",
) -> Any:
    """Execution function measuring prefix KV cache reuse across shared-prefix requests."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(model_name)

    def execution_fn() -> Dict[str, Any]:
        prefix_cache = PrefixKVCache(max_capacity=50)
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        cache_hits = 0
        t0_start = time.perf_counter()

        for req in workload.requests:
            match = prefix_cache.match_prefix(req.prompt_ids)

            if match is not None:
                cache_hits += 1
                matched_len, cached_kv_state = match
                remaining_prompt_ids = req.prompt_ids[matched_len:]

                t0_prefill = time.perf_counter()
                if remaining_prompt_ids:
                    rem_tensor = torch.tensor([remaining_prompt_ids], dtype=torch.long, device=device_obj.torch_device)
                    # Use cached KV state as past_key_values for remaining prefill
                    logits, kv_cache = backend.prefill(rem_tensor, cache=cached_kv_state)
                else:
                    # Exact 100% prefix hit: single decode evaluation
                    last_id = torch.tensor([[req.prompt_ids[-1]]], dtype=torch.long, device=device_obj.torch_device)
                    logits, kv_cache = backend.decode(last_id, cache=cached_kv_state)

                device_obj.synchronize()
                t1_prefill = time.perf_counter()
                total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

            else:
                # Cache Miss: full prefill
                prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)
                t0_prefill = time.perf_counter()
                logits, kv_cache = backend.prefill(prompt_tensor)
                device_obj.synchronize()
                t1_prefill = time.perf_counter()
                total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

                # Cache shared prefix for future requests if prefix_ratio > 0
                prefix_len = int(req.prompt_len * prefix_ratio)
                if prefix_len > 0:
                    prefix_cache.insert(req.prompt_ids[:prefix_len], kv_cache)

            # Decode loop
            curr_token = torch.argmax(logits, dim=-1, keepdim=True)
            decode_times_ms = []

            for step in range(req.max_new_tokens - 1):
                t0_dec = time.perf_counter()
                logits, kv_cache = backend.decode(curr_token, cache=kv_cache)
                device_obj.synchronize()
                t1_dec = time.perf_counter()

                decode_times_ms.append((t1_dec - t0_dec) * 1000.0)
                curr_token = torch.argmax(logits, dim=-1, keepdim=True)

            total_generated_tokens += req.max_new_tokens
            if decode_times_ms:
                total_tpot_ms += sum(decode_times_ms) / len(decode_times_ms)

        t1_end = time.perf_counter()
        num_reqs = len(workload.requests)
        tot_latency_ms = (t1_end - t0_start) * 1000.0
        hit_rate = cache_hits / num_reqs if num_reqs > 0 else 0.0

        return {
            "ttft_ms": total_ttft_ms / num_reqs,
            "tpot_ms": total_tpot_ms / num_reqs if num_reqs > 0 else 0.0,
            "generated_tokens": total_generated_tokens,
            "total_latency_ms": tot_latency_ms,
            "cache_hit_rate": hit_rate,
        }

    return execution_fn


def run_prefix_sharing_sweep(
    model_name: str = "sshleifer/tiny-gpt2",
    prefix_ratios: Optional[List[float]] = None,
    total_prompt_len: int = 256,
    num_requests: int = 5,
    n_trials: int = 5,
    device: str = "cpu",
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes shared-prefix ratio sweep comparing uncached vs prefix-cached runs."""
    if prefix_ratios is None:
        prefix_ratios = [0.0, 0.25, 0.50, 0.75, 0.90, 1.00]

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    results: List[ExperimentResult] = []

    for ratio in prefix_ratios:
        ratio_pct = int(ratio * 100)
        workload = generator.generate_shared_prefix_workload(
            num_requests=num_requests,
            total_prompt_len=total_prompt_len,
            prefix_ratio=ratio,
            seed=42,
            max_new_tokens=16,
        )

        # 1. Uncached baseline
        config_uncached = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"prefix_uncached_r{ratio_pct}",
            baseline_type="microgen_unoptimized",
            n_trials=n_trials,
            warmup_trials=1,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_uncached = ExperimentHarness(config_uncached)
        fn_uncached = create_uncached_prefix_execution_fn(model_name, workload, device_str=device)
        res_uncached = harness_uncached.run_experiment(
            workload_name=f"shared_prefix_{ratio_pct}pct",
            num_requests=len(workload.requests),
            execution_fn=fn_uncached,
        )
        results.append(res_uncached)

        # 2. Prefix-cached optimization
        config_cached = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"prefix_cached_r{ratio_pct}",
            baseline_type="microgen_optimized",
            n_trials=n_trials,
            warmup_trials=1,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_cached = ExperimentHarness(config_cached)
        fn_cached = create_cached_prefix_execution_fn(model_name, workload, prefix_ratio=ratio, device_str=device)
        res_cached = harness_cached.run_experiment(
            workload_name=f"shared_prefix_{ratio_pct}pct",
            num_requests=len(workload.requests),
            execution_fn=fn_cached,
        )
        results.append(res_cached)

    return results


if __name__ == "__main__":
    print("Executing Shared-Prefix Ratio Sweep...")
    results = run_prefix_sharing_sweep(n_trials=3, device="cpu")
    print(f"Sweep complete! Total experiments recorded: {len(results)}")
