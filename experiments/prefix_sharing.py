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


def clone_and_slice_cache(cache: Any, prefix_len: Optional[int] = None) -> Any:
    """Clone and optionally slice KV cache across PyTorch tuple and DynamicCache structures."""
    if cache is None:
        return None
    if isinstance(cache, tuple):
        if prefix_len is not None:
            return tuple((k[..., :prefix_len, :].clone(), v[..., :prefix_len, :].clone()) for k, v in cache)
        return tuple((k.clone(), v.clone()) for k, v in cache)
    if hasattr(cache, "layers"):
        try:
            from transformers.cache_utils import DynamicCache, DynamicLayer
            new_cache = DynamicCache()
            for layer in cache.layers:
                new_layer = DynamicLayer()
                k = layer.keys if prefix_len is None else layer.keys[..., :prefix_len, :]
                v = layer.values if prefix_len is None else layer.values[..., :prefix_len, :]
                new_layer.keys = k.clone()
                new_layer.values = v.clone()
                new_layer.is_initialized = True
                new_layer.dtype = k.dtype
                new_layer.device = k.device
                new_cache.layers.append(new_layer)
            return new_cache
        except Exception:
            pass
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            from transformers.cache_utils import DynamicCache
            new_cache = DynamicCache()
            for idx, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
                k_sub = k if prefix_len is None else k[..., :prefix_len, :]
                v_sub = v if prefix_len is None else v[..., :prefix_len, :]
                new_cache.update(k_sub.clone(), v_sub.clone(), layer_idx=idx)
            return new_cache
        except Exception:
            pass
    return cache


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
                cloned_kv_state = clone_and_slice_cache(cached_kv_state)

                t0_prefill = time.perf_counter()
                if remaining_prompt_ids:
                    rem_tensor = torch.tensor([remaining_prompt_ids], dtype=torch.long, device=device_obj.torch_device)
                    # Use cloned KV state as past_key_values for remaining prefill
                    logits, kv_cache = backend.prefill(rem_tensor, cache=cloned_kv_state)
                else:
                    # Exact 100% prefix hit: single decode evaluation
                    last_id = torch.tensor([[req.prompt_ids[-1]]], dtype=torch.long, device=device_obj.torch_device)
                    logits, kv_cache = backend.decode(last_id, cache=cloned_kv_state)

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
                    sliced_kv = clone_and_slice_cache(kv_cache, prefix_len)
                    prefix_cache.insert(req.prompt_ids[:prefix_len], sliced_kv)

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
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes shared-prefix ratio sweep comparing uncached vs prefix-cached runs."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
            warmup_trials=warmup_trials,
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
            warmup_trials=warmup_trials,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    trials = args.n_trials if args.n_trials is not None else (3 if args.quick else 30)
    warmups = 1 if args.quick else 5
    target_device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Executing Shared-Prefix Ratio Sweep (N={trials} trials, device={target_device})...")
    results = run_prefix_sharing_sweep(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Sweep complete! Total experiments recorded: {len(results)}")
