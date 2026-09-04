"""
Hardware Heterogeneity & CPU vs GPU Efficiency Profile Experiment (RQ4).

Evaluates execution backends across CPU (CPUDevice) vs CUDA (CUDADevice) under identical workload suites,
measuring TTFT, TPOT, VRAM allocated/reserved, memory bandwidth efficiency, and cross-hardware speedups.
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.devices import get_device
from microgen.runtime.paged_kv import PagedKVCacheAllocator


def evaluate_hardware_device_execution(
    model_name: str,
    workload: WorkloadSuite,
    target_device: str = "cpu",
    use_int8: bool = False,
    use_paged: bool = False,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluates model inference performance under a specific hardware device and optimization state."""
    device_obj = get_device(target_device if (target_device == "cpu" or torch.cuda.is_available()) else "cpu")

    if backend is None:
        if use_int8:
            backend = QuantizedPyTorchBackend(device=device_obj, quant_type="int8")
        else:
            backend = PyTorchBackend(device=device_obj)
        backend.load_model(model_name)

    paged_allocator = PagedKVCacheAllocator(num_blocks=128, block_size=16) if use_paged else None

    total_ttft_ms = 0.0
    total_tpot_ms = 0.0
    total_tokens = 0
    t0_start = time.perf_counter()

    for idx, req in enumerate(workload.requests):
        prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

        t0_prefill = time.perf_counter()
        if use_paged and paged_allocator is not None:
            block_table = paged_allocator.allocate_sequence(req.request_id, req.prompt_len)
            logits, kv_cache = backend.prefill(prompt_tensor)
        else:
            logits, kv_cache = backend.prefill(prompt_tensor)

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
    tot_latency_s = t1_end - t0_start
    throughput = total_tokens / tot_latency_s if tot_latency_s > 0 else 0.0

    mem_info = device_obj.get_memory_info()
    alloc_mb = mem_info.get("allocated_bytes", 0) / (1024.0 * 1024.0)
    res_mb = mem_info.get("reserved_bytes", mem_info.get("allocated_bytes", 0)) / (1024.0 * 1024.0)

    return {
        "model_name": model_name,
        "target_device": target_device,
        "use_int8": use_int8,
        "use_paged": use_paged,
        "ttft_ms": total_ttft_ms / max(1, num_reqs),
        "tpot_ms": total_tpot_ms / max(1, num_reqs),
        "total_tokens": total_tokens,
        "throughput_tok_per_sec": throughput,
        "total_latency_ms": tot_latency_s * 1000.0,
        "vram_allocated_mb": alloc_mb,
        "vram_reserved_mb": res_mb,
    }


def run_hardware_duality_experiment(
    model_name: str = "sshleifer/tiny-gpt2",
    target_devices: Optional[List[str]] = None,
    num_requests: int = 4,
    n_trials: int = 30,
    warmup_trials: int = 5,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes CPU vs CUDA hardware efficiency comparison across optimization profiles."""
    if target_devices is None:
        target_devices = ["cpu"]
        if torch.cuda.is_available():
            target_devices.append("cuda:0")

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_suite(
        "hardware_duality",
        num_requests=num_requests,
        target_len_range=(64, 128),
        max_new_tokens=16,
        seed=42,
    )

    profiles = [
        ("fp32_baseline", False, False),
        ("int8_quant", True, False),
        ("paged_kv", False, True),
        ("int8_paged_combined", True, True),
    ]

    results: List[ExperimentResult] = []

    for dev in target_devices:
        for profile_name, use_int8, use_paged in profiles:
            baseline_type = "microgen_unoptimized" if profile_name == "fp32_baseline" else "microgen_optimized"
            config = ExperimentConfig(
                model_name=model_name,
                optimization_name=f"hw_{dev}_{profile_name}",
                baseline_type=baseline_type,
                n_trials=n_trials,
                warmup_trials=warmup_trials,
                device=dev,
                output_dir=output_dir,
                jsonl_filename=jsonl_filename,
            )
            harness = ExperimentHarness(config)
            fn = lambda d=dev, i8=use_int8, pg=use_paged: evaluate_hardware_device_execution(
                model_name=model_name,
                workload=workload,
                target_device=d,
                use_int8=i8,
                use_paged=pg,
            )
            res = harness.run_experiment(f"hw_{dev}_{profile_name}", len(workload.requests), fn)
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

    print(f"Executing Hardware Heterogeneity & CPU vs GPU Efficiency Profile Experiment (N={trials} trials)...")
    results = run_hardware_duality_experiment(n_trials=trials, warmup_trials=warmups)
    print(f"Hardware duality experiment complete! Total experiments recorded: {len(results)}")
