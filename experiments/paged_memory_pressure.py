"""
Paged KV vs Contiguous Memory Pressure Sweep Experiment (RQ3).

Evaluates Paged KV Cache allocation vs Contiguous KV Cache under constrained memory regimes
(25%, 50%, 75%, 100% block capacity), measuring memory fragmentation ratio, peak concurrent requests
before OOM, and VRAM/RAM utilization.
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice
from microgen.runtime.kv_cache import KVCacheManager
from microgen.runtime.paged_kv import PagedKVCacheAllocator


def evaluate_contiguous_memory_pressure(
    workload: WorkloadSuite,
    max_memory_requests: int = 16,
    max_seq_len: int = 2048,
) -> Dict[str, Any]:
    """Evaluates contiguous pre-allocated KV cache under memory pressure."""
    manager = KVCacheManager()
    active_count = 0
    oom_count = 0
    total_tokens = 0
    preallocated_tokens = 0

    t0_start = time.perf_counter()

    for idx, req in enumerate(workload.requests):
        if active_count >= max_memory_requests:
            oom_count += 1
            continue

        try:
            # Contiguous allocation reserves max_seq_len upfront
            manager.allocate(req.request_id, max_seq_len=max_seq_len)
            active_count += 1
            actual_seq_len = req.prompt_len + req.max_new_tokens
            total_tokens += actual_seq_len
            preallocated_tokens += max_seq_len
        except (MemoryError, RuntimeError):
            oom_count += 1

    t1_end = time.perf_counter()
    fragmentation = 1.0 - (total_tokens / preallocated_tokens) if preallocated_tokens > 0 else 0.0

    return {
        "active_requests": active_count,
        "oom_count": oom_count,
        "fragmentation_ratio": fragmentation,
        "total_latency_ms": (t1_end - t0_start) * 1000.0,
        "total_tokens": total_tokens,
    }


def evaluate_paged_memory_pressure(
    workload: WorkloadSuite,
    num_blocks: int = 64,
    block_size: int = 16,
) -> Dict[str, Any]:
    """Evaluates block-level Paged KV cache allocation under memory pressure."""
    allocator = PagedKVCacheAllocator(num_blocks=num_blocks, block_size=block_size)
    active_tables = []
    oom_count = 0
    total_tokens = 0

    t0_start = time.perf_counter()

    for idx, req in enumerate(workload.requests):
        try:
            block_table = allocator.allocate_sequence(f"seq-{idx}", prompt_token_count=req.prompt_len)
            # Simulate token generation step by step
            for _ in range(req.max_new_tokens):
                allocator.append_token(block_table)

            active_tables.append(block_table)
            total_tokens += block_table.num_tokens
        except MemoryError:
            oom_count += 1

    t1_end = time.perf_counter()

    allocated_blocks = allocator.get_num_allocated_blocks()
    total_block_capacity_tokens = allocated_blocks * block_size
    fragmentation = 1.0 - (total_tokens / total_block_capacity_tokens) if total_block_capacity_tokens > 0 else 0.0

    return {
        "active_requests": len(active_tables),
        "oom_count": oom_count,
        "fragmentation_ratio": max(0.0, fragmentation),
        "total_latency_ms": (t1_end - t0_start) * 1000.0,
        "total_tokens": total_tokens,
    }


def run_paged_memory_pressure_sweep(
    model_name: str = "sshleifer/tiny-gpt2",
    capacity_ratios: Optional[List[float]] = None,
    total_requests: int = 16,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes contiguous vs paged memory pressure sweep across capacity constraint ratios."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if capacity_ratios is None:
        capacity_ratios = [0.25, 0.50, 0.75, 1.00]

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_suite("memory_pressure", num_requests=total_requests, target_len_range=(64, 128), max_new_tokens=16, seed=42)

    results: List[ExperimentResult] = []

    # Maximum block capacity baseline = total_requests * average_blocks_per_req
    base_blocks = total_requests * 12  # ~12 blocks per request at 16 tokens/block

    for ratio in capacity_ratios:
        ratio_pct = int(ratio * 100)
        constrained_max_reqs = max(1, int(total_requests * ratio))
        constrained_blocks = max(1, int(base_blocks * ratio))

        # 1. Contiguous Allocation Baseline
        config_cont = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"contiguous_memory_p{ratio_pct}",
            baseline_type="microgen_unoptimized",
            n_trials=n_trials,
            warmup_trials=warmup_trials,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_cont = ExperimentHarness(config_cont)
        fn_cont = lambda reqs=constrained_max_reqs: evaluate_contiguous_memory_pressure(workload, max_memory_requests=reqs)
        res_cont = harness_cont.run_experiment(f"contiguous_p{ratio_pct}", len(workload.requests), fn_cont)
        results.append(res_cont)

        # 2. Paged KV Cache Allocator
        config_paged = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"paged_kv_memory_p{ratio_pct}",
            baseline_type="microgen_optimized",
            n_trials=n_trials,
            warmup_trials=warmup_trials,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_paged = ExperimentHarness(config_paged)
        fn_paged = lambda blocks=constrained_blocks: evaluate_paged_memory_pressure(workload, num_blocks=blocks, block_size=16)
        res_paged = harness_paged.run_experiment(f"paged_p{ratio_pct}", len(workload.requests), fn_paged)
        results.append(res_paged)

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

    print(f"Executing Paged KV vs Contiguous Memory Pressure Sweep (N={trials} trials, device={target_device})...")
    results = run_paged_memory_pressure_sweep(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Memory pressure sweep complete! Total experiments recorded: {len(results)}")
