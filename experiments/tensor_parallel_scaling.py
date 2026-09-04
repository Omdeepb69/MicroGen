"""
Tensor-Parallel Multi-GPU Execution & Scaling Experiment.

Evaluates prefill TTFT latency, decode throughput (TPOT), and VRAM distribution
across 1-rank vs 2-rank Tensor-Parallel execution (TensorParallelPyTorchBackend).
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.parallel import TensorParallelPyTorchBackend
from microgen.devices import get_device, Device


def evaluate_tensor_parallel_scaling(
    model_name: str,
    workload: WorkloadSuite,
    world_size: int = 1,
    device_str: str = "cpu",
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluates model inference performance under 1-rank or multi-rank Tensor Parallelism."""
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if "cuda" in device_str and num_cuda > 0:
        if num_cuda >= world_size:
            devices = [get_device(f"cuda:{i}") for i in range(world_size)]
        else:
            primary = get_device("cuda:0")
            devices = [primary] * world_size
    else:
        primary = get_device("cpu")
        devices = [primary] * world_size

    if backend is None:
        backend = TensorParallelPyTorchBackend(world_size=world_size, devices=devices)
        backend.load_model(model_name)

    total_ttft_ms = 0.0
    total_tpot_ms = 0.0
    total_tokens = 0
    t0_start = time.perf_counter()

    for req in workload.requests:
        prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=devices[0].torch_device)

        t0_prefill = time.perf_counter()
        logits, kv_cache = backend.prefill(prompt_tensor)
        devices[0].synchronize()
        t1_prefill = time.perf_counter()
        total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

        curr_token = torch.argmax(logits, dim=-1, keepdim=True)
        decode_times_ms = []

        for step in range(req.max_new_tokens - 1):
            t0_dec = time.perf_counter()
            logits, kv_cache = backend.decode(curr_token, cache=kv_cache)
            devices[0].synchronize()
            t1_dec = time.perf_counter()

            decode_times_ms.append((t1_dec - t0_dec) * 1000.0)
            curr_token = torch.argmax(logits, dim=-1, keepdim=True)

        total_tokens += req.max_new_tokens
        if decode_times_ms:
            total_tpot_ms += sum(decode_times_ms) / len(decode_times_ms)

    t1_end = time.perf_counter()
    num_reqs = len(workload.requests)
    tot_latency_s = t1_end - t0_start
    throughput = total_tokens / tot_latency_s if tot_latency_s > 0 else 0.0

    mem_info = backend.get_memory_usage()
    alloc_mb = mem_info.get("allocated_bytes", 0) / (1024.0 * 1024.0)
    res_mb = mem_info.get("reserved_bytes", mem_info.get("allocated_bytes", 0)) / (1024.0 * 1024.0)

    return {
        "model_name": model_name,
        "world_size": world_size,
        "is_tensor_parallel": world_size > 1,
        "ttft_ms": total_ttft_ms / max(1, num_reqs),
        "tpot_ms": total_tpot_ms / max(1, num_reqs),
        "total_tokens": total_tokens,
        "throughput_tok_per_sec": throughput,
        "total_latency_ms": tot_latency_s * 1000.0,
        "vram_allocated_mb": alloc_mb,
        "vram_reserved_mb": res_mb,
    }


def run_tensor_parallel_scaling_experiment(
    model_name: str = "sshleifer/tiny-gpt2",
    world_sizes: Optional[List[int]] = None,
    num_requests: int = 4,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes Tensor Parallel scaling experiment comparing world_size=1 vs world_size=2."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if world_sizes is None:
        world_sizes = [1, 2]

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_suite(
        "tensor_parallel",
        num_requests=num_requests,
        target_len_range=(64, 128),
        max_new_tokens=16,
        seed=42,
    )

    results: List[ExperimentResult] = []

    for ws in world_sizes:
        baseline_type = "microgen_unoptimized" if ws == 1 else "microgen_optimized"
        config = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"tp_world_size_{ws}",
            baseline_type=baseline_type,
            n_trials=n_trials,
            warmup_trials=warmup_trials,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness = ExperimentHarness(config)
        fn = lambda w=ws: evaluate_tensor_parallel_scaling(
            model_name=model_name,
            workload=workload,
            world_size=w,
            device_str=device,
        )
        res = harness.run_experiment(f"tp_scaling_ws{ws}", len(workload.requests), fn)
        results.append(res)

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

    print(f"Executing Tensor-Parallel Multi-GPU Experiment (N={trials} trials, device={target_device})...")
    results = run_tensor_parallel_scaling_experiment(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Tensor Parallel scaling experiment complete! Total experiments recorded: {len(results)}")
