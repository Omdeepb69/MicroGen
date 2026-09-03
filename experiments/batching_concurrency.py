"""
Static vs Continuous Batching Concurrency Sweep Experiment (RQ3).

Evaluates static batching vs continuous batching scheduler across batch size range
B in [1, 2, 4, 8, 16, 32, 64] under heterogeneous request workloads.
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
from microgen.scheduler.queue import Request
from microgen.scheduler.scheduler import ContinuousBatchingScheduler


def create_static_batching_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    batch_size: int = 4,
    device_str: str = "cpu",
) -> Any:
    """Execution function measuring static batching (chunked sequential processing) baseline."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(model_name)

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        t0_start = time.perf_counter()

        requests = workload.requests
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            for req in chunk:
                prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

                t0_prefill = time.perf_counter()
                logits, kv_cache = backend.prefill(prompt_tensor)
                device_obj.synchronize()
                t1_prefill = time.perf_counter()
                total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

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
        }

    return execution_fn


def create_continuous_batching_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    max_batch_size: int = 4,
    device_str: str = "cpu",
) -> Any:
    """Execution function evaluating ContinuousBatchingScheduler under batch capacity B."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(model_name)

    def execution_fn() -> Dict[str, Any]:
        kv_manager = KVCacheManager()
        scheduler = ContinuousBatchingScheduler(
            backend=backend,
            kv_cache_manager=kv_manager,
            max_batch_size=max_batch_size,
        )

        t0_start = time.perf_counter()
        for idx, req in enumerate(workload.requests):
            sched_req = Request(
                request_id=f"req-{idx}",
                prompt=req.prompt_text,
                prompt_ids=req.prompt_ids,
                max_new_tokens=req.max_new_tokens,
                arrival_time=t0_start,
            )
            scheduler.add_request(sched_req)

        completed_requests = scheduler.run_until_complete()
        device_obj.synchronize()
        t1_end = time.perf_counter()

        tot_latency_ms = (t1_end - t0_start) * 1000.0
        total_gen_tokens = sum(len(r.generated_token_ids) for r in completed_requests)
        
        # Calculate mean TTFT and TPOT across requests
        ttfts = [
            (r.start_time - r.arrival_time) * 1000.0
            for r in completed_requests
            if r.start_time is not None and r.arrival_time is not None
        ]
        tpots = [
            ((r.finish_time - r.start_time) * 1000.0) / max(1, len(r.generated_token_ids))
            for r in completed_requests
            if r.finish_time is not None and r.start_time is not None
        ]

        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0

        return {
            "ttft_ms": mean_ttft,
            "tpot_ms": mean_tpot,
            "generated_tokens": total_gen_tokens,
            "total_latency_ms": tot_latency_ms,
        }

    return execution_fn


def run_batching_concurrency_sweep(
    model_name: str = "sshleifer/tiny-gpt2",
    batch_sizes: Optional[List[int]] = None,
    num_requests: int = 8,
    n_trials: int = 5,
    device: str = "cpu",
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes static vs continuous batching concurrency sweep across batch sizes B."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16]

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_suite("concurrency_eval", num_requests=num_requests, target_len_range=(32, 256), max_new_tokens=16, seed=42)

    results: List[ExperimentResult] = []

    for b in batch_sizes:
        # 1. Static Batching Baseline
        config_static = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"static_batching_b{b}",
            baseline_type="microgen_unoptimized",
            n_trials=n_trials,
            warmup_trials=1,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_static = ExperimentHarness(config_static)
        fn_static = create_static_batching_execution_fn(model_name, workload, batch_size=b, device_str=device)
        res_static = harness_static.run_experiment(f"batching_static_b{b}", len(workload.requests), fn_static)
        results.append(res_static)

        # 2. Continuous Batching Scheduler
        config_cb = ExperimentConfig(
            model_name=model_name,
            optimization_name=f"continuous_batching_b{b}",
            baseline_type="microgen_optimized",
            n_trials=n_trials,
            warmup_trials=1,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_cb = ExperimentHarness(config_cb)
        fn_cb = create_continuous_batching_execution_fn(model_name, workload, max_batch_size=b, device_str=device)
        res_cb = harness_cb.run_experiment(f"batching_continuous_b{b}", len(workload.requests), fn_cb)
        results.append(res_cb)

    return results


if __name__ == "__main__":
    print("Executing Static vs Continuous Batching Concurrency Sweep...")
    results = run_batching_concurrency_sweep(n_trials=3, device="cpu")
    print(f"Batching concurrency sweep complete! Total experiments recorded: {len(results)}")
