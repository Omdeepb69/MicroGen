"""
Speculative Decoding Acceptance Rate & Speedup Threshold Sweep Experiment (RQ1 & RQ2).

Evaluates draft lookahead length K in [1..5], empirical token acceptance rate alpha,
and target verification speedup thresholds against target-only baseline using SpeculativeEngine.
"""

import time
from typing import Any, Dict, List, Optional
import torch

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice
from microgen.scheduler.speculative import SpeculativeEngine


def create_target_only_execution_fn(
    target_model_name: str,
    workload: WorkloadSuite,
    device_str: str = "cpu",
) -> Any:
    """Execution function measuring standard target-only autoregressive generation baseline."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(target_model_name)

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        t0_start = time.perf_counter()

        for req in workload.requests:
            prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

            # Prefill
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
            "acceptance_rate": 1.0,
        }

    return execution_fn


def create_speculative_execution_fn(
    draft_model_name: str,
    target_model_name: str,
    workload: WorkloadSuite,
    k_draft: int = 4,
    device_str: str = "cpu",
) -> Any:
    """Execution function evaluating SpeculativeEngine across draft lookahead length K."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    
    draft_backend = PyTorchBackend(device=device_obj)
    draft_backend.load_model(draft_model_name)

    target_backend = PyTorchBackend(device=device_obj)
    target_backend.load_model(target_model_name)

    engine = SpeculativeEngine(
        draft_backend=draft_backend,
        target_backend=target_backend,
        num_draft_tokens=k_draft,
    )

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_generated_tokens = 0
        acceptance_rates = []
        t0_start = time.perf_counter()

        for req in workload.requests:
            prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)
            t0_req = time.perf_counter()
            spec_result = engine.generate(prompt_tensor, max_new_tokens=req.max_new_tokens)
            device_obj.synchronize()
            t1_req = time.perf_counter()

            req_latency_ms = (t1_req - t0_req) * 1000.0
            # Rough estimate of TTFT as first step latency portion
            total_ttft_ms += req_latency_ms / (spec_result.num_steps if spec_result.num_steps > 0 else 1)
            total_generated_tokens += len(spec_result.output_ids) - req.prompt_len
            acceptance_rates.append(spec_result.acceptance_rate)

        t1_end = time.perf_counter()
        num_reqs = len(workload.requests)
        tot_latency_ms = (t1_end - t0_start) * 1000.0
        mean_acc_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0

        return {
            "ttft_ms": total_ttft_ms / num_reqs,
            "tpot_ms": tot_latency_ms / max(1, total_generated_tokens),
            "generated_tokens": total_generated_tokens,
            "total_latency_ms": tot_latency_ms,
            "acceptance_rate": mean_acc_rate,
        }

    return execution_fn


def run_speculative_sweep(
    draft_model_name: str = "sshleifer/tiny-gpt2",
    target_model_name: str = "sshleifer/tiny-gpt2",
    draft_lengths: Optional[List[int]] = None,
    num_requests: int = 3,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: str = "cpu",
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes draft length K sweep comparing target-only baseline against speculative decoding."""
    if draft_lengths is None:
        draft_lengths = [1, 2, 3, 4, 5]

    generator = WorkloadGenerator(tokenizer_name_or_path=target_model_name)
    workload = generator.generate_suite("spec_sweep", num_requests=num_requests, target_len_range=(64, 64), max_new_tokens=16, seed=42)

    results: List[ExperimentResult] = []

    # 1. Target-Only Baseline
    config_target = ExperimentConfig(
        model_name=target_model_name,
        optimization_name="target_only_baseline",
        baseline_type="microgen_unoptimized",
        n_trials=n_trials,
        warmup_trials=warmup_trials,
        device=device,
        output_dir=output_dir,
        jsonl_filename=jsonl_filename,
    )
    harness_target = ExperimentHarness(config_target)
    fn_target = create_target_only_execution_fn(target_model_name, workload, device_str=device)
    res_target = harness_target.run_experiment("spec_target_baseline", len(workload.requests), fn_target)
    results.append(res_target)

    # 2. Speculative Decoding for each draft length K
    for k in draft_lengths:
        config_spec = ExperimentConfig(
            model_name=target_model_name,
            optimization_name=f"speculative_decoding_k{k}",
            baseline_type="microgen_optimized",
            n_trials=n_trials,
            warmup_trials=warmup_trials,
            device=device,
            output_dir=output_dir,
            jsonl_filename=jsonl_filename,
        )
        harness_spec = ExperimentHarness(config_spec)
        fn_spec = create_speculative_execution_fn(
            draft_model_name=draft_model_name,
            target_model_name=target_model_name,
            workload=workload,
            k_draft=k,
            device_str=device,
        )
        res_spec = harness_spec.run_experiment(f"spec_k{k}", len(workload.requests), fn_spec)
        results.append(res_spec)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()
    trials = args.n_trials if args.n_trials is not None else (3 if args.quick else 30)
    warmups = 1 if args.quick else 5

    print(f"Executing Speculative Decoding Acceptance & Speedup Sweep (N={trials} trials)...")
    results = run_speculative_sweep(n_trials=trials, warmup_trials=warmups, device="cpu")
    print(f"Speculative sweep complete! Total experiments recorded: {len(results)}")
