"""
Context Length & Output Length Scaling Sweep Experiment (RQ1 & RQ2).

Evaluates TTFT, TPOT (ITL), token throughput, and memory consumption across scaling
prompt context lengths (L_in) and generation output lengths (L_out) for standard backends.
"""

import time
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice


def create_hf_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    device_str: str = "cpu",
) -> Any:
    """Creates an execution function evaluating the reference Hugging Face PyTorch baseline."""
    device = torch.device(device_str if ("cuda" in device_str and torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        t0_start = time.perf_counter()

        with torch.no_grad():
            for req in workload.requests:
                input_ids = torch.tensor([req.prompt_ids], dtype=torch.long, device=device)
                
                # Prefill pass (TTFT)
                t0_prefill = time.perf_counter()
                out_prefill = model(input_ids, use_cache=True)
                if "cuda" in device_str and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1_prefill = time.perf_counter()
                ttft = (t1_prefill - t0_prefill) * 1000.0
                total_ttft_ms += ttft

                # Decode pass
                past_key_values = out_prefill.past_key_values
                curr_token = torch.argmax(out_prefill.logits[:, -1, :], dim=-1, keepdim=True)
                
                decode_times_ms = []
                for _ in range(req.max_new_tokens - 1):
                    t0_dec = time.perf_counter()
                    out_dec = model(curr_token, past_key_values=past_key_values, use_cache=True)
                    if "cuda" in device_str and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1_dec = time.perf_counter()
                    
                    decode_times_ms.append((t1_dec - t0_dec) * 1000.0)
                    past_key_values = out_dec.past_key_values
                    curr_token = torch.argmax(out_dec.logits[:, -1, :], dim=-1, keepdim=True)

                total_generated_tokens += req.max_new_tokens
                mean_tpot = (sum(decode_times_ms) / len(decode_times_ms)) if decode_times_ms else 0.0
                total_tpot_ms += mean_tpot

        t1_end = time.perf_counter()
        num_reqs = len(workload.requests)
        tot_latency_ms = (t1_end - t0_start) * 1000.0

        return {
            "ttft_ms": total_ttft_ms / num_reqs,
            "tpot_ms": total_tpot_ms / num_reqs,
            "generated_tokens": total_generated_tokens,
            "total_latency_ms": tot_latency_ms,
        }

    return execution_fn


def create_microgen_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    device_str: str = "cpu",
    use_cache: bool = True,
) -> Any:
    """Creates an execution function evaluating the MicroGen PyTorch engine backend."""
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
            
            # Prefill
            t0_prefill = time.perf_counter()
            logits, kv_cache = backend.prefill(prompt_tensor)
            device_obj.synchronize()
            t1_prefill = time.perf_counter()
            ttft = (t1_prefill - t0_prefill) * 1000.0
            total_ttft_ms += ttft

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
            mean_tpot = (sum(decode_times_ms) / len(decode_times_ms)) if decode_times_ms else 0.0
            total_tpot_ms += mean_tpot

        t1_end = time.perf_counter()
        num_reqs = len(workload.requests)
        tot_latency_ms = (t1_end - t0_start) * 1000.0

        return {
            "ttft_ms": total_ttft_ms / num_reqs,
            "tpot_ms": total_tpot_ms / num_reqs,
            "generated_tokens": total_generated_tokens,
            "total_latency_ms": tot_latency_ms,
        }

    return execution_fn


def run_context_length_sweep(
    model_name: str = "sshleifer/tiny-gpt2",
    prompt_lengths: Optional[List[int]] = None,
    output_lengths: Optional[List[int]] = None,
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """
    Executes a context length and output length sweep across baseline and MicroGen configurations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if prompt_lengths is None:
        prompt_lengths = [32, 128, 256]
    if output_lengths is None:
        output_lengths = [16, 32]

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    results: List[ExperimentResult] = []

    for l_in in prompt_lengths:
        for l_out in output_lengths:
            workload_name = f"context_in{l_in}_out{l_out}"
            workload = generator.generate_suite(
                name=workload_name,
                num_requests=3,
                target_len_range=(l_in, l_in),
                max_new_tokens=l_out,
                seed=42,
            )

            # 1. Baseline: HF PyTorch Reference
            config_hf = ExperimentConfig(
                model_name=model_name,
                optimization_name=f"hf_baseline_in{l_in}_out{l_out}",
                baseline_type="hf_pytorch",
                n_trials=n_trials,
                warmup_trials=warmup_trials,
                device=device,
                output_dir=output_dir,
                jsonl_filename=jsonl_filename,
            )
            harness_hf = ExperimentHarness(config_hf)
            fn_hf = create_hf_execution_fn(model_name, workload, device_str=device)
            res_hf = harness_hf.run_experiment(
                workload_name=workload_name,
                num_requests=len(workload.requests),
                execution_fn=fn_hf,
            )
            results.append(res_hf)

            # 2. MicroGen PyTorch Backend (Unoptimized)
            config_mg = ExperimentConfig(
                model_name=model_name,
                optimization_name=f"microgen_unoptimized_in{l_in}_out{l_out}",
                baseline_type="microgen_unoptimized",
                n_trials=n_trials,
                warmup_trials=warmup_trials,
                device=device,
                output_dir=output_dir,
                jsonl_filename=jsonl_filename,
            )
            harness_mg = ExperimentHarness(config_mg)
            fn_mg = create_microgen_execution_fn(model_name, workload, device_str=device, use_cache=True)
            res_mg = harness_mg.run_experiment(
                workload_name=workload_name,
                num_requests=len(workload.requests),
                execution_fn=fn_mg,
            )
            results.append(res_mg)

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

    print(f"Executing Context & Output Length Sweep (N={trials} trials, device={target_device})...")
    results = run_context_length_sweep(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Sweep complete! Total experiments recorded: {len(results)}")
