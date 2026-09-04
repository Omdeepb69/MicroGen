"""
Quantization Memory Lifecycle & Quality Validation Experiment (RQ1 & RQ2).

Measures stage-by-stage VRAM/RAM allocation across FP32 loading, INT8 weight quantization,
and FP32 deallocation, and validates logit accuracy (MSE and Cosine Similarity) against baseline FP32 logits.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from benchmarks.harness import ExperimentConfig, ExperimentHarness, ExperimentResult, reset_environment
from benchmarks.workloads import WorkloadGenerator, WorkloadSuite
from microgen.backends.pytorch import PyTorchBackend
from microgen.backends.quantized import QuantizedPyTorchBackend
from microgen.devices.cpu import CPUDevice
from microgen.devices.cuda import CUDADevice
from microgen.runtime.kv_cache import KVCacheState


def measure_quantization_lifecycle(
    model_name: str = "sshleifer/tiny-gpt2",
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Measures memory allocated/reserved across model loading and INT8 quantization lifecycle stages."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()

    # Stage 0: Clean slate
    reset_environment()
    mem_stage0 = device_obj.get_memory_info()

    # Stage 1: Load FP32 Model
    backend = PyTorchBackend(device=device_obj)
    backend.load_model(model_name)
    device_obj.synchronize()
    mem_stage1_fp32 = device_obj.get_memory_info()

    # Stage 2: Perform INT8 Weight Quantization
    quant_backend = QuantizedPyTorchBackend(device=device_obj)
    quant_backend.load_model(model_name)
    device_obj.synchronize()
    mem_stage2_int8 = device_obj.get_memory_info()

    # Calculate actual parameter sizes in bytes
    fp32_model_bytes = sum(p.numel() * p.element_size() for p in backend.model.parameters()) if backend.model else 0
    int8_model_bytes = 0
    if quant_backend.model:
        for name, buf in quant_backend.model.named_buffers():
            int8_model_bytes += buf.numel() * buf.element_size()
        for name, param in quant_backend.model.named_parameters():
            int8_model_bytes += param.numel() * param.element_size()

    return {
        "stage0_clean_mb": mem_stage0.get("allocated_mb", 0.0),
        "stage1_fp32_mb": mem_stage1_fp32.get("allocated_mb", 0.0),
        "stage2_int8_mb": mem_stage2_int8.get("allocated_mb", 0.0),
        "fp32_model_size_mb": fp32_model_bytes / (1024.0 * 1024.0),
        "int8_model_size_mb": int8_model_bytes / (1024.0 * 1024.0),
    }


def evaluate_quantization_quality(
    model_name: str = "sshleifer/tiny-gpt2",
    workload: Optional[WorkloadSuite] = None,
    device_str: str = "cpu",
) -> Dict[str, float]:
    """Validates logit accuracy (MSE and Cosine Similarity) of INT8 quantized weights vs FP32 baseline."""
    if workload is None:
        generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
        workload = generator.generate_suite("eval_quality", num_requests=2, target_len_range=(32, 32), max_new_tokens=5, seed=42)

    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()

    fp32_backend = PyTorchBackend(device=device_obj)
    fp32_backend.load_model(model_name)

    int8_backend = QuantizedPyTorchBackend(device=device_obj)
    int8_backend.load_model(model_name)

    mses = []
    cos_sims = []

    for req in workload.requests:
        prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)
        
        logits_fp32, _ = fp32_backend.prefill(prompt_tensor)
        logits_int8, _ = int8_backend.prefill(prompt_tensor)

        mse = F.mse_loss(logits_int8.float(), logits_fp32.float()).item()
        cos_sim = F.cosine_similarity(logits_int8.float().flatten(), logits_fp32.float().flatten(), dim=0).item()

        mses.append(mse)
        cos_sims.append(cos_sim)

    return {
        "mean_mse": sum(mses) / len(mses) if mses else 0.0,
        "mean_cosine_similarity": sum(cos_sims) / len(cos_sims) if cos_sims else 1.0,
    }


def create_quantized_execution_fn(
    model_name: str,
    workload: WorkloadSuite,
    device_str: str = "cpu",
    quant_kv: bool = False,
) -> Any:
    """Execution function measuring QuantizedPyTorchBackend with optional INT8 KV Cache."""
    device_obj = CUDADevice(0) if ("cuda" in device_str and torch.cuda.is_available()) else CPUDevice()
    backend = QuantizedPyTorchBackend(device=device_obj)
    backend.load_model(model_name)

    def execution_fn() -> Dict[str, Any]:
        total_ttft_ms = 0.0
        total_tpot_ms = 0.0
        total_generated_tokens = 0
        t0_start = time.perf_counter()

        for req in workload.requests:
            prompt_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=device_obj.torch_device)

            t0_prefill = time.perf_counter()
            logits, kv_cache = backend.prefill(prompt_tensor)
            device_obj.synchronize()
            t1_prefill = time.perf_counter()
            total_ttft_ms += (t1_prefill - t0_prefill) * 1000.0

            # Optional INT8 KV cache transformation
            if quant_kv and isinstance(kv_cache, KVCacheState):
                kv_cache.quantize_kv = True

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


def run_quantization_lifecycle_experiment(
    model_name: str = "sshleifer/tiny-gpt2",
    n_trials: int = 30,
    warmup_trials: int = 5,
    device: Optional[str] = None,
    output_dir: str = "results/raw",
    jsonl_filename: str = "experiments.jsonl",
) -> List[ExperimentResult]:
    """Executes quantization memory lifecycle & logit quality experiment suite."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = WorkloadGenerator(tokenizer_name_or_path=model_name)
    workload = generator.generate_suite("quant_eval", num_requests=3, target_len_range=(128, 128), max_new_tokens=16, seed=42)

    results: List[ExperimentResult] = []

    # 1. INT8 Weight Quantized
    config_int8 = ExperimentConfig(
        model_name=model_name,
        optimization_name="int8_weight_quantization",
        baseline_type="microgen_optimized",
        n_trials=n_trials,
        warmup_trials=warmup_trials,
        device=device,
        output_dir=output_dir,
        jsonl_filename=jsonl_filename,
    )
    harness_int8 = ExperimentHarness(config_int8)
    fn_int8 = create_quantized_execution_fn(model_name, workload, device_str=device, quant_kv=False)
    res_int8 = harness_int8.run_experiment("quant_weight_int8", len(workload.requests), fn_int8)
    results.append(res_int8)

    # 2. Dynamic INT8 KV Cache
    config_kv_int8 = ExperimentConfig(
        model_name=model_name,
        optimization_name="dynamic_int8_kv_cache",
        baseline_type="microgen_optimized",
        n_trials=n_trials,
        warmup_trials=warmup_trials,
        device=device,
        output_dir=output_dir,
        jsonl_filename=jsonl_filename,
    )
    harness_kv_int8 = ExperimentHarness(config_kv_int8)
    fn_kv_int8 = create_quantized_execution_fn(model_name, workload, device_str=device, quant_kv=True)
    res_kv_int8 = harness_kv_int8.run_experiment("quant_kv_int8", len(workload.requests), fn_kv_int8)
    results.append(res_kv_int8)

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

    print(f"Executing Quantization Lifecycle & Quality Validation (device={target_device})...")
    lifecycle = measure_quantization_lifecycle("sshleifer/tiny-gpt2")
    print("Lifecycle Memory (MB):", lifecycle)
    quality = evaluate_quantization_quality("sshleifer/tiny-gpt2")
    print("Logit Quality:", quality)
    results = run_quantization_lifecycle_experiment(n_trials=trials, warmup_trials=warmups, device=target_device)
    print(f"Quantization experiment suite complete! Total experiments recorded: {len(results)}")
