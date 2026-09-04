"""
Unit tests for experiments/hardware_duality.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.hardware_duality import (
    evaluate_hardware_device_execution,
    run_hardware_duality_experiment,
)


def test_evaluate_hardware_device_execution_cpu():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("hw_test", num_requests=2, target_len_range=(32, 64), max_new_tokens=5, seed=42)
    metrics = evaluate_hardware_device_execution("sshleifer/tiny-gpt2", workload, target_device="cpu", optimization="fp32_baseline")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "throughput_tok_per_sec" in metrics
    assert "arch_generation" in metrics
    assert "cuda_compute_capability" in metrics
    assert "has_tensor_cores" in metrics
    assert metrics["target_device"] == "cpu"
    assert metrics["total_tokens"] > 0


def test_evaluate_hardware_device_execution_combined():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("hw_test", num_requests=2, target_len_range=(32, 64), max_new_tokens=5, seed=42)
    metrics = evaluate_hardware_device_execution("sshleifer/tiny-gpt2", workload, target_device="cpu", optimization="int8_paged_combined")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "arch_generation" in metrics
    assert metrics["use_int8"] is True
    assert metrics["use_paged"] is True
    assert metrics["total_tokens"] > 0


def test_run_hardware_duality_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_hardware_duality_experiment(
            model_name="sshleifer/tiny-gpt2",
            target_devices=["cpu"],
            num_requests=2,
            n_trials=2,
            output_dir=tmp_dir,
            jsonl_filename="hardware_duality.jsonl",
        )

        assert len(results) == 4  # 4 profiles on CPU
        jsonl_path = os.path.join(tmp_dir, "hardware_duality.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4
