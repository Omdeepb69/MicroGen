"""
Unit tests for experiments/tensor_parallel_scaling.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.tensor_parallel_scaling import (
    evaluate_tensor_parallel_scaling,
    run_tensor_parallel_scaling_experiment,
)


def test_evaluate_tensor_parallel_scaling_ws1():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("tp_test", num_requests=2, target_len_range=(32, 64), max_new_tokens=5, seed=42)
    metrics = evaluate_tensor_parallel_scaling("sshleifer/tiny-gpt2", workload, world_size=1, device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "world_size" in metrics
    assert metrics["world_size"] == 1
    assert metrics["is_tensor_parallel"] is False
    assert metrics["total_tokens"] > 0


def test_evaluate_tensor_parallel_scaling_ws2():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("tp_test", num_requests=2, target_len_range=(32, 64), max_new_tokens=5, seed=42)
    metrics = evaluate_tensor_parallel_scaling("sshleifer/tiny-gpt2", workload, world_size=2, device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "world_size" in metrics
    assert metrics["world_size"] == 2
    assert metrics["is_tensor_parallel"] is True
    assert metrics["total_tokens"] > 0


def test_run_tensor_parallel_scaling_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_tensor_parallel_scaling_experiment(
            model_name="sshleifer/tiny-gpt2",
            world_sizes=[1, 2],
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="tp_scaling.jsonl",
        )

        assert len(results) == 2
        jsonl_path = os.path.join(tmp_dir, "tp_scaling.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
