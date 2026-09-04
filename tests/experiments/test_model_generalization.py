"""
Unit tests for experiments/model_generalization.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.model_generalization import (
    evaluate_model_optimization,
    run_model_generalization_experiment,
)


def test_evaluate_model_optimization_baseline():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(num_requests=2, total_prompt_len=64, prefix_ratio=0.5, max_new_tokens=5, seed=42)
    metrics = evaluate_model_optimization("sshleifer/tiny-gpt2", workload, optimization="baseline_fp32", device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "architecture_type" in metrics
    assert metrics["architecture_type"] == "gpt2"
    assert metrics["model_name"] == "sshleifer/tiny-gpt2"
    assert metrics["optimization"] == "baseline_fp32"
    assert metrics["total_tokens"] > 0


def test_evaluate_model_optimization_combined():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(num_requests=2, total_prompt_len=64, prefix_ratio=0.5, max_new_tokens=5, seed=42)
    metrics = evaluate_model_optimization("sshleifer/tiny-gpt2", workload, optimization="opt_all_combined", device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert "architecture_type" in metrics
    assert metrics["architecture_type"] == "gpt2"
    assert metrics["optimization"] == "opt_all_combined"
    assert metrics["total_tokens"] > 0


def test_run_model_generalization_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_model_generalization_experiment(
            models=["sshleifer/tiny-gpt2"],
            optimizations=["baseline_fp32", "opt_all_combined"],
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="model_generalization.jsonl",
        )

        assert len(results) == 2
        jsonl_path = os.path.join(tmp_dir, "model_generalization.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
