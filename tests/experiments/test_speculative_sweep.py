"""
Unit tests for experiments/speculative_sweep.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.speculative_sweep import (
    create_speculative_execution_fn,
    create_target_only_execution_fn,
    run_speculative_sweep,
)


def test_target_only_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("spec_test", num_requests=2, target_len_range=(32, 32), max_new_tokens=5, seed=42)
    exec_fn = create_target_only_execution_fn("sshleifer/tiny-gpt2", workload, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] == 10
    assert metrics["acceptance_rate"] == 1.0


def test_speculative_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("spec_test", num_requests=2, target_len_range=(32, 32), max_new_tokens=5, seed=42)
    exec_fn = create_speculative_execution_fn("sshleifer/tiny-gpt2", "sshleifer/tiny-gpt2", workload, k_draft=2, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] > 0
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0


def test_run_speculative_sweep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_speculative_sweep(
            draft_model_name="sshleifer/tiny-gpt2",
            target_model_name="sshleifer/tiny-gpt2",
            draft_lengths=[1, 2],
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="spec_sweep.jsonl",
        )

        assert len(results) == 3  # Target baseline + k=1 + k=2
        jsonl_path = os.path.join(tmp_dir, "spec_sweep.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3
