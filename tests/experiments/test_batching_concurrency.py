"""
Unit tests for experiments/batching_concurrency.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.batching_concurrency import (
    create_continuous_batching_execution_fn,
    create_static_batching_execution_fn,
    run_batching_concurrency_sweep,
)


def test_static_batching_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("batch_test", num_requests=4, target_len_range=(32, 128), max_new_tokens=5, seed=42)
    exec_fn = create_static_batching_execution_fn("sshleifer/tiny-gpt2", workload, batch_size=2, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] > 0


def test_continuous_batching_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("batch_test", num_requests=4, target_len_range=(32, 128), max_new_tokens=5, seed=42)
    exec_fn = create_continuous_batching_execution_fn("sshleifer/tiny-gpt2", workload, max_batch_size=2, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] > 0


def test_run_batching_concurrency_sweep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_batching_concurrency_sweep(
            model_name="sshleifer/tiny-gpt2",
            batch_sizes=[1, 2],
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="batch_sweep.jsonl",
        )

        assert len(results) == 4  # (static + continuous) * 2 batch sizes
        jsonl_path = os.path.join(tmp_dir, "batch_sweep.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4
