"""
Unit tests for experiments/paged_memory_pressure.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.paged_memory_pressure import (
    evaluate_contiguous_memory_pressure,
    evaluate_paged_memory_pressure,
    run_paged_memory_pressure_sweep,
)


def test_evaluate_contiguous_memory_pressure():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("mem_test", num_requests=4, target_len_range=(32, 64), max_new_tokens=8, seed=42)
    metrics = evaluate_contiguous_memory_pressure(workload, max_memory_requests=2)

    assert "active_requests" in metrics
    assert "oom_count" in metrics
    assert "fragmentation_ratio" in metrics
    assert metrics["active_requests"] == 2
    assert metrics["oom_count"] == 2


def test_evaluate_paged_memory_pressure():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("mem_test", num_requests=4, target_len_range=(32, 64), max_new_tokens=8, seed=42)
    metrics = evaluate_paged_memory_pressure(workload, num_blocks=100, block_size=16)

    assert "active_requests" in metrics
    assert "oom_count" in metrics
    assert "fragmentation_ratio" in metrics
    assert metrics["active_requests"] == 4
    assert metrics["oom_count"] == 0


def test_run_paged_memory_pressure_sweep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_paged_memory_pressure_sweep(
            model_name="sshleifer/tiny-gpt2",
            capacity_ratios=[0.5, 1.0],
            total_requests=4,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="memory_pressure.jsonl",
        )

        assert len(results) == 4  # (contiguous + paged) * 2 capacity ratios
        jsonl_path = os.path.join(tmp_dir, "memory_pressure.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4
