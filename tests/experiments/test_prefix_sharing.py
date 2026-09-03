"""
Unit tests for experiments/prefix_sharing.py module.
"""

import json
import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.prefix_sharing import (
    create_cached_prefix_execution_fn,
    create_uncached_prefix_execution_fn,
    run_prefix_sharing_sweep,
)


def test_uncached_prefix_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(
        num_requests=3,
        total_prompt_len=64,
        prefix_ratio=0.5,
        seed=42,
        max_new_tokens=5,
    )
    exec_fn = create_uncached_prefix_execution_fn("sshleifer/tiny-gpt2", workload, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["cache_hit_rate"] == 0.0
    assert metrics["generated_tokens"] == 15


def test_cached_prefix_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(
        num_requests=3,
        total_prompt_len=64,
        prefix_ratio=0.5,
        seed=42,
        max_new_tokens=5,
    )
    exec_fn = create_cached_prefix_execution_fn("sshleifer/tiny-gpt2", workload, prefix_ratio=0.5, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    # Request 0 is cache miss, requests 1 and 2 hit the prefix cache -> 2/3 hit rate
    assert metrics["cache_hit_rate"] > 0.0
    assert metrics["generated_tokens"] == 15


def test_run_prefix_sharing_sweep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_prefix_sharing_sweep(
            model_name="sshleifer/tiny-gpt2",
            prefix_ratios=[0.0, 0.5],
            total_prompt_len=64,
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="prefix_sweep.jsonl",
        )

        assert len(results) == 4  # (uncached + cached) * 2 ratios
        jsonl_path = os.path.join(tmp_dir, "prefix_sweep.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4
