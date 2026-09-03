"""
Unit tests for experiments/context_sweep.py module.
"""

import json
import os
import tempfile
import pytest
from experiments.context_sweep import (
    create_hf_execution_fn,
    create_microgen_execution_fn,
    run_context_length_sweep,
)
from benchmarks.workloads import WorkloadGenerator


def test_hf_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("test_short", num_requests=2, target_len_range=(32, 32), max_new_tokens=5, seed=42)
    exec_fn = create_hf_execution_fn("sshleifer/tiny-gpt2", workload, device_str="cpu")
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] == 10
    assert metrics["ttft_ms"] > 0.0


def test_microgen_execution_fn():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_suite("test_short", num_requests=2, target_len_range=(32, 32), max_new_tokens=5, seed=42)
    exec_fn = create_microgen_execution_fn("sshleifer/tiny-gpt2", workload, device_str="cpu", use_cache=True)
    metrics = exec_fn()

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] == 10
    assert metrics["ttft_ms"] > 0.0


def test_run_context_length_sweep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_context_length_sweep(
            model_name="sshleifer/tiny-gpt2",
            prompt_lengths=[32],
            output_lengths=[8],
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="context_sweep.jsonl",
        )

        assert len(results) == 2  # HF baseline + MicroGen unoptimized
        jsonl_path = os.path.join(tmp_dir, "context_sweep.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
