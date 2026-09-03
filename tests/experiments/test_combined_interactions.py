"""
Unit tests for experiments/combined_interactions.py module.
"""

import os
import tempfile
import pytest

from benchmarks.workloads import WorkloadGenerator
from experiments.combined_interactions import (
    evaluate_interaction_configuration,
    run_combined_interactions_matrix,
)


def test_evaluate_interaction_configuration_baseline():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(num_requests=2, total_prompt_len=64, prefix_ratio=0.5, max_new_tokens=5, seed=42)
    metrics = evaluate_interaction_configuration("sshleifer/tiny-gpt2", workload, use_int8=False, use_paged=False, use_prefix=False, device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] > 0
    assert metrics["use_int8"] is False
    assert metrics["use_paged"] is False
    assert metrics["use_prefix"] is False


def test_evaluate_interaction_configuration_combined():
    generator = WorkloadGenerator("sshleifer/tiny-gpt2")
    workload = generator.generate_shared_prefix_workload(num_requests=2, total_prompt_len=64, prefix_ratio=0.5, max_new_tokens=5, seed=42)
    metrics = evaluate_interaction_configuration("sshleifer/tiny-gpt2", workload, use_int8=True, use_paged=True, use_prefix=True, device_str="cpu")

    assert "ttft_ms" in metrics
    assert "tpot_ms" in metrics
    assert metrics["generated_tokens"] > 0
    assert metrics["use_int8"] is True
    assert metrics["use_paged"] is True
    assert metrics["use_prefix"] is True


def test_run_combined_interactions_matrix():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_combined_interactions_matrix(
            model_name="sshleifer/tiny-gpt2",
            num_requests=2,
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="combined_interactions.jsonl",
        )

        assert len(results) == 8  # 8 permutations in combinatorial matrix
        jsonl_path = os.path.join(tmp_dir, "combined_interactions.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 8
