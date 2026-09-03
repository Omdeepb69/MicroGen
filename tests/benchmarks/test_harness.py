"""
Unit tests for benchmarks/harness.py module.
"""

import json
import os
import tempfile
import pytest
from benchmarks.harness import (
    ExperimentConfig,
    ExperimentHarness,
    compute_percentile,
    compute_stats,
    reset_environment,
)


def test_compute_percentile():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50 = compute_percentile(data, 50.0)
    assert p50 == 30.0

    single = [100.0]
    assert compute_percentile(single, 95.0) == 100.0


def test_compute_stats():
    data = list(range(1, 101))  # 1 to 100
    stats = compute_stats([float(x) for x in data])
    assert "p50" in stats
    assert "p90" in stats
    assert "p95" in stats
    assert "p99" in stats
    assert stats["p50"] == 50.5
    assert stats["mean"] == 50.5


def test_reset_environment():
    reset_environment()  # Should run cleanly without exception


def test_experiment_harness_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = ExperimentConfig(
            model_name="sshleifer/tiny-gpt2",
            optimization_name="fp32_baseline",
            baseline_type="microgen_unoptimized",
            n_trials=5,
            warmup_trials=1,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="test_experiments.jsonl",
        )
        harness = ExperimentHarness(config)

        def mock_execution_fn():
            return {
                "ttft_ms": 10.0,
                "tpot_ms": 2.0,
                "generated_tokens": 10,
                "total_latency_ms": 30.0,
            }

        res = harness.run_experiment(
            workload_name="short_workload",
            num_requests=5,
            execution_fn=mock_execution_fn,
        )

        assert res.workload_name == "short_workload"
        assert len(res.trials) == 5
        assert res.ttft_stats_ms["p50"] == 10.0
        assert res.tpot_stats_ms["p50"] == 2.0

        # Check JSONL file creation
        jsonl_path = os.path.join(tmp_dir, "test_experiments.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["workload_name"] == "short_workload"
            assert record["num_trials_recorded"] == 5
