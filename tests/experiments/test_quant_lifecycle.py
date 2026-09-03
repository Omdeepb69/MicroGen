"""
Unit tests for experiments/quant_lifecycle.py module.
"""

import os
import tempfile
import pytest

from experiments.quant_lifecycle import (
    evaluate_quantization_quality,
    measure_quantization_lifecycle,
    run_quantization_lifecycle_experiment,
)


def test_measure_quantization_lifecycle():
    metrics = measure_quantization_lifecycle("sshleifer/tiny-gpt2", device_str="cpu")
    assert "fp32_model_size_mb" in metrics
    assert "int8_model_size_mb" in metrics
    assert metrics["fp32_model_size_mb"] > 0.0
    assert metrics["int8_model_size_mb"] > 0.0


def test_evaluate_quantization_quality():
    quality = evaluate_quantization_quality("sshleifer/tiny-gpt2", device_str="cpu")
    assert "mean_mse" in quality
    assert "mean_cosine_similarity" in quality
    assert quality["mean_cosine_similarity"] > 0.5  # High correlation expected


def test_run_quantization_lifecycle_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_quantization_lifecycle_experiment(
            model_name="sshleifer/tiny-gpt2",
            n_trials=2,
            device="cpu",
            output_dir=tmp_dir,
            jsonl_filename="quant_sweep.jsonl",
        )

        assert len(results) == 2
        jsonl_path = os.path.join(tmp_dir, "quant_sweep.jsonl")
        assert os.path.exists(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
