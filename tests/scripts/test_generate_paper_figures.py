"""Unit tests for scripts/generate_paper_figures.py module."""

import json
import os
import tempfile
import pytest

from scripts.generate_paper_figures import (
    generate_all_figures,
    load_experiment_data,
)


def _create_sample_jsonl(filepath: str) -> None:
    records = [
        {
            "config": {
                "optimization_name": "hf_baseline_in32_out16",
                "baseline_type": "hf_pytorch",
                "model_name": "gpt2",
                "device": "cuda:0",
            },
            "ttft_stats_ms": {"mean": 10.0},
            "tpot_stats_ms": {"mean": 2.0},
            "peak_allocated_mb_stats": {"mean": 500.0},
            "throughput_stats_tps": {"mean": 100.0},
        }
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_experiment_data_raises_on_missing_file():
    with pytest.raises(RuntimeError):
        load_experiment_data("non_existent_file.jsonl")


def test_load_experiment_data_valid_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = os.path.join(tmp_dir, "experiments.jsonl")
        _create_sample_jsonl(jsonl_path)
        records = load_experiment_data(jsonl_path)
        assert len(records) > 0


def test_generate_all_figures():
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = os.path.join(tmp_dir, "experiments.jsonl")
        _create_sample_jsonl(jsonl_path)
        files = generate_all_figures(jsonl_path=jsonl_path, output_dir=tmp_dir)

        assert len(files) == 10  # 5 figures x 2 formats (.png, .pdf)
        for f in files:
            assert os.path.exists(f)
            assert os.path.getsize(f) > 0
            assert f.endswith(".png") or f.endswith(".pdf")
