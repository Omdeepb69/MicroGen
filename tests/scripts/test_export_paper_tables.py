"""Unit tests for scripts/export_paper_tables.py module."""

import json
import os
import tempfile
import pytest

from scripts.export_paper_tables import (
    export_all_paper_artifacts,
    export_reproducibility_doc,
    load_experiment_records,
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


def test_load_experiment_records_raises_on_missing_file():
    with pytest.raises(RuntimeError):
        load_experiment_records("non_existent_file.jsonl")


def test_load_experiment_records_valid_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = os.path.join(tmp_dir, "experiments.jsonl")
        _create_sample_jsonl(jsonl_path)
        records = load_experiment_records(jsonl_path)
        assert len(records) > 0


def test_export_reproducibility_doc():
    with tempfile.TemporaryDirectory() as tmp_dir:
        repro_path = os.path.join(tmp_dir, "reproducibility.md")
        out = export_reproducibility_doc(repro_path)

        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            content = f.read()
            assert "# MicroGen Benchmark Reproducibility Package" in content


def test_export_all_paper_artifacts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = os.path.join(tmp_dir, "experiments.jsonl")
        _create_sample_jsonl(jsonl_path)
        tables_dir = os.path.join(tmp_dir, "tables")
        repro_path = os.path.join(tmp_dir, "reproducibility.md")
        paths = export_all_paper_artifacts(
            jsonl_path=jsonl_path,
            tables_dir=tables_dir,
            repro_path=repro_path,
        )

        assert len(paths) == 4
        for p in paths:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0
