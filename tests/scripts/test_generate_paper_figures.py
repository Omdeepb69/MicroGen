"""
Unit tests for scripts/generate_paper_figures.py module.
"""

import os
import tempfile
import pytest

from scripts.generate_paper_figures import (
    generate_all_figures,
    load_experiment_data,
)


def test_load_experiment_data_synthetic_fallback():
    records = load_experiment_data("non_existent_file.jsonl")
    assert len(records) > 0
    assert any("metrics" in r for r in records)


def test_generate_all_figures():
    with tempfile.TemporaryDirectory() as tmp_dir:
        files = generate_all_figures(jsonl_path="non_existent_file.jsonl", output_dir=tmp_dir)

        assert len(files) == 10  # 5 figures x 2 formats (.png, .pdf)
        for f in files:
            assert os.path.exists(f)
            assert os.path.getsize(f) > 0
            assert f.endswith(".png") or f.endswith(".pdf")
