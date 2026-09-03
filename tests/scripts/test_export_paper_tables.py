"""
Unit tests for scripts/export_paper_tables.py module.
"""

import os
import tempfile
import pytest

from scripts.export_paper_tables import (
    export_all_paper_artifacts,
    export_reproducibility_doc,
    load_experiment_records,
)


def test_load_experiment_records_synthetic_fallback():
    records = load_experiment_records("non_existent_file.jsonl")
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
        tables_dir = os.path.join(tmp_dir, "tables")
        repro_path = os.path.join(tmp_dir, "reproducibility.md")
        paths = export_all_paper_artifacts(
            jsonl_path="non_existent_file.jsonl",
            tables_dir=tables_dir,
            repro_path=repro_path,
        )

        assert len(paths) == 4
        for p in paths:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 0
