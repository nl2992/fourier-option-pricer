"""Verify the validation matrix document and key benchmark CSV files load cleanly."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BENCHMARKS = ROOT / "benchmarks"


def test_paper_validation_matrix_exists():
    assert (DOCS / "paper_validation_matrix.md").exists()


def test_validation_matrix_has_required_sections():
    text = (DOCS / "paper_validation_matrix.md").read_text()
    assert "## Validation snapshot" in text
    assert "done" in text
    assert "partial" in text


def test_validation_matrix_has_minimum_rows():
    text = (DOCS / "paper_validation_matrix.md").read_text()
    # Count table rows (lines starting with "|" that have content)
    data_rows = [
        ln
        for ln in text.splitlines()
        if ln.startswith("|") and "---" not in ln and ln.count("|") >= 4
    ]
    assert len(data_rows) >= 20, f"Expected >= 20 validation rows, got {len(data_rows)}"


def test_capability_snapshot_exists():
    assert (DOCS / "current_capability_snapshot.md").exists()


def test_capability_snapshot_has_model_table():
    text = (DOCS / "current_capability_snapshot.md").read_text()
    assert "bsm" in text
    assert "heston" in text
    assert "26" in text


def test_fo2008_benchmark_dir_exists():
    assert (BENCHMARKS / "paper_replications").exists()


@pytest.mark.parametrize(
    "relpath",
    [
        "cos_method_improved/outputs/cos_method_improved_paper_compare.csv",
    ],
)
def test_benchmark_csv_exists(relpath):
    p = BENCHMARKS / relpath
    assert p.exists(), f"Benchmark CSV not found: {p}"


def test_benchmark_csv_has_data():
    p = BENCHMARKS / "cos_method_improved/outputs/cos_method_improved_paper_compare.csv"
    import csv

    rows = list(csv.DictReader(p.open()))
    assert len(rows) > 0, "Benchmark CSV is empty"
