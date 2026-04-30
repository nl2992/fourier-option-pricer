"""Paper replication asset coverage tests.

These checks are intentionally about coverage, not pricing numerics:

- the published paper-anchor registry should stay explicit and complete,
- the FO2008 case registry should not silently lose or rename replay cases,
- the notebook-produced paper output bundles should keep all required CSVs,
  summaries, and figures.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.paper_replications.fo2008_cos.params import CASES
from benchmarks.paper_replications.manifest import (
    FO2008_EXPECTED_CASE_IDS,
    FO2008_EXPECTED_FAMILIES,
    PAPER_OUTPUT_BUNDLES,
    PUBLISHED_PAPER_ANCHOR_KEYS,
)
from foureng.refs.paper_refs import PAPER_ANCHORS


ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path: Path) -> None:
    payload = path.read_bytes()
    assert payload.startswith(PNG_SIGNATURE), f"{path} is not a valid PNG"
    assert len(payload) > 256, f"{path} looks unexpectedly small"


def test_published_paper_anchor_registry_is_exhaustive():
    assert tuple(PUBLISHED_PAPER_ANCHOR_KEYS) == (
        "fo2008_heston_atm",
        "lewis_heston_strip",
        "cm1999_vg_case4",
    )
    assert set(PUBLISHED_PAPER_ANCHOR_KEYS).issubset(PAPER_ANCHORS)
    assert PAPER_ANCHORS["heston_published_strip"] is PAPER_ANCHORS["lewis_heston_strip"]


def test_fo2008_replay_registry_is_exhaustive():
    assert tuple(CASES) == FO2008_EXPECTED_CASE_IDS

    families = tuple(dict.fromkeys(case.model for case in CASES.values()))
    assert families == FO2008_EXPECTED_FAMILIES


def test_paper_output_bundles_have_required_core_files():
    for bundle in PAPER_OUTPUT_BUNDLES:
        outdir = ROOT / bundle.output_dir
        assert outdir.is_dir(), f"missing output dir: {outdir}"

        summary = outdir / bundle.summary_file
        assert summary.is_file(), f"missing summary: {summary}"

        for rel in bundle.csv_files:
            path = outdir / rel
            assert path.is_file(), f"missing CSV: {path}"
            assert path.stat().st_size > 0, f"empty CSV: {path}"


def test_paper_output_bundles_have_required_figures():
    for bundle in PAPER_OUTPUT_BUNDLES:
        outdir = ROOT / bundle.output_dir
        figure_paths = [
            outdir / bundle.family_figure_pattern.format(family=family)
            for family in FO2008_EXPECTED_FAMILIES
        ]
        figure_paths.extend(outdir / rel for rel in bundle.extra_figures)

        for path in figure_paths:
            assert path.is_file(), f"missing figure: {path}"
            _assert_valid_png(path)


def test_paper_output_summaries_cover_all_tables():
    for bundle in PAPER_OUTPUT_BUNDLES:
        summary = (ROOT / bundle.output_dir / bundle.summary_file).read_text()
        for label in bundle.expected_table_labels:
            assert label in summary, f"{label} missing from {bundle.name} summary"
