"""Expected paper-backed replication coverage.

This module is a small audit manifest for the repo's paper-facing assets:

- the published paper anchors used in tests,
- the FO2008 replay registry,
- the figure / CSV / summary bundles written by the two replication notebooks.

The goal is not to drive pricing logic. It exists so tests can fail loudly if
we add or rename paper material without updating the replication outputs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperOutputBundle:
    """One notebook-produced paper replication output bundle."""

    name: str
    output_dir: str
    summary_file: str
    csv_files: tuple[str, ...]
    family_figure_pattern: str
    extra_figures: tuple[str, ...]
    expected_table_labels: tuple[str, ...]


# Distinct published anchors. ``heston_published_strip`` is intentionally not
# listed here because it is an alias of ``lewis_heston_strip``.
PUBLISHED_PAPER_ANCHOR_KEYS: tuple[str, ...] = (
    "fo2008_heston_atm",
    "lewis_heston_strip",
    "cm1999_vg_case4",
)


FO2008_EXPECTED_CASE_IDS: tuple[str, ...] = (
    "bsm_table2",
    "heston_table4_t1",
    "heston_table5_t10",
    "heston_table6_strip",
    "vg_table7_t01",
    "vg_table7_t1",
    "cgmy_table8_y05",
    "cgmy_table9_y15",
    "cgmy_table10_y198",
)


FO2008_EXPECTED_FAMILIES: tuple[str, ...] = (
    "bsm",
    "heston",
    "vg",
    "cgmy",
)


FO2008_EXPECTED_TABLE_LABELS: tuple[str, ...] = (
    "Table 2",
    "Table 4",
    "Table 5",
    "Table 6",
    "Table 7",
    "Table 8",
    "Table 9",
    "Table 10",
)


JUNIKE_SUMMARY_CASE_IDS: tuple[str, ...] = (
    "bsm_table2",
    "heston_table4_t1",
    "heston_table5_t10",
    "heston_table6_strip",
    "vg_table7_t01",
    "vg_table7_t1",
    "cgmy_table8_y05",
    "cgmy_table10_y198",
)


REGRESSION_STRIP_MODELS: tuple[str, ...] = (
    "ousv",
    "cgmy",
    "nig",
    "bates",
    "heston_kou",
    "heston_cgmy",
)


PAPER_OUTPUT_BUNDLES: tuple[PaperOutputBundle, ...] = (
    PaperOutputBundle(
        name="fo2008_replication",
        output_dir="benchmarks/paper_replications/fo2008_cos/outputs",
        summary_file="summary.md",
        csv_files=(
            "fo2008_replication_errors.csv",
            "fo2008_replication_prices.csv",
            "fo2008_replication_timings.csv",
            "fo2008_extended_comparison.csv",
        ),
        family_figure_pattern="figures/fig_family_{family}.png",
        extra_figures=(
            "figures/fig_extension_error_vs_n.png",
            "figures/fig_extension_error_vs_time.png",
            "figures/fig_frontier.png",
        ),
        expected_table_labels=FO2008_EXPECTED_TABLE_LABELS,
    ),
    PaperOutputBundle(
        name="cos_paper_replication",
        output_dir="benchmarks/paper_replications/cos_paper_replication/outputs",
        summary_file="summary.md",
        csv_files=(
            "fo2008_replication_errors.csv",
            "fo2008_replication_prices.csv",
            "fo2008_replication_timings.csv",
            "extended_comparison.csv",
        ),
        family_figure_pattern="figures/family_{family}.png",
        extra_figures=(
            "figures/extension_error_vs_n.png",
            "figures/extension_error_vs_time.png",
            "figures/frontier.png",
        ),
        expected_table_labels=FO2008_EXPECTED_TABLE_LABELS,
    ),
)


__all__ = [
    "PaperOutputBundle",
    "PUBLISHED_PAPER_ANCHOR_KEYS",
    "FO2008_EXPECTED_CASE_IDS",
    "FO2008_EXPECTED_FAMILIES",
    "FO2008_EXPECTED_TABLE_LABELS",
    "JUNIKE_SUMMARY_CASE_IDS",
    "REGRESSION_STRIP_MODELS",
    "PAPER_OUTPUT_BUNDLES",
]
