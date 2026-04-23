"""FO2008 paper-case registry.

Encodes the benchmark setups, reference values, paper error/time columns and
N-grids for the replication notebook ``notebooks/fo2008_replication.ipynb``.

Source: Fang & Oosterlee (2008), "A Novel Pricing Method for European Options
Based on Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2).

Each :class:`PaperCase` carries *only* what the paper reports — our own
errors and timings are computed in the notebook against ``reference_values``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperCase:
    """One replication case from FO2008.

    Attributes
    ----------
    case_id :
        Stable registry key (e.g. ``"heston_table4_t1"``).
    table_id :
        Human-readable table label (e.g. ``"Table 4"``).
    model :
        Pipeline model name: ``"bsm" | "heston" | "vg" | "cgmy"``.
    strikes :
        Strike list. Single-scalar cases still use a 1-element list.
    maturity :
        Time to maturity ``T`` (years).
    params :
        Model-specific parameters as a plain dict; the notebook converts
        these to the right ``*Params`` dataclass.
    forward :
        ``S0`` for the forward spec (``r``, ``q`` live in ``params``).
    reference_values :
        Ground-truth price(s) for error computation. Float for scalar
        cases, list for strip cases.
    reference_source :
        Short string tag — ``"FO2008_summary"`` for single reference
        points, ``"FO2008_strip"`` for the 21-strike strip, or
        ``"BS_closed_form"`` for Table 2.
    Ns :
        N-grid used in the paper (and reused for our COS runs).
    paper_errors :
        The paper's reported errors on that N-grid (same length as ``Ns``).
    paper_times_ms :
        The paper's reported CPU time in milliseconds.
    extras :
        Free-form dict for table-specific knobs — e.g. the COS truncation
        ``L`` the paper used, extra grid bounds mentioned in the summary,
        or ``"cm_max_err_paper"`` for Table 2.
    """

    case_id: str
    table_id: str
    model: str
    strikes: list[float]
    maturity: float
    params: dict
    forward: float
    reference_values: Any
    reference_source: str
    Ns: list[int]
    paper_errors: list[float] | None = None
    paper_times_ms: list[float] | None = None
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Table 2 — BSM, COS vs Carr-Madan
# ---------------------------------------------------------------------------

BSM_TABLE2 = PaperCase(
    case_id="bsm_table2",
    table_id="Table 2",
    model="bsm",
    strikes=[80.0, 100.0, 120.0],
    maturity=0.1,
    params={"sigma": 0.25, "r": 0.1, "q": 0.0},
    forward=100.0,
    # Analytic Black-Scholes prices the paper reports:
    reference_values=[20.7992, 3.6600, 0.0446],
    reference_source="BS_closed_form",
    Ns=[32, 64, 128, 256, 512],
    # These are the paper's max-absolute-error across the three strikes
    # for the COS column (Table 2).
    paper_errors=[2.43e-07, 3.55e-15, 3.55e-15, 3.55e-15, 3.55e-15],
    paper_times_ms=[0.0303, 0.0327, 0.0349, 0.0434, 0.0588],
    extras={
        "L": 10.0,
        "cm_paper_errors": [9.77e-01, 1.23e+00, 7.84e-02, 6.04e-04, 4.12e-04],
        "cm_paper_times_ms": [0.0857, 0.0791, 0.0853, 0.0907, 0.1111],
    },
)


# ---------------------------------------------------------------------------
# Tables 4, 5, 6 — Heston
# ---------------------------------------------------------------------------

_HESTON_BASE = {
    "kappa": 1.5768,
    "theta": 0.0398,
    "nu": 0.5751,
    "rho": -0.5711,
    "v0": 0.0175,
    "r": 0.0,
    "q": 0.0,
}

HESTON_TABLE4_T1 = PaperCase(
    case_id="heston_table4_t1",
    table_id="Table 4",
    model="heston",
    strikes=[100.0],
    maturity=1.0,
    params=dict(_HESTON_BASE),
    forward=100.0,
    reference_values=5.78515545,
    reference_source="FO2008_summary",
    Ns=[40, 80, 120, 160, 200],
    paper_errors=[4.69e-02, 3.81e-04, 1.17e-05, 6.18e-07, 3.70e-09],
    paper_times_ms=[0.0607, 0.0805, 0.1078, 0.1300, 0.1539],
    extras={"L": 10.0},
)

HESTON_TABLE5_T10 = PaperCase(
    case_id="heston_table5_t10",
    table_id="Table 5",
    model="heston",
    strikes=[100.0],
    maturity=10.0,
    params=dict(_HESTON_BASE),
    forward=100.0,
    reference_values=22.318945791,
    reference_source="FO2008_summary",
    Ns=[40, 65, 90, 115, 140],
    paper_errors=[4.96e-01, 4.63e-03, 1.35e-05, 1.08e-07, 9.88e-10],
    paper_times_ms=[0.0598, 0.0747, 0.0916, 0.1038, 0.1230],
    extras={"L": 32.0},
)

# 21-strike strip K = 50, 55, ..., 150
_HESTON_T6_STRIKES = [float(k) for k in range(50, 151, 5)]

HESTON_TABLE6_STRIP = PaperCase(
    case_id="heston_table6_strip",
    table_id="Table 6",
    model="heston",
    strikes=_HESTON_T6_STRIKES,
    maturity=1.0,
    params=dict(_HESTON_BASE),
    forward=100.0,
    # No closed form reference for the full strip in the paper summary;
    # we derive a numerical reference in the notebook from a large-N COS
    # run and record that. For the registry, flag it explicitly.
    reference_values=None,
    reference_source="FO2008_strip",
    Ns=[40, 80, 160, 200],
    paper_errors=[5.19e-02, 7.18e-04, 6.18e-07, 2.05e-08],
    paper_times_ms=[0.1015, 0.1766, 0.3383, 0.4214],
    extras={"L": 10.5, "reference_N": 1024},
)


# ---------------------------------------------------------------------------
# Table 7 — Variance Gamma (two maturities)
# ---------------------------------------------------------------------------

_VG_BASE = {"sigma": 0.12, "nu": 0.2, "theta": -0.14, "r": 0.1, "q": 0.0}

VG_TABLE7_T01 = PaperCase(
    case_id="vg_table7_t01",
    table_id="Table 7 (T=0.1)",
    model="vg",
    strikes=[90.0],
    maturity=0.1,
    params=dict(_VG_BASE),
    forward=100.0,
    reference_values=10.993703187,
    reference_source="FO2008_summary",
    Ns=[128, 256, 512, 1024, 2048],
    paper_errors=[6.97e-04, 4.19e-06, 6.80e-06, 5.70e-07, 7.98e-08],
    paper_times_ms=None,
    extras={"L": 10.0},
)

VG_TABLE7_T1 = PaperCase(
    case_id="vg_table7_t1",
    table_id="Table 7 (T=1.0)",
    model="vg",
    strikes=[90.0],
    maturity=1.0,
    params=dict(_VG_BASE),
    forward=100.0,
    reference_values=19.099354724,
    reference_source="FO2008_summary",
    Ns=[30, 60, 90, 120, 150],
    paper_errors=[7.06e-03, 1.29e-05, 2.81e-07, 3.16e-08, 1.51e-09],
    paper_times_ms=None,
    extras={"L": 10.0},
)


# ---------------------------------------------------------------------------
# Tables 8, 9, 10 — CGMY
# ---------------------------------------------------------------------------

_CGMY_BASE = {"C": 1.0, "G": 5.0, "M": 5.0, "r": 0.1, "q": 0.0}

CGMY_TABLE8_Y05 = PaperCase(
    case_id="cgmy_table8_y05",
    table_id="Table 8 (Y=0.5)",
    model="cgmy",
    strikes=[100.0],
    maturity=1.0,
    params={**_CGMY_BASE, "Y": 0.5},
    forward=100.0,
    reference_values=19.812948843,
    reference_source="FO2008_summary",
    Ns=[40, 60, 80, 100, 120, 140],
    paper_errors=[3.82e-02, 6.87e-04, 2.11e-05, 9.45e-07, 5.56e-08, 4.04e-09],
    paper_times_ms=[0.0560, 0.0645, 0.0844, 0.1280, 0.1051, 0.1216],
    extras={"trunc_ab": (-5.0, 5.0)},
)

CGMY_TABLE9_Y15 = PaperCase(
    case_id="cgmy_table9_y15",
    table_id="Table 9 (Y=1.5)",
    model="cgmy",
    strikes=[100.0],
    maturity=1.0,
    params={**_CGMY_BASE, "Y": 1.5},
    forward=100.0,
    reference_values=49.790905469,
    reference_source="FO2008_summary",
    Ns=[40, 45, 50, 55, 60, 65],
    paper_errors=[1.38e+00, 1.98e-02, 4.52e-04, 9.59e-06, 1.22e-09, 7.53e-10],
    paper_times_ms=[0.0545, 0.0589, 0.0689, 0.0690, 0.0732, 0.0748],
    extras={"trunc_ab": (-15.0, 15.0)},
)

CGMY_TABLE10_Y198 = PaperCase(
    case_id="cgmy_table10_y198",
    table_id="Table 10 (Y=1.98)",
    model="cgmy",
    strikes=[100.0],
    maturity=1.0,
    params={**_CGMY_BASE, "Y": 1.98},
    forward=100.0,
    reference_values=99.999905510,
    reference_source="FO2008_summary",
    Ns=[20, 25, 30, 35, 40],
    paper_errors=[4.17e-02, 5.15e-01, 6.54e-05, 1.10e-09, 1.94e-15],
    paper_times_ms=[0.0463, 0.0438, 0.0485, 0.0511, 0.0538],
    extras={"trunc_ab": (-100.0, 20.0)},
)


CASES: dict[str, PaperCase] = {
    c.case_id: c
    for c in (
        BSM_TABLE2,
        HESTON_TABLE4_T1,
        HESTON_TABLE5_T10,
        HESTON_TABLE6_STRIP,
        VG_TABLE7_T01,
        VG_TABLE7_T1,
        CGMY_TABLE8_Y05,
        CGMY_TABLE9_Y15,
        CGMY_TABLE10_Y198,
    )
}


__all__ = ["PaperCase", "CASES"] + [c.case_id.upper() for c in CASES.values()]
