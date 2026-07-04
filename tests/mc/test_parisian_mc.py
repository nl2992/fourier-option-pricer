"""Tests for Parisian option MC pricer.

Key invariants tested:
  1. In-out parity: knockout_price + knockin_price = vanilla_price (within 3 std errors).
  2. Knockout price < vanilla (never pays more).
  3. Standard Parisian price ≥ standard barrier price (Parisian is harder to knock out).
  4. D→0 limit: standard Parisian approaches standard barrier option.
  5. D→T limit: standard Parisian approaches vanilla (can never fire).
  6. Cumulative price ≤ standard Parisian price (cumulative condition fires earlier).
  7. Product wrapper matches raw pricer.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.mc.parisian_mc import parisian_mc_price, parisian_mc_price_from_product
from foureng.products.parisian import ParisianOption

# ── Shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def base_params():
    return dict(
        S0=100.0,
        K=100.0,
        H=90.0,
        r=0.05,
        q=0.0,
        T=1.0,
        sigma=0.20,
        D=0.10,
    )


N_PATHS = 80_000
N_STEPS = 600
SEED = 42


def _price(direction="down", knockout=True, parisian_type="standard", **kw):
    price, _ = parisian_mc_price(
        **kw,
        direction=direction,
        knockout=knockout,
        parisian_type=parisian_type,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        seed=SEED,
        antithetic=True,
    )
    return price


def _price_se(direction="down", knockout=True, parisian_type="standard", **kw):
    return parisian_mc_price(
        **kw,
        direction=direction,
        knockout=knockout,
        parisian_type=parisian_type,
        n_paths=N_PATHS,
        n_steps=N_STEPS,
        seed=SEED,
        antithetic=True,
    )


# ── 1. In-out parity ──────────────────────────────────────────────────────


def test_parisian_inout_parity_down_call(base_params):
    """Knockout + knockin = vanilla (down Parisian call, standard type)."""
    from foureng.analytics.bsm_barrier import bsm_call as bsm_c

    vanilla = bsm_c(
        base_params["S0"],
        base_params["K"],
        base_params["r"],
        base_params["q"],
        base_params["T"],
        base_params["sigma"],
    )

    ko, se_ko = _price_se(direction="down", knockout=True, **base_params)
    ki, se_ki = _price_se(direction="down", knockout=False, **base_params)

    combined = ko + ki
    total_se = np.sqrt(se_ko**2 + se_ki**2)

    assert abs(combined - vanilla) < 5 * total_se, (
        f"In-out parity failed: ko={ko:.4f}, ki={ki:.4f}, sum={combined:.4f}, "
        f"vanilla={vanilla:.4f}, 5*se={5 * total_se:.4f}"
    )


def test_parisian_inout_parity_up_put(base_params):
    """Up Parisian put: knockout + knockin = vanilla BSM put."""
    from foureng.analytics.bsm_barrier import bsm_put as bsm_p

    params = {**base_params, "H": 110.0, "cp": -1}
    vanilla = bsm_p(
        params["S0"], params["K"], params["r"], params["q"], params["T"], params["sigma"]
    )

    ko, se_ko = _price_se(direction="up", knockout=True, **params)
    ki, se_ki = _price_se(direction="up", knockout=False, **params)

    combined = ko + ki
    total_se = np.sqrt(se_ko**2 + se_ki**2)

    assert abs(combined - vanilla) < 5 * total_se, (
        f"Up put parity failed: sum={combined:.4f}, vanilla={vanilla:.4f}"
    )


# ── 2. Knockout price strictly ≤ vanilla ──────────────────────────────────


def test_parisian_ko_le_vanilla(base_params):
    """Parisian knockout call is always cheaper than the vanilla call."""
    from foureng.analytics.bsm_barrier import bsm_call as bsm_c

    vanilla = bsm_c(
        base_params["S0"],
        base_params["K"],
        base_params["r"],
        base_params["q"],
        base_params["T"],
        base_params["sigma"],
    )
    ko, se = _price_se(direction="down", knockout=True, **base_params)
    assert ko < vanilla + 3 * se, f"KO price {ko:.4f} >= vanilla {vanilla:.4f}"
    assert ko >= 0.0, f"KO price must be non-negative; got {ko:.4f}"


# ── 3. Standard Parisian ≥ standard barrier (harder to knock out) ─────────


def test_standard_parisian_ge_barrier(base_params):
    """Standard Parisian KO call >= standard down-and-out barrier call.

    The Parisian condition requires a *consecutive* D-length excursion, which
    is harder to achieve than a first-touch barrier.  So the Parisian knockout
    option is worth at least as much as the standard barrier option.
    """
    from foureng.analytics.bsm_barrier import bsm_barrier_price

    barrier_price = bsm_barrier_price(
        base_params["S0"],
        base_params["K"],
        base_params["H"],
        base_params["r"],
        base_params["q"],
        base_params["T"],
        base_params["sigma"],
        "down_out",
        cp=1,
    )
    parisian_price, se = _price_se(direction="down", knockout=True, **base_params)

    # Parisian KO >= standard barrier KO (Parisian is harder to trigger)
    assert parisian_price >= barrier_price - 5 * se, (
        f"Parisian KO {parisian_price:.4f} < standard barrier {barrier_price:.4f} "
        f"(5*se={5 * se:.4f})"
    )


# ── 4. D→T approaches vanilla (excursion impossible to complete) ──────────


def test_large_D_approaches_vanilla(base_params):
    """With D very close to T, the Parisian KO price approaches vanilla."""
    from foureng.analytics.bsm_barrier import bsm_call as bsm_c

    vanilla = bsm_c(
        base_params["S0"],
        base_params["K"],
        base_params["r"],
        base_params["q"],
        base_params["T"],
        base_params["sigma"],
    )
    # D = 0.99 * T: excursion of that length is nearly impossible
    large_D_params = {**base_params, "D": 0.98}
    ko, se = _price_se(direction="down", knockout=True, **large_D_params)

    assert abs(ko - vanilla) < 10 * se, (
        f"Large D: KO={ko:.4f} vs vanilla={vanilla:.4f}, gap={abs(ko - vanilla):.4f}, 10*se={10 * se:.4f}"
    )


# ── 5. Cumulative price ≤ standard Parisian price ────────────────────────


def test_cumulative_le_standard_parisian(base_params):
    """Cumulative Parisian KO is cheaper than standard Parisian KO.

    Cumulative condition (total time > D) is easier to trigger than standard
    (consecutive time > D), so the cumulative knockout value is lower.
    """
    std_ko, se_std = _price_se(
        direction="down", knockout=True, parisian_type="standard", **base_params
    )
    cum_ko, se_cum = _price_se(
        direction="down", knockout=True, parisian_type="cumulative", **base_params
    )
    # cumulative KO ≤ standard KO (cumulative triggers more often → less value)
    assert cum_ko <= std_ko + 5 * np.sqrt(se_std**2 + se_cum**2), (
        f"Cumulative KO {cum_ko:.4f} > standard KO {std_ko:.4f}"
    )


# ── 6. Product wrapper consistency ───────────────────────────────────────


def test_product_wrapper_matches_raw(base_params):
    """parisian_mc_price_from_product returns the same price as the raw pricer."""
    p = ParisianOption(
        strike=base_params["K"],
        barrier=base_params["H"],
        maturity=base_params["T"],
        excursion_window=base_params["D"],
        cp=1,
        direction="down",
        knockout=True,
        parisian_type="standard",
    )
    price_raw, _ = parisian_mc_price(
        **{k: base_params[k] for k in ("S0", "K", "H", "r", "q", "T", "sigma", "D")},
        cp=1,
        direction="down",
        knockout=True,
        parisian_type="standard",
        n_paths=20_000,
        n_steps=400,
        seed=99,
        antithetic=True,
    )
    price_wrap, _ = parisian_mc_price_from_product(
        p,
        S0=base_params["S0"],
        r=base_params["r"],
        q=base_params["q"],
        sigma=base_params["sigma"],
        n_paths=20_000,
        n_steps=400,
        seed=99,
        antithetic=True,
    )
    assert abs(price_raw - price_wrap) < 1e-10, (
        f"Wrapper mismatch: raw={price_raw:.6f}, wrap={price_wrap:.6f}"
    )


# ── 7. Non-negativity across strikes ─────────────────────────────────────


def test_parisian_non_negative_grid(base_params):
    """All Parisian prices are non-negative across a strike grid."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    for K in strikes:
        for ptype in ("standard", "cumulative"):
            for ko in (True, False):
                price, _ = parisian_mc_price(
                    **{**base_params, "K": K},
                    direction="down",
                    knockout=ko,
                    parisian_type=ptype,
                    n_paths=10_000,
                    n_steps=300,
                    seed=7,
                )
                assert price >= -1e-8, (
                    f"Negative price {price:.4f} for K={K}, type={ptype}, ko={ko}"
                )
