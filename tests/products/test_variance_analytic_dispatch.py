from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import foureng as fe
from foureng.analytics.bsm_variance import bsm_variance_option_integrated, bsm_variance_swap
from foureng.pipeline import price
from foureng.products import VarianceOption, VarianceSwap


def test_variance_swap_analytic_dispatch_matches_public_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=3.0)
    params = fe.BsmParams(sigma=0.2)
    sampling = np.array([0.25, 0.5, 1.0])
    product = VarianceSwap(maturity=1.0, sampling_times=sampling, notional=2.0)

    actual = price(product, "bsm", "variance_analytic_bsm", fwd, params)
    expected = bsm_variance_swap(
        fe.ForwardSpec(S0=100.0, r=0.05, q=0.01, T=1.0),
        params,
        product,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_variance_swap_analytic_matches_mc_expectation_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.04, q=0.01, T=1.0)
    params = fe.BsmParams(sigma=0.25)
    sampling = np.linspace(1.0 / 12.0, 1.0, 12)
    product = VarianceSwap(maturity=1.0, sampling_times=sampling, notional=1.0)
    mc = fe.MCSpec(n_paths=60_000, seed=11, antithetic=True)

    analytic = price(product, "bsm", "variance_analytic_bsm", fwd, params)
    simulated = price(product, "bsm", "variance_mc", fwd, params, grid=mc)

    assert abs(simulated - analytic) < 0.01


def test_integrated_variance_option_analytic_dispatch_matches_formula():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=2.0)
    params = fe.BsmParams(sigma=0.2)
    product = VarianceOption(
        strike=0.03,
        maturity=1.0,
        cp=1,
        sampling_times=np.array([0.5, 1.0]),
        variance_type="integrated",
    )

    actual = price(product, "bsm", "variance_analytic_bsm", fwd, params)
    expected = bsm_variance_option_integrated(
        fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0),
        params,
        product,
    )

    assert actual == approx(expected, rel=1e-12, abs=1e-12)


def test_realised_variance_option_analytic_rejects():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)
    params = fe.BsmParams(sigma=0.2)
    product = VarianceOption(
        strike=0.03,
        maturity=1.0,
        cp=1,
        sampling_times=np.array([0.25, 0.5, 0.75, 1.0]),
        variance_type="realised",
    )

    with pytest.raises(NotImplementedError, match="integrated-variance options only"):
        price(product, "bsm", "variance_analytic_bsm", fwd, params)


def test_variance_analytic_rejects_non_bsm_model():
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0)
    params = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.3, rho=-0.5, v0=0.04)
    product = VarianceSwap(maturity=1.0, sampling_times=np.array([1.0]))

    with pytest.raises(NotImplementedError, match="model='bsm'"):
        price(product, "heston", "variance_analytic_bsm", fwd, params)
