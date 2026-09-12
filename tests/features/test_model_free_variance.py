"""Model-free (log-contract) variance replicated from option strips."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=30.0 / 365.0)


def _strip(model, params, log_lo, log_hi, n, fwd=FWD):
    K = fwd.F0 * np.exp(np.linspace(log_lo, log_hi, n))
    calls = fe.price_strip(model, "contour", K, fwd, params, cp=1)
    puts = fe.price_strip(model, "contour", K, fwd, params, cp=-1)
    return K, calls, puts


def test_bsm_strip_recovers_sigma_squared():
    K, C, P = _strip("bsm", fe.BsmParams(0.2), -0.6, 0.6, 801)
    var = fe.log_contract_variance_from_strip(K, FWD, calls=C, puts=P)
    assert var == pytest.approx(0.04, rel=1e-8)
    assert fe.log_contract_variance("bsm", FWD, fe.BsmParams(0.2)) == pytest.approx(0.04, rel=1e-12)


def test_heston_strip_recovers_expected_integrated_variance():
    p = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.05)
    K, C, P = _strip("heston", p, -1.5, 1.0, 1201)
    exact = p.theta + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * FWD.T)) / (p.kappa * FWD.T)
    assert fe.log_contract_variance_from_strip(K, FWD, calls=C, puts=P) == pytest.approx(
        exact, rel=1e-7
    )


def test_jumps_separate_the_log_contract_from_the_variance_swap():
    """Kou: the strip gives -2 c1 / T; realised variance (c2 / T) is ~10% higher."""
    p = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
    K, C, P = _strip("kou", p, -6.0, 3.0, 1601)
    strip = fe.log_contract_variance_from_strip(K, FWD, calls=C, puts=P)
    assert strip == pytest.approx(fe.log_contract_variance("kou", FWD, p), rel=1e-6)
    swap = fe.levy_variance_fair_strike("kou", FWD, p, np.linspace(0.0, FWD.T, 31)[1:])
    assert swap > 1.05 * strip


def test_calls_only_puts_only_and_both_agree():
    p = fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1)
    K, C, P = _strip("nig", p, -1.0, 1.0, 401)
    both = fe.log_contract_variance_from_strip(K, FWD, calls=C, puts=P)
    assert fe.log_contract_variance_from_strip(K, FWD, calls=C) == pytest.approx(both, rel=1e-10)
    assert fe.log_contract_variance_from_strip(K, FWD, puts=P) == pytest.approx(both, rel=1e-10)


def test_truncated_strip_underestimates():
    p = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.05)
    K_wide, C_wide, _ = _strip("heston", p, -1.5, 1.0, 801)
    K_narrow, C_narrow, _ = _strip("heston", p, -0.1, 0.1, 81)
    wide = fe.log_contract_variance_from_strip(K_wide, FWD, calls=C_wide)
    narrow = fe.log_contract_variance_from_strip(K_narrow, FWD, calls=C_narrow)
    assert narrow < wide


def test_vix_style_index_converges_with_strike_spacing():
    params = fe.BsmParams(0.2)
    errors = []
    for dk in (5.0, 1.0, 0.25):
        K = np.arange(50.0, 200.0 + 1e-9, dk)
        C = fe.price_strip("bsm", "contour", K, FWD, params)
        P = fe.price_strip("bsm", "contour", K, FWD, params, cp=-1)
        errors.append(abs(fe.vix_style_index(K, FWD, calls=C, puts=P) - 20.0))
    assert errors[0] > errors[1] > errors[2] and errors[2] < 0.01


def test_input_validation():
    K = np.array([90.0, 100.0, 110.0])
    C = np.array([11.0, 3.0, 0.5])
    with pytest.raises(ValueError, match="increasing"):
        fe.log_contract_variance_from_strip(K[::-1], FWD, calls=C)
    with pytest.raises(ValueError, match="bracket"):
        fe.log_contract_variance_from_strip(K + 50.0, FWD, calls=C)
    with pytest.raises(ValueError, match="calls, puts"):
        fe.log_contract_variance_from_strip(K, FWD)
    with pytest.raises(ValueError, match="same shape"):
        fe.log_contract_variance_from_strip(K, FWD, calls=C[:2])
    with pytest.raises(ValueError, match="unknown model"):
        fe.log_contract_variance("nope", FWD, None)
