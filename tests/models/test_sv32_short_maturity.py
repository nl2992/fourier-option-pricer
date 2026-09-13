"""3/2 SV model at short maturities: CF bounded by 1, engines agree, no arbitrage.

Regression for the PyFENG ``Sv32Fft.logp_cf`` overflow. PyFENG sums
``1F1(a; b; -X)`` as a plain Taylor series with ``X ~ 2 / (nu^2 v0 T)``, so for
``v0 = 0.04, nu = 1.2`` it returned ``|phi(1)|`` of about 1e118 at ``T = 0.1``
and 1e30 at ``T = 0.25`` while ``phi(0) = phi(-i) = 1`` still held. SINC and
contour then priced calls below intrinsic. ``sv32_cf`` now sums the closed
form as a Poisson mixture (see ``foureng/models/sv32.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
import foureng.models.sv32 as sv32_module

pytestmark = [pytest.mark.numerical_stability]

MATURITIES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
PARAMS = fe.Sv32Params(v0=0.04, kappa=3.0, theta=0.05, nu=1.2, rho=-0.6)
STRIKES = np.array([60.0, 70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 135.0, 150.0])
U_REAL = np.concatenate([-np.linspace(0.0, 50.0, 101), np.linspace(0.0, 400.0, 801)])


def _fwd(T: float) -> fe.ForwardSpec:
    return fe.ForwardSpec(S0=100.0, r=0.02, q=0.01, T=T)


@pytest.mark.parametrize("T", MATURITIES)
def test_cf_is_bounded_by_one_for_real_u(T):
    phi = fe.sv32_cf(U_REAL, _fwd(T), PARAMS)
    assert np.all(np.isfinite(phi))
    assert np.max(np.abs(phi)) <= 1.0 + 1e-12
    np.testing.assert_allclose(fe.sv32_cf(np.array([0.0, -1j]), _fwd(T), PARAMS), 1.0, atol=1e-12)


@pytest.mark.parametrize("T", MATURITIES)
def test_cf_matches_the_three_halves_limit_of_sv42(T):
    """Independent derivation: 4/2 with a = 0 is the 3/2 model for V = b^2 / v.

    With ``b = 1`` the CIR factor ``v = 1 / V`` has speed ``kappa theta``,
    level ``(kappa + nu^2) / (kappa theta)``, vol-of-vol ``nu`` and
    correlation ``-rho`` (Grasselli 2017, Section 2).
    """
    p = PARAMS
    kt = p.kappa * p.theta
    p42 = fe.Sv42Params(
        v0=1.0 / p.v0,
        kappa=kt,
        theta=(p.kappa + p.nu**2) / kt,
        nu=p.nu,
        rho=-p.rho,
        a=0.0,
        b=1.0,
    )
    u = np.linspace(0.0, 150.0, 151)
    # sv42 sums its Kummer series from k = 0 in log space, which is good to
    # about eps * X (3e-12 at T = 0.05); the mpmath test below is the tight one.
    np.testing.assert_allclose(
        fe.sv32_cf(u, _fwd(T), p), fe.sv42_cf(u, _fwd(T), p42), rtol=0.0, atol=1e-10
    )


@pytest.mark.parametrize("T", MATURITIES)
def test_cf_matches_mpmath(T):
    """Same closed form in 30-digit arithmetic, via Kummer's transformation."""
    mp = pytest.importorskip("mpmath")
    p = PARAMS
    u = np.array([0.3, 1.0, 5.0, 20.0, 60.0, 150.0, 3.0 - 0.5j, 40.0 - 0.75j, -25.0 + 0.4j])

    def ref(x):
        with mp.workdps(30):
            s = 1j * mp.mpc(x)
            nu2 = mp.mpf(p.nu) ** 2
            mu = mp.mpf(0.5) + (p.kappa - s * p.rho * p.nu) / nu2
            delta = mp.sqrt(mu**2 + s * (1 - s) / nu2)
            a, b = delta - mu, 1 + 2 * delta
            kt = mp.mpf(p.kappa) * p.theta
            X = 2 * kt / (nu2 * p.v0 * mp.expm1(kt * T))
            log_pre = mp.loggamma(b - a) - mp.loggamma(b) + a * mp.log(X) - X
            return complex(mp.exp(log_pre) * mp.hyp1f1(b - a, b, X))

    expected = np.array([ref(x) for x in u])
    np.testing.assert_allclose(fe.sv32_cf(u, _fwd(T), p), expected, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("T", MATURITIES)
def test_sinc_and_contour_agree_within_no_arbitrage_bounds(T):
    fwd = _fwd(T)
    sinc = fe.price_strip("sv32", "sinc", STRIKES, fwd, PARAMS)
    contour = fe.price_strip("sv32", "contour", STRIKES, fwd, PARAMS)
    np.testing.assert_allclose(sinc, contour, rtol=0.0, atol=1e-7)

    df = np.exp(-fwd.r * T)
    forward = fwd.S0 * np.exp((fwd.r - fwd.q) * T)
    lower = df * np.maximum(forward - STRIKES, 0.0)
    upper = df * forward
    for prices in (sinc, contour):
        # deep in the money the time value is below 1e-15, so allow rounding
        assert np.all(prices >= lower - 1e-10), f"call below intrinsic at T={T}: {prices - lower}"
        assert np.all(prices <= upper)
        assert np.all(np.diff(prices) <= 1e-12), "calls must decrease in strike"
        slopes = np.diff(prices) / np.diff(STRIKES)
        assert np.all(np.diff(slopes) > -1e-10), "calls must be convex in strike"


def test_pyfeng_fft_uses_the_stable_cf():
    pytest.importorskip("pyfeng")
    fwd = _fwd(0.25)
    got = fe.price_strip("sv32", "pyfeng_fft", STRIKES, fwd, PARAMS)
    ref = fe.price_strip("sv32", "contour", STRIKES, fwd, PARAMS)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-3)


def test_cf_raises_instead_of_returning_modulus_above_one(monkeypatch):
    monkeypatch.setattr(sv32_module, "_sv32_mgf", lambda s, T, p: np.full(np.shape(s), 2.0 + 0j))
    with pytest.raises(FloatingPointError, match=r"\|phi\(u\)\|"):
        fe.sv32_cf(np.array([0.5, 1.0]), _fwd(0.1), PARAMS)
    # complex u is not checked: |phi| > 1 is legitimate off the real axis
    fe.sv32_cf(np.array([1.0 - 0.5j]), _fwd(0.1), PARAMS)
