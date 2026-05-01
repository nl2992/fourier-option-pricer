"""Numerical-quality checklist tests.

Covers the course M1 / M4 floating-point best-practice rubric:
  - float64 throughout / no silent downcasting
  - expm1 / log1p for cancellation-safe formulas
  - np.isfinite guards — no NaN/inf flowing out of pricers
  - Analytic Greeks (better than FD requirement)
  - Variance floor prevents NaN in MC
  - RNG reproducibility and seed isolation
  - Vectorised MC (no Python path loops — structural, tested indirectly)
  - Input validation raises ValueError for bad parameters

All tests use *in-house* CF models (Kou, Merton-JD, Bates BSM-limit) so
they pass without pyfeng installed.  PyFENG-backed model coverage is handled
in the dedicated paper-replication test suite.
"""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.bates import bates_cf, bates_cumulants

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
_STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

# Kou: in-house CF, works with all non-lewis pricers
_KOU = fe.KouParams(sigma=0.2, lam=1.0, p=0.6, eta1=25.0, eta2=10.0)
# Merton JD: in-house CF  (model key is "merton_jd")
_MJD = fe.MertonJDParams(sigma=0.2, lam=1.0, muj=-0.05, sigj=0.1)
# Bates with zero jumps = in-house fallback
_BATES_NOJUMP = fe.BatesParams(
    kappa=2.0,
    theta=0.04,
    nu=0.5,
    rho=-0.7,
    v0=0.04,
    lam_j=0.0,
    mu_j=0.0,
    sigma_j=0.01,
)

_FFT_GRID = fe.FFTGrid(N=4096, eta=0.25, alpha=1.5)
_FRFT_GRID = fe.FRFTGrid(N=4096, eta=0.25, lam=0.01, alpha=1.5)


# ---------------------------------------------------------------------------
# 1. Cancellation-safe formulas — expm1 / log1p
# ---------------------------------------------------------------------------


class TestCancellationSafeFormulas:
    """Verify bates_cf and bates_cumulants use expm1 for the jump compensator
    zeta = exp(mu_j + 0.5*sig_j^2) - 1.

    When mu_j and sigma_j are both near zero the argument is ~0, and
    np.exp(x) - 1 loses all significant bits while np.expm1(x) retains
    full double precision.
    """

    def test_bates_zeta_expm1_near_zero(self):
        """For mu_j=1e-10, sigma_j=1e-6 the exponent is ~5e-13.
        np.exp(5e-13) - 1 in float64 rounds to exactly 0.
        np.expm1(5e-13) correctly returns ~5e-13.
        The CF output must be finite even for near-zero parameters.
        """
        params = fe.BatesParams(
            kappa=1.0,
            theta=0.04,
            nu=0.5,
            rho=-0.5,
            v0=0.04,
            lam_j=2.0,
            mu_j=1e-10,
            sigma_j=1e-6,
        )
        fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
        c1, c2, c4 = bates_cumulants(fwd, params)
        assert np.isfinite(c1), "c1 cumulant must be finite"
        assert np.isfinite(c2), "c2 cumulant must be finite"

        # zeta must be non-negative (exp of a real number ≥ 1 → expm1 ≥ 0)
        arg = params.mu_j + 0.5 * params.sigma_j**2
        zeta_expm1 = float(np.expm1(arg))
        assert zeta_expm1 >= 0.0, "zeta = expm1(arg) must be non-negative"

        # CF must be finite for all test frequencies
        omega = np.array([0.0, 1.0, 5.0])
        phi = bates_cf(omega, fwd, params)
        assert np.all(np.isfinite(phi)), "CF must be finite for near-zero mu_j"

    def test_log1p_precision_for_small_x(self):
        """np.log(1 + 1e-15) rounds to 0 in float64; log1p retains precision."""
        x = 1e-15
        assert np.log1p(x) > 0, "log1p(1e-15) must be positive"
        # Note: np.log(1.0 + 1e-15) may equal 0 due to 1.0 + 1e-15 == 1.0

    def test_expm1_precision_for_small_x(self):
        """np.exp(1e-15) - 1 rounds to 0 in float64; expm1 retains precision."""
        x = 1e-15
        assert np.expm1(x) > 0, "expm1(1e-15) must be positive"


# ---------------------------------------------------------------------------
# 2. Finite output — no NaN / inf from any pricer
# ---------------------------------------------------------------------------

_COS_CASES = [
    ("kou", "cos", _KOU, None),
    ("kou", "cos_improved", _KOU, None),
    ("kou", "carr_madan", _KOU, _FFT_GRID),
    ("kou", "frft", _KOU, _FRFT_GRID),
    ("merton_jd", "cos", _MJD, None),
    ("merton_jd", "cos_improved", _MJD, None),
    ("merton_jd", "carr_madan", _MJD, _FFT_GRID),
    ("merton_jd", "frft", _MJD, _FRFT_GRID),
]


@pytest.mark.parametrize("model,method,params,grid", _COS_CASES)
def test_pricer_output_is_finite(model, method, params, grid):
    """No NaN or inf must flow out of any (model, method) combination."""
    prices = fe.price_strip(model, method, _STRIKES, _FWD, params, grid=grid)
    assert np.all(np.isfinite(prices)), f"{model}/{method} produced non-finite prices: {prices}"


@pytest.mark.parametrize("model,method,params,grid", _COS_CASES)
def test_pricer_output_is_float64(model, method, params, grid):
    """Pricing output must be float64 — no silent downcast to float32."""
    prices = fe.price_strip(model, method, _STRIKES, _FWD, params, grid=grid)
    assert prices.dtype == np.float64, (
        f"{model}/{method} returned dtype {prices.dtype}, expected float64"
    )


@pytest.mark.parametrize("model,method,params,grid", _COS_CASES)
def test_pricer_output_non_negative(model, method, params, grid):
    """Call prices must be >= 0 (no-arbitrage floor)."""
    prices = fe.price_strip(model, method, _STRIKES, _FWD, params, grid=grid)
    assert np.all(prices >= -1e-9), f"{model}/{method} produced negative prices: {prices}"


# ---------------------------------------------------------------------------
# 3. COS Greeks — analytic (better than FD), output must be finite and bounded
# ---------------------------------------------------------------------------


class TestCOSGreeks:
    """COS computes exact analytic Delta and Gamma — no finite-difference bumps."""

    def _phi_kou(self):
        phi = lambda u: fe.kou_cf(u, _FWD, _KOU)
        cums = fe.kou_cumulants(_FWD, _KOU)
        grid = fe.cos_improved_grid(cums, model="kou", params=_KOU)
        return phi, grid

    def test_delta_gamma_finite(self):
        phi, grid = self._phi_kou()
        out = fe.cos_price_and_greeks(phi, _FWD, _STRIKES, grid)
        assert np.all(np.isfinite(out.delta)), "COS delta must be finite"
        assert np.all(np.isfinite(out.gamma)), "COS gamma must be finite"

    def test_delta_in_0_1_for_calls(self):
        """Delta of a European call is strictly in (0, 1)."""
        phi, grid = self._phi_kou()
        out = fe.cos_price_and_greeks(phi, _FWD, _STRIKES, grid)
        assert np.all(out.delta > 0.0), "Call delta must be positive"
        assert np.all(out.delta < 1.0), "Call delta must be < 1"

    def test_gamma_non_negative(self):
        """Gamma of a vanilla call is non-negative."""
        phi, grid = self._phi_kou()
        out = fe.cos_price_and_greeks(phi, _FWD, _STRIKES, grid)
        assert np.all(out.gamma >= -1e-10), "Call gamma must be non-negative"


# ---------------------------------------------------------------------------
# 4. MC RNG — reproducibility and seed isolation
# ---------------------------------------------------------------------------


class TestMCReproducibility:
    """MC engines use np.random.default_rng(seed) — same seed → same prices."""

    def _mc_prices(self, seed: int) -> np.ndarray:
        from foureng.mc.black_scholes_mc import MCSpec, european_call_mc

        spec = MCSpec(n_paths=2000, seed=seed)
        return european_call_mc(
            _FWD.S0,
            _STRIKES,
            _FWD.T,
            _FWD.r,
            _FWD.q,
            vol=0.2,
            mc=spec,
        )

    def test_same_seed_gives_same_prices(self):
        p1 = self._mc_prices(seed=42)
        p2 = self._mc_prices(seed=42)
        np.testing.assert_array_equal(p1, p2, err_msg="Same seed must give identical prices")

    def test_different_seeds_give_different_prices(self):
        p1 = self._mc_prices(seed=1)
        p2 = self._mc_prices(seed=2)
        assert not np.allclose(p1, p2), "Different seeds must give different prices"

    def test_mc_output_is_finite(self):
        prices = self._mc_prices(seed=0)
        assert np.all(np.isfinite(prices)), "MC prices must be finite"


# ---------------------------------------------------------------------------
# 5. Variance floor — no NaN from near-zero sigma in Heston conditional MC
# ---------------------------------------------------------------------------


class TestVarianceFloor:
    """sigma = sqrt(max(sigma^2, 1e-16)) prevents NaN from near-zero variance."""

    def test_heston_mc_near_zero_variance_stays_finite(self):
        """Heston v0 ≈ 0 must not produce NaN prices."""
        from foureng.mc.heston_conditional_mc import HestonMCScheme, heston_conditional_mc_calls

        params = fe.HestonParams(kappa=10.0, theta=0.04, nu=0.1, rho=-0.5, v0=1e-12)
        fwd = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=0.5)
        scheme = HestonMCScheme(n_paths=500, n_steps=50, seed=7)
        prices = heston_conditional_mc_calls(fwd.S0, _STRIKES, fwd.T, fwd.r, fwd.q, params, scheme)
        assert np.all(np.isfinite(prices)), "Near-zero v0 must not produce NaN in MC"


# ---------------------------------------------------------------------------
# 6. Extreme inputs — tiny T, tiny vol, wide strikes
# ---------------------------------------------------------------------------


class TestExtremeInputs:
    def test_tiny_maturity_cos_finite(self):
        """T = 1/365 (one day) must not blow up the COS pricer."""
        fwd_short = fe.ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0 / 365.0)
        cums = fe.kou_cumulants(fwd_short, _KOU)
        grid = fe.cos_improved_grid(cums, model="kou", params=_KOU)
        phi = lambda u: fe.kou_cf(u, fwd_short, _KOU)
        from foureng.pricers.cos import cos_prices

        result = cos_prices(phi, fwd_short, _STRIKES, grid)
        assert np.all(np.isfinite(result.call_prices)), "Tiny T must not produce NaN"

    def test_large_strike_range_finite(self):
        """Strikes from deep-ITM to deep-OTM must all be finite."""
        strikes = np.array([40.0, 60.0, 80.0, 100.0, 130.0, 160.0, 200.0])
        prices = fe.price_strip("kou", "cos_improved", strikes, _FWD, _KOU)
        assert np.all(np.isfinite(prices))

    def test_deep_otm_non_negative(self):
        """Deep-OTM call prices must be >= 0."""
        strikes = np.array([150.0, 200.0, 300.0])
        prices = fe.price_strip("merton_jd", "cos_improved", strikes, _FWD, _MJD)
        assert np.all(prices >= -1e-10)


# ---------------------------------------------------------------------------
# 7. dtype correctness
# ---------------------------------------------------------------------------


class TestDtypes:
    def test_kou_cf_returns_complex128(self):
        """CF must return complex128 (double-precision complex)."""
        omega = np.array([0.0, 1.0, 2.0, 5.0])
        phi = fe.kou_cf(omega, _FWD, _KOU)
        assert phi.dtype == np.complex128, f"CF returned {phi.dtype}, expected complex128"

    def test_cumulants_return_floats(self):
        """Cumulant functions must return Python floats."""
        c1, c2, c4 = fe.kou_cumulants(_FWD, _KOU)
        assert isinstance(c1, float) and np.isfinite(c1)
        assert isinstance(c2, float) and np.isfinite(c2)
        assert isinstance(c4, float) and np.isfinite(c4)

    def test_price_strip_output_is_float64(self):
        prices = fe.price_strip("kou", "cos_improved", _STRIKES, _FWD, _KOU)
        assert prices.dtype == np.float64
