"""Time-changed Levy models (Carr, Geman, Madan & Yor 2003)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY
from foureng.models.time_changed import _raw_exponent

U = np.concatenate([np.linspace(0.0, 5.0, 11), np.linspace(5.0, 200.0, 20)])
BASES = {
    "bsm": fe.BsmParams(0.2),
    "vg": fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14),
    "nig": fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1),
    "cgmy": fe.CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.8),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "merton_jd": fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.15),
}
CIR = fe.CirClock(y0=1.0, kappa=1.5, eta=1.2, lam=1.1)
GOU = fe.GammaOUClock(y0=0.8, lam=2.0, a=1.5, b=1.2)


@pytest.mark.parametrize("base", list(BASES))
@pytest.mark.parametrize("T", [0.3, 1.7])
def test_base_exponents_reproduce_the_registry_cf(base, T):
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=T)
    psi = _raw_exponent(base, BASES[base], U)
    psi_mart = _raw_exponent(base, BASES[base], np.array([-1j]))[0]
    np.testing.assert_allclose(
        np.exp(T * (psi - 1j * U * psi_mart)),
        MODEL_REGISTRY[base].cf(U, fwd, BASES[base]),
        atol=1e-13,
    )


def test_vg_on_a_cir_clock_is_vgsa():
    G, M = 5.0, 8.0  # unit-activity VG: nu = 1, theta = 1/M - 1/G, sigma^2 = 2/(GM)
    vg = fe.VGParams(sigma=np.sqrt(2.0 / (G * M)), nu=1.0, theta=1.0 / M - 1.0 / G)
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.9)
    tc = fe.TimeChangedLevyParams("vg", vg, fe.CirClock(y0=1.3, kappa=2.0, eta=1.1, lam=0.8))
    vgsa = fe.VGSAParams(C=1.3, G=G, M=M, kappa=2.0, eta=1.1, lam=0.8)
    np.testing.assert_allclose(
        fe.time_changed_levy_cf(U, fwd, tc), fe.vgsa_cf(U, fwd, vgsa), atol=1e-13
    )


@pytest.mark.parametrize("base", list(BASES))
def test_deterministic_clocks_reduce_to_the_base(base):
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.8)
    flat = fe.TimeChangedLevyParams(
        base, BASES[base], fe.CirClock(y0=1.0, kappa=3.0, eta=1.0, lam=0.0)
    )
    ref = MODEL_REGISTRY[base].cf(U, fwd, BASES[base])
    np.testing.assert_allclose(fe.time_changed_levy_cf(U, fwd, flat), ref, atol=1e-13)
    # Gamma-OU without jumps: y decays deterministically, clock time y0 (1 - e^{-lam T}) / lam.
    lam, y0 = 2.0, 1.5
    decay = fe.TimeChangedLevyParams(
        base, BASES[base], fe.GammaOUClock(y0=y0, lam=lam, a=0.0, b=1.0)
    )
    t_eff = y0 * (1.0 - np.exp(-lam * fwd.T)) / lam
    ref_eff = MODEL_REGISTRY[base].cf(
        U, fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=t_eff), BASES[base]
    )
    np.testing.assert_allclose(fe.time_changed_levy_cf(U, fwd, decay), ref_eff, atol=1e-13)


def test_clock_means_match_closed_forms():
    t, h = 1.3, 1e-6
    for clock, mean in (
        (
            CIR,
            CIR.y0 * (1 - np.exp(-CIR.kappa * t)) / CIR.kappa
            + CIR.eta * (t - (1 - np.exp(-CIR.kappa * t)) / CIR.kappa),
        ),
        (
            GOU,
            GOU.y0 * (1 - np.exp(-GOU.lam * t)) / GOU.lam
            + GOU.a / GOU.b * (t - (1 - np.exp(-GOU.lam * t)) / GOU.lam),
        ),
    ):
        d = clock.log_mgf(np.array([h]), t)[0] - clock.log_mgf(np.array([-h]), t)[0]
        assert np.real(d) / (2 * h) == pytest.approx(mean, rel=1e-7)


def _clock_paths(clock, T, n=200_000, steps=400, seed=2):
    rng = np.random.default_rng(seed)
    if isinstance(clock, fe.CirClock):
        dt = T / steps
        k, eta, lam = clock.kappa, clock.eta, clock.lam
        c = lam * lam * (1 - np.exp(-k * dt)) / (4 * k)
        d = 4 * k * eta / (lam * lam)
        y, Y = np.full(n, clock.y0), np.zeros(n)
        for _ in range(steps):
            yn = c * rng.noncentral_chisquare(d, y * np.exp(-k * dt) / c)
            Y += 0.5 * (y + yn) * dt
            y = yn
        return Y
    lam, a, b = clock.lam, clock.a, clock.b
    Y = np.full(n, clock.y0 * (1 - np.exp(-lam * T)) / lam)
    counts = rng.poisson(lam * a * T, n)
    for i in range(int(counts.max())):
        tau, jump = rng.uniform(0, T, n), rng.exponential(1 / b, n)
        Y += np.where(counts > i, jump * (1 - np.exp(-lam * (T - tau))) / lam, 0.0)
    return Y


@pytest.mark.parametrize("clock", [CIR, GOU], ids=["cir", "gamma_ou"])
def test_bsm_base_matches_exact_clock_simulation(clock):
    """Given the clock, a BSM base is lognormal: price by the conditional Black
    formula with E[F_cond] = F as control variate."""
    sig = 0.2
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0)
    K = np.array([80.0, 100.0, 125.0])
    Y = _clock_paths(clock, fwd.T)
    log_norm = float(np.real(clock.log_mgf(np.array([0.5 * sig * sig]), fwd.T)[0]))
    f_cond = fwd.F0 * np.exp(-log_norm + 0.5 * sig * sig * Y)
    sd = sig * np.sqrt(Y)
    d1 = (np.log(f_cond[None, :] / K[:, None]) + 0.5 * sd * sd) / sd
    pay = fwd.disc * (f_cond * ndtr(d1) - K[:, None] * ndtr(d1 - sd))
    cv = f_cond - fwd.F0
    est = pay - np.array([np.cov(row, cv)[0, 1] / cv.var() for row in pay])[:, None] * cv
    se = est.std(axis=1) / np.sqrt(est.shape[1])
    got = fe.price_strip(
        "time_changed_levy",
        "contour",
        K,
        fwd,
        fe.TimeChangedLevyParams("bsm", fe.BsmParams(sig), clock),
    )
    assert np.all(np.abs(got - est.mean(axis=1)) < 4.0 * se)


@pytest.mark.parametrize("base,clock", [("nig", CIR), ("cgmy", GOU), ("kou", CIR)])
def test_martingale_and_engine_agreement(base, clock):
    fwd = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.7)
    p = fe.TimeChangedLevyParams(base, BASES[base], clock)
    assert fe.time_changed_levy_cf(np.array([-1j]), fwd, p)[0] == pytest.approx(1.0, abs=1e-12)
    K = np.linspace(75.0, 130.0, 12)
    np.testing.assert_allclose(
        fe.price_strip("time_changed_levy", "cos_improved", K, fwd, p),
        fe.price_strip("time_changed_levy", "contour", K, fwd, p),
        atol=5e-10,
    )


def test_validation():
    with pytest.raises(ValueError, match="base_model"):
        fe.TimeChangedLevyParams("heston", fe.BsmParams(0.2), CIR)
    with pytest.raises(TypeError, match="clock"):
        fe.TimeChangedLevyParams("bsm", fe.BsmParams(0.2), (1.0, 1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="kappa"):
        fe.CirClock(y0=1.0, kappa=0.0, eta=1.0, lam=0.1)
    exploding = fe.TimeChangedLevyParams(
        "bsm", fe.BsmParams(2.0), fe.CirClock(y0=1.0, kappa=0.2, eta=1.0, lam=1.5)
    )
    with pytest.raises(ValueError, match="explodes"):
        fe.time_changed_levy_cf(U, fe.ForwardSpec(S0=100.0, r=0.0, q=0.0, T=5.0), exploding)
