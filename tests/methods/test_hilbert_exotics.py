"""Discretely monitored barriers and lookbacks by the fast Hilbert transform."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve
from scipy.special import ndtr

import foureng as fe
from foureng.products.barrier import BarrierOption
from foureng.products.lookback import LookbackOption

FWD = fe.ForwardSpec(S0=100.0, r=0.05, q=0.02, T=1.0)
KOU = fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0)
NIG = fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1)


# --------------------------------------------------------------------------- #
# Independent BSM references: backward induction with the exact Gaussian kernel
# on a fine log-price grid, Richardson-extrapolated in the grid spacing.
# --------------------------------------------------------------------------- #
def _bsm_barrier_grid(K, H, bt, cp, sigma, M, dx):
    S0, r, q, T = FWD.S0, FWD.r, FWD.q, FWD.T
    dt = T / M
    mu, s = (r - q - 0.5 * sigma * sigma) * dt, sigma * np.sqrt(dt)
    lvl, k = np.log(H / S0), np.log(K / S0)
    x = np.arange(lvl, lvl + 3.0, dx) if bt == "down_out" else np.arange(lvl, lvl - 3.0, -dx)[::-1]
    V = np.maximum(cp * (np.exp(x) - np.exp(k)), 0.0)
    n = int(np.ceil(9 * s / dx))
    z = np.arange(-n, n + 1) * dx
    kern = np.exp(-0.5 * ((z - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi)) * dx
    w = np.ones_like(x)
    w[0] = w[-1] = 0.5
    for _ in range(M - 1):
        V = np.exp(-r * dt) * fftconvolve(V * w, kern[::-1], mode="same")
    p0 = np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return S0 * np.exp(-r * dt) * np.sum(V * p0 * w) * dx


def _bsm_lookback_grid(cp, sigma, M, dz):
    S0, r, q, T = FWD.S0, FWD.r, FWD.q, FWD.T
    dt = T / M
    s, m = sigma * np.sqrt(dt), (r - q + 0.5 * sigma * sigma) * dt  # share measure
    sgn, t = (-1.0, 1.0) if cp == -1 else (1.0, -1.0)
    z = np.arange(0.0, 12 * sigma * np.sqrt(T) + 1.0, dz)
    u = np.exp(t * z)
    n = int(np.ceil(9 * s / dz))
    d = np.arange(-n, n + 1) * dz
    pd = np.exp(-0.5 * ((d - sgn * m) / s) ** 2) / (s * np.sqrt(2 * np.pi)) * dz
    for _ in range(M):
        w = np.ones_like(z)
        w[0] = 0.5
        cont = fftconvolve(u * w, pd[::-1], mode="full")[len(pd) // 2 : len(pd) // 2 + len(z)]
        u = u[0] * ndtr((-z - sgn * m) / s) + cont
    sv = S0 * np.exp(-q * T)
    return sv * (u[0] - 1.0) if cp == -1 else sv * (1.0 - u[0])


@pytest.mark.derived_reference
@pytest.mark.parametrize(
    "K,H,bt,cp",
    [
        (100, 85, "down_out", 1),
        (100, 85, "down_out", -1),
        (100, 120, "up_out", 1),
        (100, 120, "up_out", -1),
    ],
)
def test_bsm_barriers_match_exact_kernel_grid(K, H, bt, cp):
    M, sigma = 12, 0.2
    a = _bsm_barrier_grid(K, H, bt, cp, sigma, M, 2e-4)
    b = _bsm_barrier_grid(K, H, bt, cp, sigma, M, 1e-4)
    ref = (4 * b - a) / 3
    got = fe.hilbert_barrier_price(
        "bsm",
        FWD,
        fe.BsmParams(sigma),
        strike=K,
        barrier=H,
        maturity=1.0,
        barrier_type=bt,
        cp=cp,
        n_monitor=M,
    )
    assert got == pytest.approx(ref, abs=5e-7)


@pytest.mark.derived_reference
@pytest.mark.parametrize("M", [12, 52])
@pytest.mark.parametrize("cp", [-1, 1])
def test_bsm_lookbacks_match_exact_kernel_grid(M, cp):
    a = _bsm_lookback_grid(cp, 0.25, M, 4e-4)
    b = _bsm_lookback_grid(cp, 0.25, M, 2e-4)
    ref = (4 * b - a) / 3
    got = fe.hilbert_lookback_price(
        "bsm", FWD, fe.BsmParams(0.25), maturity=1.0, cp=cp, n_monitor=M
    )
    assert got == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("model,params", [("bsm", fe.BsmParams(0.2)), ("kou", KOU), ("nig", NIG)])
def test_far_barriers_reduce_to_the_european(model, params):
    for M in (1, 3):
        call = fe.hilbert_barrier_price(
            model,
            FWD,
            params,
            strike=100.0,
            barrier=1e-3,
            maturity=1.0,
            barrier_type="down_out",
            cp=1,
            n_monitor=M,
        )
        put = fe.hilbert_barrier_price(
            model,
            FWD,
            params,
            strike=100.0,
            barrier=1e5,
            maturity=1.0,
            barrier_type="up_out",
            cp=-1,
            n_monitor=M,
        )
        eu = fe.price_strip(model, "contour", [100.0], FWD, params)[0]
        eu_put = fe.price_strip(model, "contour", [100.0], FWD, params, cp=-1)[0]
        assert call == pytest.approx(eu, abs=1e-10)
        assert put == pytest.approx(eu_put, abs=1e-10)


def test_converged_in_sinc_step_and_grid_size():
    vals = [
        fe.hilbert_barrier_price(
            "kou",
            FWD,
            KOU,
            strike=100.0,
            barrier=120.0,
            maturity=1.0,
            barrier_type="up_out",
            cp=-1,
            n_monitor=52,
            h=h,
            N=N,
        )
        for h, N in ((np.pi / 16, 1 << 12), (np.pi / 32, 1 << 14), (np.pi / 64, 1 << 16))
    ]
    assert np.ptp(vals) < 1e-9


def _kou_paths(n, M, seed):
    """Exact Kou log-price increments on M equal steps (Gaussian + DE compound Poisson)."""
    rng = np.random.default_rng(seed)
    k = KOU
    zeta = k.p * k.eta1 / (k.eta1 - 1) + (1 - k.p) * k.eta2 / (k.eta2 + 1) - 1
    dt = FWD.T / M
    mu = (FWD.r - FWD.q - 0.5 * k.sigma**2 - k.lam * zeta) * dt
    counts = rng.poisson(k.lam * dt, (n, M))
    jumps = np.zeros((n, M))
    for i in range(int(counts.max())):
        up = rng.random((n, M)) < k.p
        size = np.where(
            up, rng.exponential(1 / k.eta1, (n, M)), -rng.exponential(1 / k.eta2, (n, M))
        )
        jumps += np.where(counts > i, size, 0.0)
    incr = mu + k.sigma * np.sqrt(dt) * rng.standard_normal((n, M)) + jumps
    return FWD.S0 * np.exp(np.cumsum(incr, axis=1))


def test_kou_barrier_and_lookback_match_exact_simulation():
    M = 12
    S = _kou_paths(200_000, M, seed=11)
    disc = np.exp(-FWD.r * FWD.T)
    alive = np.all(S < 120.0, axis=1)
    pay_b = disc * np.maximum(100.0 - S[:, -1], 0.0) * alive
    pay_l = disc * (np.maximum(S.max(axis=1), FWD.S0) - S[:, -1])
    for pay, got in (
        (
            pay_b,
            fe.hilbert_barrier_price(
                "kou",
                FWD,
                KOU,
                strike=100.0,
                barrier=120.0,
                maturity=1.0,
                barrier_type="up_out",
                cp=-1,
                n_monitor=M,
            ),
        ),
        (pay_l, fe.hilbert_lookback_price("kou", FWD, KOU, maturity=1.0, cp=-1, n_monitor=M)),
    ):
        se = pay.std() / np.sqrt(pay.size)
        assert abs(got - pay.mean()) < 4.0 * se


def test_structure_parity_and_monitoring_monotonicity():
    kw = dict(strike=100.0, barrier=85.0, maturity=1.0, cp=1)
    out = fe.hilbert_barrier_price("nig", FWD, NIG, barrier_type="down_out", n_monitor=52, **kw)
    inn = fe.hilbert_barrier_price("nig", FWD, NIG, barrier_type="down_in", n_monitor=52, **kw)
    eu = fe.price_strip("nig", "contour", [100.0], FWD, NIG)[0]
    assert out + inn == pytest.approx(eu, abs=1e-10)
    # More monitoring dates: more chances to knock out, and a larger running max.
    ko = [
        fe.hilbert_barrier_price("nig", FWD, NIG, barrier_type="down_out", n_monitor=m, **kw)
        for m in (4, 12, 52)
    ]
    lb = [
        fe.hilbert_lookback_price("nig", FWD, NIG, maturity=1.0, cp=-1, n_monitor=m)
        for m in (4, 12, 52)
    ]
    assert ko[0] > ko[1] > ko[2] and lb[0] < lb[1] < lb[2]
    # Discrete monitoring stays below the continuous BSM lookback.
    cont = fe.price(
        LookbackOption(maturity=1.0, cp=-1), "bsm", "lookback_bsm", FWD, fe.BsmParams(0.25)
    )
    disc = fe.hilbert_lookback_price(
        "bsm", FWD, fe.BsmParams(0.25), maturity=1.0, cp=-1, n_monitor=252
    )
    assert disc < cont < disc * 1.08


def test_pipeline_dispatch_and_guards():
    prod = BarrierOption(
        strike=100.0,
        barrier=85.0,
        maturity=1.0,
        cp=1,
        barrier_type="down_out",
        monitoring="discrete",
    )
    assert fe.price(prod, "kou", "hilbert_barrier", FWD, KOU, grid=12) == pytest.approx(
        fe.hilbert_barrier_price(
            "kou",
            FWD,
            KOU,
            strike=100.0,
            barrier=85.0,
            maturity=1.0,
            barrier_type="down_out",
            n_monitor=12,
        )
    )
    look = LookbackOption(maturity=1.0, cp=1, monitoring="discrete")
    assert fe.price(look, "kou", "hilbert_lookback", FWD, KOU, grid=12) == pytest.approx(
        fe.hilbert_lookback_price("kou", FWD, KOU, maturity=1.0, cp=1, n_monitor=12)
    )
    heston = fe.HestonParams(kappa=2.0, theta=0.04, nu=0.4, rho=-0.6, v0=0.04)
    with pytest.raises(NotImplementedError, match="1-D Levy"):
        fe.price(prod, "heston", "hilbert_barrier", FWD, heston)
    fixed = LookbackOption(maturity=1.0, cp=1, strike_type="fixed", strike=110.0)
    assert fe.price(fixed, "kou", "hilbert_lookback", FWD, KOU, grid=12) == pytest.approx(
        fe.hilbert_lookback_price(
            "kou", FWD, KOU, maturity=1.0, cp=1, strike_type="fixed", strike=110.0, n_monitor=12
        )
    )
    with pytest.raises(ValueError, match="far side"):
        fe.hilbert_barrier_price(
            "bsm",
            FWD,
            fe.BsmParams(0.2),
            strike=90.0,
            barrier=110.0,
            maturity=1.0,
            barrier_type="down_out",
        )


# --------------------------------------------------------------------------- #
# Fixed-strike lookbacks (Spitzer duality + Parseval)
# --------------------------------------------------------------------------- #
def _fixed(model, params, cp, K, M=12, **kw):
    return fe.hilbert_lookback_price(
        model, FWD, params, maturity=1.0, cp=cp, strike_type="fixed", strike=K, n_monitor=M, **kw
    )


@pytest.mark.parametrize("model,params", [("bsm", fe.BsmParams(0.25)), ("kou", KOU)])
def test_fixed_strike_linear_region_matches_floating_engine(model, params):
    """Two different recursions/measures: (M - K)^+ = M - K for K <= S0 gives
    fixed call = floating put + S0 e^{-qT} - K e^{-rT}; the put mirrors it."""
    disc, share = np.exp(-FWD.r), FWD.S0 * np.exp(-FWD.q)
    flt_put = fe.hilbert_lookback_price(model, FWD, params, maturity=1.0, cp=-1, n_monitor=12)
    flt_call = fe.hilbert_lookback_price(model, FWD, params, maturity=1.0, cp=1, n_monitor=12)
    assert _fixed(model, params, 1, 90.0) == pytest.approx(flt_put + share - 90.0 * disc, abs=1e-11)
    assert _fixed(model, params, -1, 110.0) == pytest.approx(
        flt_call + 110.0 * disc - share, abs=1e-11
    )


def test_fixed_strike_is_continuous_across_the_spot_and_converged():
    """The running extremum includes S0, so P(max = S0) > 0 and the price has a
    genuine kink at K = S0: slope -e^{-rT} below it (call), flatter above."""
    disc, dk = np.exp(-FWD.r), 1e-3
    c = [_fixed("bsm", fe.BsmParams(0.25), 1, k) for k in (100.0 - dk, 100.0, 100.0 + dk)]
    assert (c[1] - c[0]) / dk == pytest.approx(-disc, abs=1e-6)
    assert -disc < (c[2] - c[1]) / dk < -0.5 * disc  # continuous, less steep
    p = [_fixed("bsm", fe.BsmParams(0.25), -1, k) for k in (100.0 - dk, 100.0, 100.0 + dk)]
    assert (p[2] - p[1]) / dk == pytest.approx(disc, abs=1e-6)
    assert 0.5 * disc < (p[1] - p[0]) / dk < disc
    fine = _fixed("bsm", fe.BsmParams(0.25), 1, 115.0, h=np.pi / 64, N=1 << 18)
    assert _fixed("bsm", fe.BsmParams(0.25), 1, 115.0) == pytest.approx(fine, abs=3e-7)


@pytest.mark.parametrize("model,params", [("bsm", fe.BsmParams(0.25)), ("kou", KOU)])
def test_fixed_strike_matches_simulation_with_extremum_control(model, params):
    M = 12
    if model == "bsm":
        rng = np.random.default_rng(1)
        dt = 1.0 / M
        sig = params.sigma
        incr = (FWD.r - FWD.q - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * rng.standard_normal(
            (300_000, M)
        )
        S = FWD.S0 * np.exp(np.cumsum(incr, axis=1))
    else:
        S = _kou_paths(300_000, M, seed=12)
    disc = np.exp(-FWD.r)
    mx = np.maximum(S.max(axis=1), FWD.S0)
    mn = np.minimum(S.min(axis=1), FWD.S0)
    e_max = _fixed(model, params, 1, 1e-9) / disc  # linear branch: E[max]
    e_min = FWD.S0 - _fixed(model, params, -1, FWD.S0) / disc
    for cp, K, ext, mean in ((1, 115.0, mx, e_max), (-1, 88.0, mn, e_min)):
        pay = disc * np.maximum(cp * (ext - K), 0.0)
        cv = ext - mean
        est = pay - np.cov(pay, cv)[0, 1] / cv.var() * cv
        se = est.std() / np.sqrt(est.size)
        assert abs(_fixed(model, params, cp, K) - est.mean()) < 4.0 * se
