"""Generic affine jump-diffusions (Duffie, Pan & Singleton 2000)."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.9)
U = np.concatenate([np.linspace(0.0, 5.0, 11), np.linspace(5.0, 80.0, 16)])
K_, TH, NU, RHO, V0 = 1.5, 0.04, 0.5, -0.7, 0.05


def _heston(drift=0.02, v_drift=-0.5, **jumps):
    H1 = np.zeros((2, 2, 2))
    H1[1] = [[1.0, RHO * NU], [RHO * NU, NU * NU]]
    return fe.AffineParams(
        x0=[np.log(100.0), V0],
        K0=[drift, K_ * TH],
        K1=[[0.0, v_drift], [0.0, -K_]],
        H0=np.zeros((2, 2)),
        H1=H1,
        **jumps,
    )


def _lognormal_jumps(mu, sig):
    return lambda c: np.exp(c[..., 0] * mu + 0.5 * sig * sig * c[..., 0] ** 2)


def test_heston_bates_and_heston_kou_are_special_cases():
    hp = fe.HestonParams(kappa=K_, theta=TH, nu=NU, rho=RHO, v0=V0)
    np.testing.assert_allclose(
        fe.affine_cf(U, FWD, _heston()), fe.heston_cf_form2(U, FWD, hp), atol=1e-11
    )
    bates = _heston(l0=0.5, jump_transform=_lognormal_jumps(-0.1, 0.15))
    bp = fe.BatesParams(
        kappa=K_, theta=TH, nu=NU, rho=RHO, v0=V0, lam_j=0.5, mu_j=-0.1, sigma_j=0.15
    )
    np.testing.assert_allclose(fe.affine_cf(U, FWD, bates), fe.bates_cf(U, FWD, bp), atol=1e-11)
    p, e1, e2 = 0.4, 10.0, 5.0
    kou = _heston(
        l0=1.0, jump_transform=lambda c: p * e1 / (e1 - c[..., 0]) + (1 - p) * e2 / (e2 + c[..., 0])
    )
    kp = fe.HestonKouParams(
        kappa=K_, theta=TH, nu=NU, rho=RHO, v0=V0, lam_j=1.0, p_j=p, eta1=e1, eta2=e2
    )
    np.testing.assert_allclose(fe.affine_cf(U, FWD, kou), fe.heston_kou_cf(U, FWD, kp), atol=1e-11)


def test_three_factor_double_heston_and_one_factor_merton():
    a1, b1, s1, r1, v1 = 1.2, 0.03, 0.4, -0.6, 0.02
    a2, b2, s2, r2, v2 = 3.0, 0.02, 0.6, -0.3, 0.03
    H1 = np.zeros((3, 3, 3))
    H1[1] = [[1, r1 * s1, 0], [r1 * s1, s1 * s1, 0], [0, 0, 0]]
    H1[2] = [[1, 0, r2 * s2], [0, 0, 0], [r2 * s2, 0, s2 * s2]]
    dh = fe.AffineParams(
        x0=[0.0, v1, v2],
        K0=[0.0, a1 * b1, a2 * b2],
        K1=[[0, -0.5, -0.5], [0, -a1, 0], [0, 0, -a2]],
        H0=np.zeros((3, 3)),
        H1=H1,
    )
    ref = fe.double_heston_cf(U, FWD, fe.DoubleHestonParams(a1, b1, s1, r1, v1, a2, b2, s2, r2, v2))
    np.testing.assert_allclose(fe.affine_cf(U, FWD, dh), ref, atol=1e-11)
    merton = fe.AffineParams(
        x0=[0.0], K0=[0.0], K1=[[0.0]], H0=[[0.0225]], H1=np.zeros((1, 1, 1)),
        l0=0.5, jump_transform=_lognormal_jumps(-0.1, 0.15),
    )  # fmt: skip
    mp = fe.MertonJDParams(sigma=0.15, lam=0.5, muj=-0.1, sigj=0.15)
    np.testing.assert_allclose(
        fe.affine_cf(U, FWD, merton), fe.merton_jd_cf(U, FWD, mp), atol=1e-12
    )


def test_log_price_drift_is_normalised_away():
    np.testing.assert_allclose(
        fe.affine_cf(U, FWD, _heston(drift=0.0)),
        fe.affine_cf(U, FWD, _heston(drift=0.3)),
        atol=1e-12,
    )
    assert fe.affine_cf(np.array([-1j]), FWD, _heston())[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_variance_proportional_jump_intensity_matches_monte_carlo():
    """A specification with no closed form in the registry: jump intensity
    l1 * v (more jumps when volatility is high)."""
    mu_j, sig_j, lam1 = -0.08, 0.1, 20.0
    zeta = np.exp(mu_j + 0.5 * sig_j**2) - 1.0
    # The compensator -lam1 * zeta * v is state dependent, so it belongs in K1.
    spec = _heston(
        v_drift=-0.5 - lam1 * zeta, l1=[0.0, lam1], jump_transform=_lognormal_jumps(mu_j, sig_j)
    )
    K = np.array([85.0, 100.0, 115.0])
    got = fe.price_strip("affine", "sinc", K, FWD, spec)
    grid = fe.cos_auto_grid(fe.affine_cumulants(FWD, spec), N=2048, L=14.0)
    np.testing.assert_allclose(
        fe.price_strip("affine", "cos", K, FWD, spec, grid=grid), got, atol=1e-9
    )
    rng = np.random.default_rng(9)
    n, steps = 200_000, 400
    dt = FWD.T / steps
    x, v = np.zeros(n), np.full(n, V0)
    for _ in range(steps):
        vp = np.maximum(v, 0.0)
        z1 = rng.standard_normal(n)
        z2 = RHO * z1 + np.sqrt(1 - RHO**2) * rng.standard_normal(n)
        jumps = rng.poisson(lam1 * vp * dt)
        jump_size = mu_j * jumps + sig_j * np.sqrt(jumps) * rng.standard_normal(n)
        x += -(0.5 * vp + lam1 * vp * zeta) * dt + np.sqrt(vp * dt) * z2 + jump_size
        v += K_ * (TH - vp) * dt + NU * np.sqrt(vp * dt) * z1
    S = FWD.F0 * np.exp(x)
    pay = FWD.disc * np.maximum(S[None, :] - K[:, None], 0.0)
    cv = S - FWD.F0
    est = pay - np.array([np.cov(r, cv)[0, 1] / cv.var() for r in pay])[:, None] * cv
    se = est.std(axis=1) / np.sqrt(n)
    assert np.all(np.abs(got - est.mean(axis=1)) < 4.0 * se + 0.01)


def test_validation():
    with pytest.raises(ValueError, match="K1 must have shape"):
        fe.AffineParams(
            x0=[0.0, 0.1], K0=[0, 0], K1=[[0.0]], H0=np.zeros((2, 2)), H1=np.zeros((2, 2, 2))
        )
    with pytest.raises(ValueError, match="jump_transform"):
        fe.AffineParams(x0=[0.0], K0=[0.0], K1=[[0.0]], H0=[[0.04]], H1=np.zeros((1, 1, 1)), l0=1.0)
