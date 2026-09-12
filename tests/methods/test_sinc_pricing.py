"""SINC pricer (Baschetti et al. 2022) and its identity with the Hilbert sum."""

from __future__ import annotations

import numpy as np
import pytest

import foureng as fe
from foureng.models.registry import MODEL_REGISTRY

FWD = fe.ForwardSpec(S0=100.0, r=0.03, q=0.01, T=0.5)
MODELS = {
    "bsm": fe.BsmParams(0.2),
    "heston": fe.HestonParams(kappa=2.0, theta=0.04, nu=0.6, rho=-0.7, v0=0.04),
    "vg": fe.VGParams(sigma=0.12, nu=0.2, theta=-0.14),
    "kou": fe.KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=10.0, eta2=5.0),
    "nig": fe.NigParams(sigma=0.2, nu=0.3, theta=-0.1),
}
K = np.linspace(70.0, 140.0, 15)


def _phi(model):
    return lambda u: MODEL_REGISTRY[model].cf(u, FWD, MODELS[model])


@pytest.mark.parametrize("model", list(MODELS))
def test_matches_the_contour_reference(model):
    for cp in (1, -1):
        got = fe.price_strip(model, "sinc", K, FWD, MODELS[model], cp=cp)
        ref = fe.price_strip(model, "contour", K, FWD, MODELS[model], cp=cp)
        np.testing.assert_allclose(got, ref, atol=1e-12)


def test_is_the_papers_square_wave_formula_and_the_hilbert_sum():
    """Q(X > k) = 1/2 + (2/pi) sum_{n odd} Im[e^{-i n pi k/X_c} phi(n pi/X_c)] / n,
    which is the half-integer Hilbert sum with h = 2 pi / X_c."""
    phi, X_c, n_terms = _phi("heston"), 3.0, 2048
    k = np.log(K / FWD.F0)
    n = 2 * np.arange(n_terms) + 1
    w = n * np.pi / X_c

    def q(shift):
        terms = np.imag(np.exp(-1j * np.outer(k, w)) * phi(w + shift)) / n
        return 0.5 + (2 / np.pi) * np.sum(terms, axis=1)

    paper = FWD.disc * (FWD.F0 * q(-1j) - K * q(0.0))
    grid = fe.SincGrid(X_c=X_c, N=n_terms)
    cums = MODEL_REGISTRY["heston"].cumulants(FWD, MODELS["heston"])
    sinc = fe.sinc_price_at_strikes(phi, FWD, K, cums, grid=grid)
    hil = fe.hilbert_price_at_strikes(
        phi, FWD, K, grid=fe.HilbertGrid(h=2 * np.pi / X_c, N=n_terms)
    )
    np.testing.assert_allclose(sinc, paper, atol=1e-13)
    np.testing.assert_array_equal(sinc, hil)


@pytest.mark.parametrize("model", ["heston", "nig", "kou"])
def test_fft_smile_matches_pointwise_prices(model):
    cums = MODEL_REGISTRY[model].cumulants(FWD, MODELS[model])
    strikes, calls = fe.sinc_smile(_phi(model), FWD, cums)
    assert strikes.size >= 256 and np.all(np.diff(np.log(strikes)) > 0)
    inside = (strikes > 60.0) & (strikes < 160.0)
    ref = fe.price_strip(model, "contour", strikes[inside], FWD, MODELS[model])
    np.testing.assert_allclose(calls[inside], ref, atol=1e-12)
    _, puts = fe.sinc_smile(_phi(model), FWD, cums, cp=-1)
    np.testing.assert_allclose(calls - puts, FWD.disc * (FWD.F0 - strikes), atol=1e-12)


def test_window_is_sized_from_the_density():
    """Few terms suffice; the fixed fine Hilbert default is far larger."""
    from foureng.pricers.sinc import _n_terms

    c1, c2, c4 = MODEL_REGISTRY["kou"].cumulants(FWD, MODELS["kou"])
    X_c = 2 * 10 * np.sqrt(c2 + np.sqrt(abs(c4)))
    assert _n_terms(_phi("kou"), 2 * np.pi / X_c, None) < fe.HilbertGrid().N // 8
    with pytest.raises(ValueError, match="strikes"):
        fe.sinc_price_at_strikes(_phi("kou"), FWD, [-1.0], (c1, c2, c4))
