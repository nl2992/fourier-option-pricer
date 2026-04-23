"""Black-Scholes-Merton characteristic function — PyFENG-backed.

BSM is the simplest Fourier sanity check: the log-forward CF is Gaussian

    phi_{X_T}(u) = exp(-0.5 * sigma^2 * T * (u^2 + i*u))

with ``X_T = log(S_T / F_0) = -0.5*sigma^2*T + sigma*sqrt(T)*Z`` under
the martingale convention. PyFENG's :class:`pyfeng.BsmFft` exposes this
through :meth:`charfunc_logprice` with the same convention we use
throughout (verified to ~1e-14 in
:mod:`tests/test_pyfeng_cf_wrappers.py`).

Why bother at all given we have a closed form? Two reasons:
  * Cross-model adapter symmetry — BSM becomes one of the ``_MODELS``
    dispatcher entries, so the pipeline + COS + FRFT + CM + PyFENG-FFT
    paths can all be driven by the same API for the BSM baseline.
  * A pricing sanity check that is known exact lets us bound numerical
    error floors for the other models on the same grid.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from ._pyfeng_backend import build_cached, import_pyfeng


@dataclass(frozen=True)
class BsmParams(ModelSpec):
    """Black-Scholes-Merton parameters — one scalar volatility.

    ``sigma`` is the **lognormal diffusion volatility** (not variance),
    i.e. ``d log S = (r - q - 0.5*sigma^2) dt + sigma dW``.
    """

    sigma: float

    def __init__(self, sigma: float):
        object.__setattr__(self, "name", "bsm")
        object.__setattr__(self, "sigma", sigma)


# ---------------------------------------------------------------------------
# PyFENG-backed CF
# ---------------------------------------------------------------------------

_BSM_MODEL_CACHE: dict[tuple, object] = {}


def _pyfeng_bsm_model(fwd: ForwardSpec, p: BsmParams):
    """Build-and-cache a :class:`pyfeng.BsmFft` for ``(fwd, p)``."""
    def _factory():
        pf = import_pyfeng()
        return pf.BsmFft(sigma=p.sigma, intr=fwd.r, divr=fwd.q)
    return build_cached(_BSM_MODEL_CACHE, (p, fwd), _factory)


def bsm_cf(u: np.ndarray, fwd: ForwardSpec, p: BsmParams) -> np.ndarray:
    """CF of ``X_T = log(S_T / F_0)`` under BSM — via PyFENG's ``BsmFft``."""
    m = _pyfeng_bsm_model(fwd, p)
    u_arr = np.asarray(u)
    return np.asarray(m.charfunc_logprice(u_arr, texp=fwd.T),
                      dtype=np.complex128)


# ---------------------------------------------------------------------------
# Cumulants — closed form (BSM is Gaussian in log-forward)
# ---------------------------------------------------------------------------

def bsm_cumulants(fwd: ForwardSpec, p: BsmParams) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of ``X_T = log(S_T / F_0)`` under BSM.

    With ``X_T = -0.5*sigma^2*T + sigma*sqrt(T)*Z``:

        c1 = -0.5*sigma^2*T
        c2 = sigma^2*T
        c3 = 0  (Gaussian)
        c4 = 0  (Gaussian)

    Returned as the ``(c1, c2, c4)`` tuple the COS auto-grid expects.
    """
    sigma2T = p.sigma * p.sigma * fwd.T
    return (-0.5 * sigma2T, sigma2T, 0.0)
