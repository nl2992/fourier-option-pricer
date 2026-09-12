"""Discretely monitored arithmetic Asian options under Levy models (ASCOS-type recursion).

With log-returns ``Y_j = log(S_{t_j} / S_{t_{j-1}})`` (independent), the sum of
the monitored prices nests as

    sum_{j=1}^N S_{t_j} / S_0 = e^{Y_1} (1 + e^{Y_2} (1 + ... (1 + e^{Y_N}))) = e^{B_1},

    B_N = Y_N,    B_j = Y_j + log(1 + e^{B_{j+1}})    (Carverhill & Clewlow 1990).

So the CF of ``B_j`` is ``phi_{Y_j}(u) * phi_{Z_{j+1}}(u)`` with
``Z = log(1 + e^B)``. Each backward step recovers the density of ``B_{j+1}``
from its CF by a Fourier-cosine series on a cumulant-based interval (Fang &
Oosterlee 2008), and computes ``phi_Z(u) = int (1 + e^x)^{iu} f_B(x) dx`` by
Gauss-Legendre quadrature, exactly as in the ASCOS method of Zhang & Oosterlee
(2013). The interval of every ``B_j`` comes from its first four cumulants
(``Y`` cumulants from the model plus ``Z`` cumulants from the quadrature). The
option on ``A = (S_0 / N) e^{B_1}`` is then priced by COS; puts follow from
parity with ``E[A] = (S_0 / N) sum_j e^{(r - q) t_j}``.

References
----------
* Carverhill, A. & Clewlow, L. (1990), Flexible convolution, *Risk* 3(4).
* Zhang, B. & Oosterlee, C.W. (2013), Efficient pricing of European-style
  Asian options under exponential Levy processes based on Fourier cosine
  expansions, *SIAM J. Financial Math.* 4, 399-426.
* Fusai, G. & Meucci, A. (2008), Pricing discretely monitored Asian options
  under Levy processes, *J. Banking & Finance* 32, 2076-2088.
"""

from __future__ import annotations

import numpy as np

from ..models.base import ForwardSpec
from ..models.registry import MODEL_REGISTRY

__all__ = ["levy_arithmetic_asian_price"]


def _cumulants_from_raw(m1, m2, m3, m4) -> tuple[float, float, float]:
    c2 = m2 - m1 * m1
    c4 = m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1**4
    return m1, c2, c4


class _ZLaw:
    """Law of ``Z = log(1 + e^B)`` held as quadrature atoms ``(z_q, c_q)``."""

    def __init__(self, z: np.ndarray, c: np.ndarray):
        self.z, self.c = z, c

    def cf(self, u: np.ndarray) -> np.ndarray:
        return np.exp(1j * np.outer(u, self.z)) @ self.c

    def cumulants(self) -> tuple[float, float, float]:
        m = [float(np.sum(self.c * self.z**n)) for n in (1, 2, 3, 4)]
        return _cumulants_from_raw(*m)


def levy_arithmetic_asian_price(
    model: str,
    fwd: ForwardSpec,
    params,
    *,
    strike: float,
    monitoring_times,
    maturity: float | None = None,
    cp: int = 1,
    n_cos: int = 256,
    n_quad: int = 1024,
    L: float = 10.0,
) -> float:
    """Fixed-strike arithmetic Asian on ``(1/N) sum_j S_{t_j}``, 1-D Levy models.

    Parameters
    ----------
    model, fwd, params :
        Registry key of a 1-D Levy model, market inputs (``S0, r, q``), params.
    strike : float
        Fixed strike.
    monitoring_times : array_like
        Strictly increasing dates in ``(0, maturity]`` (need not be uniform).
    maturity : float, optional
        Payment date; defaults to the last monitoring date.
    cp : int
        ``+1`` call, ``-1`` put.
    n_cos, n_quad, L :
        Cosine terms for each density recovery, Gauss-Legendre nodes for each
        ``phi_Z`` integral, and the truncation multiplier of the intervals.

    Returns
    -------
    float
        Discounted price.
    """
    from ..pricers.cos_bermudan import _SUPPORTED_MODELS

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"arithmetic Asian: needs a 1-D Levy model {sorted(_SUPPORTED_MODELS)}; got {model!r}"
        )
    if cp not in (1, -1):
        raise ValueError(f"cp must be +1 or -1; got {cp}")
    t = np.asarray(monitoring_times, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("monitoring_times must be positive and strictly increasing")
    T = float(t[-1] if maturity is None else maturity)
    if T < t[-1] - 1e-12:
        raise ValueError("maturity must not precede the last monitoring date")
    if strike <= 0.0:
        raise ValueError("strike must be > 0")

    entry = MODEL_REGISTRY[model]
    dts = np.diff(np.concatenate([[0.0], t]))
    n = t.size

    def y_cf(j: int):
        f = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=float(dts[j]))
        drift = (fwd.r - fwd.q) * dts[j]
        return lambda u: np.exp(1j * u * drift) * np.asarray(entry.cf(u, f, params), dtype=complex)

    def y_cumulants(j: int) -> tuple[float, float, float]:
        f = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=float(dts[j]))
        c1, c2, c4 = entry.cumulants(f, params)
        return c1 + (fwd.r - fwd.q) * dts[j], c2, c4

    def interval(c1, c2, c4):
        half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        return c1 - half, c1 + half

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    k = np.arange(n_cos)

    # Start from B_N = Y_N and walk back to B_1.
    b_cf = y_cf(n - 1)
    b_cum = y_cumulants(n - 1)
    for j in range(n - 2, -1, -1):
        a, b = interval(*b_cum)
        w = k * np.pi / (b - a)
        coef = (2.0 / (b - a)) * np.real(b_cf(w) * np.exp(-1j * w * a))
        coef[0] *= 0.5
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        dens = np.cos(np.outer(x - a, w)) @ coef
        law = _ZLaw(np.logaddexp(0.0, x), 0.5 * (b - a) * weights * dens)
        yf = y_cf(j)
        b_cf = lambda u, yf=yf, law=law: yf(u) * law.cf(u)  # noqa: E731
        zc = law.cumulants()
        yc = y_cumulants(j)
        b_cum = (yc[0] + zc[0], yc[1] + zc[1], yc[2] + zc[2])

    # Price the call on A = (S0/N) e^{B_1} by COS.
    a, b = interval(*b_cum)
    w = k * np.pi / (b - a)
    scale = fwd.S0 / n
    x_k = float(np.log(strike / scale))
    lo = max(x_k, a)
    phi = b_cf(w) * np.exp(-1j * w * a)
    call_payoff = np.zeros(n_cos)
    if lo < b:
        ww = w
        wc, wd = ww * (lo - a), ww * (b - a)
        chi = (
            np.cos(wd) * np.exp(b)
            - np.cos(wc) * np.exp(lo)
            + ww * (np.sin(wd) * np.exp(b) - np.sin(wc) * np.exp(lo))
        ) / (1.0 + ww * ww)
        psi = np.empty(n_cos)
        psi[0] = b - lo
        psi[1:] = (np.sin(wd[1:]) - np.sin(wc[1:])) / ww[1:]
        call_payoff = (2.0 / (b - a)) * (scale * chi - strike * psi)
    terms = np.real(phi) * call_payoff
    terms[0] *= 0.5
    call = float(np.exp(-fwd.r * T) * np.sum(terms))
    if cp == 1:
        return max(call, 0.0)
    mean_a = scale * float(np.sum(np.exp((fwd.r - fwd.q) * t)))
    return max(call - np.exp(-fwd.r * T) * (mean_a - strike), 0.0)
