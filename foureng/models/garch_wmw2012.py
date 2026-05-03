"""GARCH diffusion model (Wu, Ma & Wang 2012) — native CF implementation.

The GARCH diffusion model couples a log-normal spot process with a
mean-reverting variance process whose diffusion coefficient is proportional
to the variance itself (not its square root as in Heston):

    dS/S = sqrt(v_t) dW1
    dv   = kappa * (theta - v_t) dt + nu * v_t dW2,
           <dW1, dW2> = rho dt

This is the continuous-time analogue of the discrete GARCH(1,1) model
(Nelson 1990). The vol-of-vol term ``nu * v`` (versus ``nu * sqrt(v)`` in
Heston and ``nu * v^(3/2)`` in the 3/2 model) produces a qualitatively
different implied-vol surface shape.

MGF derivation
--------------
Wu, Ma & Wang (2012) derive a closed-form (approximate) MGF via the Lemma
2.1 formula. The CF of X_T = log(S_T/F_0) is obtained as phi(u) = MGF(iu).
PyFENG ships this as :class:`pyfeng.GarchFftWuMaWang2012`, but the
``charfunc_logprice`` / ``price`` methods are broken under NumPy ≥ 1.25
because :func:`pyfeng.util.avg_exp` pre-allocates a float64 output array
before dividing by a complex argument (``UFuncTypeError``). We implement
the MGF formula directly from Lemma 2.1, bypassing PyFENG entirely.

PyFENG compatibility note
-------------------------
The PyFENG docstring for ``GarchFftWuMaWang2012`` lists the same benchmark
prices as ``Sv32Fft`` with identical parameters — this appears to be a
copy-paste error in the upstream docstring. The GARCH diffusion and 3/2
SV models are mathematically distinct: with the Lewis (2000) benchmark
parameters (sigma=0.06, mr=20.48, theta=0.218, vov=3.20, rho=-0.99,
spot=100, T=0.5), the GARCH model gives prices ≈ [14.77, 12.25, 10.04]
whereas Sv32 gives ≈ [11.72, 8.98, 6.71]. This module uses the GARCH
formula's own output as the reference.

References
----------
* Wu, X.-Y., Ma, C.-Q. & Wang, S.-Y. (2012), "Warrant pricing under GARCH
  diffusion model", *Economic Modelling*, 29, 2237–2244.
* Nelson, D. B. (1990), "ARCH models as diffusion approximations",
  *Journal of Econometrics*, 45, 7–38.
* Lewis, A. L. (2000), *Option Valuation Under Stochastic Volatility*,
  Finance Press (same benchmark parameters as Sv32).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


@dataclass(frozen=True)
class GarchWMW2012Params(ModelSpec):
    """GARCH diffusion parameters (Wu, Ma & Wang 2012).

    Parameters
    ----------
    v0 :
        Initial instantaneous variance. Must be ``> 0``.  Note: this is
        *variance* (not volatility); the initial vol is ``sqrt(v0)``.
    kappa :
        Mean-reversion speed of the variance process. Must be ``> 0``.
    theta :
        Long-term mean of the variance process. Must be ``> 0``.
    nu :
        Vol-of-vol — coefficient on the ``v * dW2`` diffusion term.
        Must be ``> 0``.
    rho :
        Correlation between spot and variance Brownian motions.
        Must be in ``(-1, 1)``.
    """

    v0: float
    kappa: float
    theta: float
    nu: float
    rho: float

    def __init__(
        self,
        v0: float,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
    ):
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"GarchWMW2012Params: v0 must be > 0; got {v0}")
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"GarchWMW2012Params: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"GarchWMW2012Params: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"GarchWMW2012Params: nu must be > 0; got {nu}")
        if not (np.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(f"GarchWMW2012Params: rho must be in (-1, 1); got {rho}")
        object.__setattr__(self, "name", "garch_wmw2012")
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)


# ---------------------------------------------------------------------------
# Native MGF / CF — bypasses the broken PyFENG charfunc_logprice
# ---------------------------------------------------------------------------


def _avg_exp_safe(x: np.ndarray) -> np.ndarray:
    """(1 - exp(x)) / x, complex-safe. Returns 1.0 where x ≈ 0.

    This is ``pyfeng.util.avg_exp`` re-implemented to accept complex arrays
    without pre-allocating a float64 output buffer (the root cause of the
    upstream ``UFuncTypeError``).
    """
    x = np.asarray(x, dtype=np.complex128)
    near_zero = np.abs(x) < 1e-14
    result = np.where(near_zero, 1.0 + 0j, np.expm1(x) / np.where(near_zero, 1.0, x))
    return result.astype(np.complex128)


def _garch_mgf(uu: np.ndarray, p: "GarchWMW2012Params", texp: float) -> np.ndarray:
    """MGF of X_T = log(S_T/F_0) under GARCH diffusion.

    Implements Lemma 2.1 of Wu et al. (2012) in its refined form
    (as coded in ``pyfeng.GarchFftWuMaWang2012.mgf_logprice``), extended
    to complex arguments ``uu`` via :func:`_avg_exp_safe`.

    Parameters
    ----------
    uu : array_like (complex)
        MGF argument. CF is obtained by passing ``uu = 1j * u_real``.
    p : GarchWMW2012Params
    texp : float
        Time to expiry.

    Returns
    -------
    np.ndarray (complex128)
    """
    uu = np.asarray(uu, dtype=np.complex128)

    zeta = uu - uu**2
    uu_etc = p.rho * p.nu * np.sqrt(p.theta) * uu

    gg = p.kappa - (3 / 2) * uu_etc
    dd = np.sqrt(gg**2 + 2 * p.nu**2 * p.theta * zeta)
    dd_m_gg = 2 * p.nu**2 * p.theta * zeta / (dd + gg)  # stable: dd - gg

    avgexp = _avg_exp_safe(-dd * texp)  # (1 - exp(-dd*texp)) / (dd*texp)

    tmp = 1.0 - 0.5 * dd_m_gg * avgexp * texp
    log_tmp = np.log(tmp)

    # C term (Lemma 2.1)
    C = -(p.kappa - 0.5 * uu_etc) * dd_m_gg * texp - (p.kappa + 0.5 * uu_etc) * log_tmp
    term3_num = 2 * p.nu**2 * p.theta * zeta + dd_m_gg**2 * np.exp(-dd * texp) - 4 * dd**2 * avgexp
    C -= (term3_num * dd_m_gg * texp / (2 * dd) / tmp) / 4.0

    # D term
    D = 0.5 * avgexp / tmp * texp * zeta

    # v0 plays the role of sigma in PyFENG's convention (initial variance)
    return np.exp(C * (0.5 / p.nu**2) - D * p.v0)


def garch_wmw2012_cf(u: np.ndarray, fwd: ForwardSpec, p: GarchWMW2012Params) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under the GARCH diffusion model.

    Evaluates the Wu et al. (2012) MGF formula at imaginary argument:
    phi(u) = MGF(iu), where MGF(uu) = E[exp(uu * X_T)].

    Parameters
    ----------
    u : array_like (real or complex)
        Frequency grid.
    fwd : ForwardSpec
    p : GarchWMW2012Params

    Returns
    -------
    np.ndarray (complex128)
    """
    u_arr = np.asarray(u, dtype=np.complex128)
    return _garch_mgf(1j * u_arr, p, fwd.T)


# ---------------------------------------------------------------------------
# Cumulants — numerical Cauchy integral
# ---------------------------------------------------------------------------


def garch_wmw2012_cumulants(fwd: ForwardSpec, p: GarchWMW2012Params) -> tuple[float, float, float]:
    """Cumulants ``(c1, c2, c4)`` of X_T under the GARCH diffusion model.

    The GARCH MGF is analytic in a disk around u=0 (under typical parameters
    satisfying the non-explosion conditions), so the numerical Cauchy-circle
    FFT gives machine-precision cumulants.
    """
    from ..utils.cumulants import cumulants_from_cf

    def _phi(u):
        return garch_wmw2012_cf(u, fwd, p)

    c = cumulants_from_cf(_phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
