from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FFTGrid:
    """Carr-Madan FFT grid. Nyquist binds eta and lam: eta*lam = 2*pi/N.

    Parameters
    ----------
    N : int
        Number of FFT points (power of 2 recommended, e.g. 4096).
    eta : float
        Frequency-domain step size. Smaller eta → finer frequency resolution
        but coarser log-strike spacing (lam = 2π / (N·η)).
    alpha : float
        Carr-Madan dampening exponent. Requires E[S_T^{alpha+1}] < ∞.
        The standard choice **alpha = 1.5** (from CM1999) is valid for
        BSM, Heston, VG, and most common models at typical maturities.
        For Kou, the hard upper bound is alpha < eta1 - 1; for VG it
        depends on the parametrisation. Use
        :func:`~foureng.utils.validity.check_alpha` to verify.
    """

    N: int
    eta: float
    alpha: float

    def u(self) -> np.ndarray:
        return np.arange(self.N) * self.eta

    @property
    def lam(self) -> float:
        return 2.0 * np.pi / (self.N * self.eta)

    @property
    def b(self) -> float:
        """Half-width of the log-strike grid, centered at k0."""
        return self.N * self.lam / 2.0

    def k_grid(self, k0: float = 0.0) -> np.ndarray:
        """Log-strike grid centered at k0. Spacing = lam."""
        return (k0 - self.b) + np.arange(self.N) * self.lam


@dataclass(frozen=True)
class FRFTGrid:
    """Fractional FFT grid  -  eta (freq step) and lam (log-strike step) independent.

    Unlike :class:`FFTGrid`, the Nyquist constraint does not bind here:
    eta and lam are chosen freely, and the FRFT fraction
    ``zeta = eta·lam / (2π)`` need not equal ``1/N``. This decoupling
    lets you tune frequency resolution and strike spacing independently.

    Parameters
    ----------
    N : int
        Number of FRFT points (power of 2 recommended).
    eta : float
        Frequency-domain step size (same role as in :class:`FFTGrid`).
    lam : float
        Log-strike grid spacing. Choose independently of eta.
    alpha : float
        Dampening exponent  -  same meaning and default guidance as in
        :class:`FFTGrid`. The standard choice **alpha = 1.5** works for
        most models; see :func:`~foureng.utils.validity.check_alpha`.
    """

    N: int
    eta: float
    lam: float
    alpha: float

    def u(self) -> np.ndarray:
        return np.arange(self.N) * self.eta

    @property
    def zeta(self) -> float:
        """FRFT fraction: zeta = eta * lam / (2*pi)."""
        return self.eta * self.lam / (2.0 * np.pi)

    def k_grid(self, k0: float = 0.0) -> np.ndarray:
        return (k0 - self.N * self.lam / 2.0) + np.arange(self.N) * self.lam


@dataclass(frozen=True)
class CONVGrid:
    """Fourier-inversion grid for the CONV-style European pricer.

    The first implementation uses the probability-transform form behind
    convolution/Fourier European pricing: two one-dimensional inversions for
    the share and cash probabilities.  ``N`` controls the number of positive
    frequency nodes and ``u_max`` the integration cutoff.
    """

    N: int = 4096
    u_max: float = 200.0

    def u(self) -> np.ndarray:
        if self.N < 2:
            raise ValueError("CONVGrid: N must be at least 2")
        if self.u_max <= 0.0:
            raise ValueError("CONVGrid: u_max must be > 0")
        return np.linspace(1e-10, self.u_max, self.N)


@dataclass(frozen=True)
class HilbertGrid:
    """Frequency grid for the Hilbert-transform pricer of Feng & Linetsky (2008).

    The pricer evaluates Gil-Pelaez-style probability integrals on the
    half-integer sinc grid ``u_m = (m + 1/2) * h`` for ``m = 0..N-1``. The
    half-integer offset skips the ``u = 0`` singularity of the integrand and
    turns the midpoint sum into the discrete Hilbert transform, whose error
    decays like ``exp(-c/h)`` for characteristic functions analytic in a
    horizontal strip (Feng & Linetsky 2008; Stenger 1993).

    Parameters
    ----------
    h : float
        Frequency step. Smaller ``h`` reduces the discretization error
        (exponentially, for strip-analytic CFs).
    N : int
        Number of half-integer nodes; the effective cutoff is ``N * h``,
        which must be large enough for the CF tail to have decayed.
    """

    h: float = 0.05
    N: int = 1 << 13

    def u(self) -> np.ndarray:
        if self.N < 2:
            raise ValueError("HilbertGrid: N must be at least 2")
        if self.h <= 0.0:
            raise ValueError("HilbertGrid: h must be > 0")
        return (np.arange(self.N) + 0.5) * self.h


@dataclass(frozen=True)
class ProjGrid:
    """Grid for the PROJ (frame-projection) pricer of Kirkby (2015, 2017).

    PROJ expands the risk-neutral density on a uniform grid of shifted
    B-spline generators of a given ``order`` and recovers the coefficients
    from the characteristic function via a single FFT.

    Parameters
    ----------
    N : int
        Number of grid / basis points. Power of two (e.g. ``2**12``) so the
        FFT is efficient. Effective spatial resolution is ``2*alph/(N-1)``.
    alph : float
        Half-width of the log-return grid; the grid spans ``[-alph, alph]``
        around the (cumulant-centered) mean. Choose ``alph`` large enough to
        cover the tails — a few times ``sqrt(c2)`` plus skew/kurtosis slack.
    order : int
        B-spline order: ``0`` Haar, ``1`` linear, ``2`` quadratic, ``3`` cubic.
        Cubic (the default) gives the highest convergence order and is the
        standard PROJ choice.
    """

    N: int = 1 << 12
    alph: float = 0.5
    order: int = 3

    @property
    def dx(self) -> float:
        """Grid spacing ``2*alph/(N-1)``."""
        return 2.0 * self.alph / (self.N - 1)


@dataclass(frozen=True)
class COSGrid:
    """COS truncation interval [a,b] and number of cosine terms N.

    ``center`` is an optional shift of the integration variable. With
    ``center = m`` the COS expansion variable is

        z = log(S_T / F_0) - m,

    while the payoff is still evaluated on the true asset level
    ``S_T = F_0 * exp(m) * exp(z)``. This lets us keep a symmetric interval
    around zero for the centered state variable without changing the public
    pricing API.
    """

    N: int
    a: float
    b: float
    center: float = 0.0
    label: str = "manual"

    def u(self) -> np.ndarray:
        """Frequency grid for the COS expansion: k * π / (b - a) for k = 0 … N-1."""
        return np.arange(self.N) * np.pi / (self.b - self.a)

    @property
    def width(self) -> float:
        return float(self.b - self.a)

    @property
    def dx(self) -> float:
        """Effective spatial resolution (b - a) / N."""
        return self.width / float(self.N)


@dataclass(frozen=True)
class COSGridPolicy:
    """Adaptive COS grid-selection policy.

    Parameters
    ----------
    mode
        High-level preset. ``"benchmark"`` targets tighter spatial resolution
        while ``"surface"`` trades some accuracy for speed.
    truncation
        One of:
        - ``"heuristic"``  : classic Fang-Oosterlee style rule with a fixed L,
        - ``"tolerance"``  : increase L until a tail proxy drops below
          ``eps_trunc``,
        - ``"paper"``      : reproduce a paper-specified L exactly.
    centered
        If True, the interval is built for the centered variable
        ``X_T - E[X_T]`` and made symmetric around zero.
    dx_target
        Target effective spatial resolution ``(b-a)/N``. If omitted, a
        model-dependent default is chosen.
    fixed_N
        Optional hard override. When omitted, N is chosen adaptively from the
        interval width and ``dx_target``.
    L, paper_L
        Heuristic or paper truncation multipliers.
    eps_trunc
        Target tail proxy for ``truncation="tolerance"``.
    width_fallback, fallback_method
        If the chosen COS interval is too wide, the improved pipeline can route
        the request to another Fourier engine rather than forcing COS into an
        unfavorable geometry.
    """

    mode: str = "benchmark"
    truncation: str = "tolerance"
    centered: bool = True
    dx_target: float | None = None
    fixed_N: int | None = None
    L: float | None = None
    paper_L: float | None = None
    eps_trunc: float = 1e-10
    eps_series: float = 1e-10
    min_N: int = 32
    max_N: int = 16384
    width_fallback: float = 40.0
    fallback_method: str | None = None
