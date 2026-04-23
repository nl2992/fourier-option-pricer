"""Heston-CGMY SVJ model — Heston SV + CGMY tempered-stable jumps.

CGMY (Carr, Geman, Madan, Yor 2002) is a pure-jump, infinite-activity,
tempered-stable Lévy process. Combined with Heston's diffusion we have
the independent-factor decomposition:

    phi(u) = phi_H(u) * phi_jump(u)

The CGMY Lévy exponent ``psi(u)`` gives ``E[exp(i u J_T)] = exp(T psi(u))``.
Because the CF of a Lévy process without drift does *not* satisfy
``E[exp(J_T)] = 1`` in general, we add a deterministic drift
``-psi(-i)`` to land in the project's log-forward convention:

    phi_jump(u) = exp( T * ( psi(u) - i u * psi(-i) ) )

At ``u = -i`` (equivalently ``i u = 1``) the exponent becomes
``T*(psi(-i) - psi(-i)) = 0`` so ``E[exp(X_T^{jump})] = 1``. This is the
same generic "subtract the MGF(1)" trick used across Lévy/Merton/Bates
compensators.

Parameter regime
----------------
We implement the standard form for ``Y in (0, 2) \\ {1}``:

    psi(u) = C * Gamma(-Y) * [ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ].

The ``Y = 1`` case involves a logarithmic branch and is not supported
here. ``M > 1`` is required so the martingale MGF ``E[exp(J_T)]`` is
finite; ``G > 0`` and ``C > 0`` are the usual positivity constraints.

Cumulants
---------
Closed-form cumulants of CGMY have restricted existence regimes. We
instead delegate to :func:`foureng.utils.cumulants.cumulants_from_cf`
which evaluates a 64-point Cauchy integral on a small complex circle
around ``u = 0``. This is the same routine used for Heston and works
for any CF that is analytic near the origin — which CGMY is for the
standard parameter regimes used in practice.

References
----------
* Carr, Geman, Madan, Yor (2002), "The Fine Structure of Asset Returns:
  An Empirical Investigation", Journal of Business 75(2).
* Fang & Oosterlee (2008) — COS truncation rule.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma_fn

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_cf


@dataclass(frozen=True)
class HestonCGMYParams(ModelSpec):
    """Heston-CGMY SVJ parameters.

    Heston block
    ------------
    ``kappa, theta, nu, rho, v0`` — see :class:`HestonParams`.

    CGMY block
    ----------
    ``C >= 0`` : activity / scale. ``C = 0`` is the degenerate no-jump
                case (CGMY collapses to 1) and is accepted so Heston-CGMY
                reduces exactly to Heston when ``C = 0``.
    ``G > 0`` : exponential tilt controlling negative-jump tail.
    ``M > 1`` : exponential tilt controlling positive-jump tail;
                ``M > 1`` required for finite ``E[exp(J_T)]`` and
                therefore a valid martingale correction.
    ``Y``     : tail/activity index, ``Y in (0, 2)`` and ``Y != 1``.
                ``Y < 1`` ⇒ finite variation, ``Y in (1, 2)`` ⇒ infinite
                variation. ``Y = 1`` is excluded (log branch).
    """

    # Heston part
    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    # CGMY part
    C: float
    G: float
    M: float
    Y: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        C: float,
        G: float,
        M: float,
        Y: float,
    ):
        object.__setattr__(self, "name", "heston_cgmy")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "Y", Y)

    @property
    def heston_params(self) -> HestonParams:
        """View of the Heston block for reuse."""
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


# ---------------------------------------------------------------------------
# CGMY Lévy exponent
# ---------------------------------------------------------------------------

def _validate_cgmy(p: HestonCGMYParams) -> None:
    # C == 0 is the degenerate "no CGMY jumps" edge case — the Lévy exponent
    # becomes identically zero and Heston-CGMY collapses to pure Heston.
    # We allow it so the reduction-to-Heston gate can be bit-identical.
    if p.C < 0.0:
        raise ValueError(f"CGMY requires C >= 0; got C={p.C}")
    if p.G <= 0.0:
        raise ValueError(f"CGMY requires G > 0; got G={p.G}")
    if p.M <= 1.0:
        raise ValueError(
            f"CGMY martingale correction requires M > 1 (so E[exp(J_T)] is "
            f"finite); got M={p.M}"
        )
    if not (0.0 < p.Y < 2.0) or abs(p.Y - 1.0) < 1e-12:
        raise ValueError(
            f"CGMY requires Y in (0, 2) with Y != 1; got Y={p.Y}. "
            "The Y = 1 case involves a log-branch not implemented here."
        )


def cgmy_levy_exponent(u: np.ndarray, p: HestonCGMYParams) -> np.ndarray:
    """CGMY characteristic exponent per unit time.

        psi(u) = C * Gamma(-Y) * [ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ]

    with principal-branch complex power ``(.)**Y`` (numpy default).
    Evaluated at complex ``u`` in the returned array.
    """
    _validate_cgmy(p)
    u_c = np.asarray(u, dtype=np.complex128)
    Y = p.Y
    gY = _gamma_fn(-Y)  # Γ(-Y) is a real scalar for Y in (0,2)\{1}
    term_plus = (p.M - 1j * u_c) ** Y - p.M ** Y
    term_minus = (p.G + 1j * u_c) ** Y - p.G ** Y
    return p.C * gY * (term_plus + term_minus)


# ---------------------------------------------------------------------------
# Full Heston-CGMY CF
# ---------------------------------------------------------------------------

def heston_cgmy_cf(
    u: np.ndarray, fwd: ForwardSpec, p: HestonCGMYParams
) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under Heston-CGMY SVJ.

    Independent factorisation:

        phi(u) = phi_H(u) * exp( T * ( psi(u) - i u * psi(-i) ) )

    with ``phi_H`` from the PyFENG-backed :func:`heston_cf` and the
    drift correction ``-psi(-i)`` imposing ``E[exp(X_T^{jump})] = 1``.
    """
    T = fwd.T
    u_c = np.asarray(u, dtype=np.complex128)

    phi_H = heston_cf(u_c, fwd, p.heston_params)

    psi_u = cgmy_levy_exponent(u_c, p)
    # psi(-i) is real for CGMY in the admitted regime (M > 1, G > 0):
    # evaluating at u=-i gives real arguments (M-1)^Y, (G+1)^Y etc.
    psi_minus_i = cgmy_levy_exponent(np.array([-1j], dtype=np.complex128), p)[0]
    phi_jump = np.exp(T * (psi_u - 1j * u_c * psi_minus_i))

    return phi_H * phi_jump


# ---------------------------------------------------------------------------
# Cumulants for COS (numerical via Cauchy integral on the CF)
# ---------------------------------------------------------------------------

def heston_cgmy_cumulants(
    fwd: ForwardSpec, p: HestonCGMYParams
) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under Heston-CGMY.

    Computed numerically from the CF via
    :func:`foureng.utils.cumulants.cumulants_from_cf` — the same
    FFT-on-circle routine used for Heston. No closed-form CGMY moments
    required; works as long as the CF is analytic on a small
    neighbourhood of ``u = 0``, which holds for the supported parameter
    regime (``Y in (0, 2) \\ {1}``, ``M > 1``, ``G > 0``, ``C > 0``).
    """
    from ..utils.cumulants import cumulants_from_cf

    phi = lambda u: heston_cgmy_cf(u, fwd, p)
    c = cumulants_from_cf(phi, order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
