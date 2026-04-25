"""Alpha-validity checks for the damped Carr-Madan / FRFT call integrand.

The damped call transform (Carr-Madan 1999 eq. 6) is
    psi(v) = exp(-rT) * phi_logS(v - (alpha+1)i)
             / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)
For this to be integrable we need phi to be finite and analytic at
u = v - (alpha+1)i for all real v, which in turn requires
    E[S_T^{alpha+1}] < infinity.

Different models have different constraints. A few are clean (and cheap to
compute analytically); for the rest we fall back to a generic runtime
finiteness probe of phi at u = -i*(alpha+1).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..models.base import CharFunc
from ..models.kou import KouParams
from ..models.variance_gamma import VGParams


@dataclass(frozen=True)
class AlphaCheck:
    ok: bool
    reason: str
    phi_at_damping: complex | None = None


def check_alpha(phi: CharFunc, alpha: float) -> AlphaCheck:
    """Generic runtime check: evaluate phi at u = -i*(alpha+1) and nearby points.

    Passes iff phi is finite (non-NaN, non-inf) at u = -i*(alpha+1) and at
    u = -i*(alpha+1) +/- 0.5. That rules out cases where alpha lands on or
    past a pole.

    Use this whenever you aren't sure whether a particular alpha is admissible
    for a given model/parameter set.
    """
    if alpha <= 0.0:
        return AlphaCheck(False, f"alpha must be > 0 for damping (got {alpha})")

    u0 = -1j * (alpha + 1.0)
    probes = np.array([u0 - 0.5, u0, u0 + 0.5], dtype=np.complex128)
    try:
        vals = phi(probes)
    except Exception as exc:  # pragma: no cover
        return AlphaCheck(False, f"phi raised at u=-i(alpha+1): {exc!r}")

    if not np.all(np.isfinite(vals.real)) or not np.all(np.isfinite(vals.imag)):
        bad = vals[np.logical_or(~np.isfinite(vals.real), ~np.isfinite(vals.imag))]
        return AlphaCheck(False, f"phi non-finite near damping line: {bad}",
                          phi_at_damping=complex(vals[1]))

    # If phi(-i*(alpha+1)) has huge magnitude, we're probably near a pole.
    mag = float(np.abs(vals[1]))
    if mag > 1e12:
        return AlphaCheck(False, f"|phi(-i(alpha+1))| = {mag:.3e} — pole nearby",
                          phi_at_damping=complex(vals[1]))

    return AlphaCheck(True, "phi finite at damping line", phi_at_damping=complex(vals[1]))


# --- Model-specific analytic bounds ------------------------------------------

def kou_alpha_max(p: KouParams) -> float:
    """Largest valid alpha for Kou: needs eta1 > alpha + 1, so alpha < eta1 - 1.

    Derivation: at u = -i*(alpha+1), the Kou CF contains
        eta1 / (eta1 - i*u) = eta1 / (eta1 - (alpha+1))
    which blows up as alpha -> eta1 - 1.
    """
    return float(p.eta1 - 1.0)


def vg_alpha_max(p: VGParams) -> float:
    """Largest valid alpha for VG.

    The VG CF is (1 - i*theta*nu*u + 0.5*sigma^2*nu*u^2)^{-T/nu}.
    At u = -i*s (s real > 0), the base becomes
        1 - theta*nu*s - 0.5*sigma^2*nu*s^2.
    This must stay strictly positive, so
        s^2 + (2*theta/sigma^2)*s - 2/(sigma^2*nu) < 0
    Positive root of the quadratic:
        s_* = -theta/sigma^2 + sqrt(theta^2/sigma^4 + 2/(sigma^2*nu))
    and we need s = alpha + 1 < s_*, so alpha < s_* - 1.
    """
    sigma, nu, theta = p.sigma, p.nu, p.theta
    s_star = -theta / sigma**2 + np.sqrt(theta**2 / sigma**4 + 2.0 / (sigma**2 * nu))
    return float(s_star - 1.0)


def assert_alpha_valid(
    phi: CharFunc,
    alpha: float,
    model_params: object | None = None,
) -> None:
    """Raise ValueError if alpha is outside the admissible range.

    model_params, if supplied (KouParams or VGParams), adds a cheap analytic
    bound check *in addition to* the runtime probe. That catches
    "borderline-but-still-finite" alphas that would give a numerically
    unstable psi(v).
    """
    if isinstance(model_params, KouParams):
        amax = kou_alpha_max(model_params)
        if alpha >= amax:
            raise ValueError(
                f"Kou: alpha={alpha} >= eta1-1={amax}; damped call integrand diverges"
            )
    elif isinstance(model_params, VGParams):
        amax = vg_alpha_max(model_params)
        if alpha >= amax:
            raise ValueError(
                f"VG: alpha={alpha} >= alpha_max={amax:.4f}; "
                "E[S^{alpha+1}] is infinite"
            )

    chk = check_alpha(phi, alpha)
    if not chk.ok:
        raise ValueError(f"alpha={alpha} invalid: {chk.reason}")
