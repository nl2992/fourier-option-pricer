"""SVJJ model  -  Heston SV with simultaneous jumps in price and variance.

The SVJJ (Stochastic Volatility with Jumps in both returns and Variance)
model extends Heston by adding a compound Poisson process whose arrivals
trigger jumps in *both* the log-price and the variance simultaneously:

    d log S = (r - q - V_t/2 - lambda*m_J) dt + sqrt(V_t) dW_S + J_S dN_t
    dV_t    = kappa*(theta - V_t) dt + nu*sqrt(V_t) dW_V + J_V dN_t

    <dW_S, dW_V> = rho dt
    N_t  ~  Poisson(lambda)
    J_V  ~  Exp(mu_V)                       (non-negative variance jump)
    J_S | J_V  ~  N(mu_J + rho_J * J_V, sigma_J^2)  (correlated price jump)

The term ``-lambda*m_J`` in the price drift is the compensator that keeps
``S_t / F_t`` a martingale under Q, where:

    m_J = E[exp(J_S)] - 1 = exp(mu_J + sigma_J^2/2) / (1 - rho_J*mu_V) - 1

Characteristic function
-----------------------
The SVJJ model is affine in (log S, V), so the CF has the form:

    phi(u) = exp(C(u,T) + D(u,T) * v0)

Key insight: because the jump intensity lambda is *state-independent*, the
Riccati ODE for D(u,T) is unchanged from the standard Heston model.  Only
the C equation picks up the jump contribution.  Writing:

    Theta_J(u, D) = E[exp(iu*J_S + D*J_V)]
                  = exp(iu*mu_J - u^2*sigma_J^2/2) / (1 - mu_V*(iu*rho_J + D))

the system decomposes as:

    D(u, T)  =  D_Heston(u, T)                          [analytic]

    C(u, T)  =  C_Heston(u, T)
               + integral_0^T lambda * (Theta_J(u, D_H(tau)) - 1)
                             - iu * lambda * m_J   d tau   [numerical]

The Heston (C, D) are obtained via :func:`heston_riccati_cd`.  The scalar
integral over tau is evaluated with a 200-point trapezoidal rule; for
typical maturities T <= 5 and smooth integrands this gives errors < 1e-10.

Special cases
-------------
* mu_V = 0, rho_J = 0: jumps in S only → reduces to Bates (1996).
* lambda = 0: no jumps → reduces to pure Heston.

References
----------
* Duffie, D., Pan, J. & Singleton, K. (2000), "Transform Analysis and
  Asset Pricing for Affine Jump-Diffusions", *Econometrica*, 68(6), 1343-76.
* Pan, J. (2002), "The Jump-Risk Premia Implicit in Options", *Journal of
  Financial Economics*, 63, 3-50.
* Eraker, B., Johannes, M. & Polson, N. (2003), "The Impact of Jumps in
  Volatility and Returns", *Journal of Finance*, 58(3), 1269-1300.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec
from .heston import HestonParams, heston_riccati_cd

# np.trapezoid was added in NumPy 2.0; np.trapz is deprecated there and
# removed from stubs.  This shim works on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, "trapezoid", np.trapz)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SVJJParams(ModelSpec):
    """SVJJ parameters (Duffie-Pan-Singleton 2000).

    Heston SV block
    ---------------
    kappa : float  Mean-reversion speed of variance. ``> 0``.
    theta : float  Long-run mean of variance. ``> 0``.
    nu : float     Vol-of-vol. ``> 0`` (PyFENG requirement for D).
    rho : float    Price-variance correlation. ``[-1, 1]``.
    v0 : float     Initial variance. ``> 0``.

    Jump block
    ----------
    lam : float     Poisson jump arrival intensity. ``>= 0``.
    mu_J : float    Mean of the normal price-jump conditional on J_V=0.
    sigma_J : float Std-dev of the price-jump. ``>= 0``.
    mu_V : float    Mean of the exponential variance-jump. ``>= 0``.
                    ``mu_V = 0`` gives no variance jumps (Bates model).
    rho_J : float   Correlation loading: J_S | J_V ~ N(mu_J + rho_J*J_V, sigma_J^2).
                    Requires ``1 - rho_J * mu_V > 0``.
    """

    kappa: float
    theta: float
    nu: float
    rho: float
    v0: float
    lam: float
    mu_J: float
    sigma_J: float
    mu_V: float
    rho_J: float

    def __init__(
        self,
        kappa: float,
        theta: float,
        nu: float,
        rho: float,
        v0: float,
        lam: float,
        mu_J: float,
        sigma_J: float,
        mu_V: float,
        rho_J: float,
    ):
        if not (np.isfinite(kappa) and kappa > 0):
            raise ValueError(f"SVJJParams: kappa must be > 0; got {kappa}")
        if not (np.isfinite(theta) and theta > 0):
            raise ValueError(f"SVJJParams: theta must be > 0; got {theta}")
        if not (np.isfinite(nu) and nu > 0):
            raise ValueError(f"SVJJParams: nu must be > 0; got {nu}")
        if not (np.isfinite(rho) and -1.0 <= rho <= 1.0):
            raise ValueError(f"SVJJParams: rho must be in [-1,1]; got {rho}")
        if not (np.isfinite(v0) and v0 > 0):
            raise ValueError(f"SVJJParams: v0 must be > 0; got {v0}")
        if not (np.isfinite(lam) and lam >= 0):
            raise ValueError(f"SVJJParams: lam must be >= 0; got {lam}")
        if not np.isfinite(mu_J):
            raise ValueError(f"SVJJParams: mu_J must be finite; got {mu_J}")
        if not (np.isfinite(sigma_J) and sigma_J >= 0):
            raise ValueError(f"SVJJParams: sigma_J must be >= 0; got {sigma_J}")
        if not (np.isfinite(mu_V) and mu_V >= 0):
            raise ValueError(f"SVJJParams: mu_V must be >= 0; got {mu_V}")
        if not np.isfinite(rho_J):
            raise ValueError(f"SVJJParams: rho_J must be finite; got {rho_J}")
        if 1.0 - rho_J * mu_V <= 0:
            raise ValueError(
                f"SVJJParams: existence condition 1 - rho_J*mu_V must be > 0; "
                f"got rho_J={rho_J}, mu_V={mu_V}"
            )
        object.__setattr__(self, "name", "svjj")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "v0", v0)
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "mu_J", mu_J)
        object.__setattr__(self, "sigma_J", sigma_J)
        object.__setattr__(self, "mu_V", mu_V)
        object.__setattr__(self, "rho_J", rho_J)

    @property
    def heston_params(self) -> HestonParams:
        return HestonParams(
            kappa=self.kappa,
            theta=self.theta,
            nu=self.nu,
            rho=self.rho,
            v0=self.v0,
        )


# ---------------------------------------------------------------------------
# Jump transform
# ---------------------------------------------------------------------------


def _theta_J(u_c: np.ndarray, D: np.ndarray, p: SVJJParams) -> np.ndarray:
    """Joint CF of a single jump: E[exp(iu*J_S + D*J_V)].

    Theta_J(u, D) = exp(iu*mu_J - sigma_J^2*u^2/2) / (1 - mu_V*(iu*rho_J + D))

    Parameters
    ----------
    u_c : complex array  (frequency grid)
    D   : complex array  (Heston Riccati D at a given time tau; same shape as u_c)
    p   : SVJJParams
    """
    numer = np.exp(1j * u_c * p.mu_J - 0.5 * p.sigma_J**2 * u_c**2)
    denom = 1.0 - p.mu_V * (1j * p.rho_J * u_c + D)
    return numer / denom


# ---------------------------------------------------------------------------
# SVJJ characteristic function
# ---------------------------------------------------------------------------


def svjj_cf(u: np.ndarray, fwd: ForwardSpec, p: SVJJParams) -> np.ndarray:
    """CF of X_T = log(S_T/F_0) under SVJJ.

    Uses the affine decomposition:
        phi(u) = exp(C(u,T) + D(u,T)*v0)
    where D = D_Heston (analytic) and C = C_Heston + C_jump (numerical
    200-point trapezoidal integral over the D_Heston path).
    """
    u_c = np.asarray(u, dtype=np.complex128)
    T = fwd.T

    # Heston analytic (C, D)  -- shape: (len(u),)
    C_H, D_H_T = heston_riccati_cd(u_c, T, p.heston_params)

    if p.lam == 0.0:
        # Pure Heston: no jump correction needed
        return np.exp(C_H + D_H_T * p.v0)

    # Martingale compensator m_J = E[exp(J_S)] - 1
    # At u=-i, D_J_V component: rho_J*mu_V (from Exp(mu_V) with rho_J loading)
    m_J = np.exp(p.mu_J + 0.5 * p.sigma_J**2) / (1.0 - p.rho_J * p.mu_V) - 1.0

    # Numerical integral of lambda*(Theta_J(u, D_H(tau)) - 1) - iu*lambda*m_J
    # over tau in [0, T].  D_H(tau) is the analytic Heston D at time tau.
    N_tau = 200
    taus = np.linspace(0.0, T, N_tau + 1)  # shape (N_tau+1,)

    # Vectorise: compute D_H at every tau for all u simultaneously.
    # heston_riccati_cd expects scalar T; loop over tau grid.
    # For efficiency, broadcast: u_c shape (M,), taus shape (N_tau+1,) → (M, N_tau+1)
    u_col = u_c[:, np.newaxis]  # (M, 1)
    taus_row = taus[np.newaxis, :]  # (1, N_tau+1)

    _, D_grid = heston_riccati_cd(
        u_col.repeat(taus_row.shape[1], axis=1).ravel(),
        taus_row.repeat(u_col.shape[0], axis=0).ravel(),
        p.heston_params,
    )
    D_grid = D_grid.reshape(len(u_c), N_tau + 1)  # (M, N_tau+1)

    u_col2 = u_c[:, np.newaxis]  # (M, 1)
    Theta = _theta_J(u_col2, D_grid, p)  # (M, N_tau+1)

    # Integrand: lambda*(Theta - 1) - iu*lambda*m_J
    integrand = p.lam * (Theta - 1.0) - 1j * u_col2 * p.lam * m_J  # (M, N_tau+1)

    # Trapezoidal rule along tau axis
    C_jump = _trapezoid(integrand, taus_row, axis=1)  # (M,)

    return np.exp(C_H + C_jump + D_H_T * p.v0)


# ---------------------------------------------------------------------------
# Cumulants
# ---------------------------------------------------------------------------


def svjj_cumulants(fwd: ForwardSpec, p: SVJJParams) -> tuple[float, float, float]:
    """Cumulants (c1, c2, c4) of X_T under SVJJ via Cauchy integral."""
    from ..utils.cumulants import cumulants_from_cf

    c = cumulants_from_cf(lambda u: svjj_cf(u, fwd, p), order=4, radius=0.25, M=64)
    return float(c[0]), float(c[1]), float(c[3])
