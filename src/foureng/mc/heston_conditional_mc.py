from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from ..char_func.heston import HestonParams


@dataclass(frozen=True)
class HestonMCScheme:
    n_paths: int
    n_steps: int
    seed: int | None = None
    scheme: str = "exact"   # "exact" or "milstein"


def _sim_var_exact(v0, kappa, theta, nu, T, n_paths, n_steps, rng):
    """CIR variance via exact noncentral chi-squared sampling.

    Returns (v_T, V_T) where V_T = int_0^T v_s ds approximated by trapezoidal rule.
    """
    h = T / n_steps
    df = 4.0 * kappa * theta / (nu * nu)
    v = np.full(n_paths, v0, dtype=float)
    VT = v * (h / 2.0)
    c = nu * nu * (1.0 - np.exp(-kappa * h)) / (4.0 * kappa)
    exp_kh = np.exp(-kappa * h)
    for s in range(1, n_steps + 1):
        lam = v * exp_kh / c
        v = c * rng.noncentral_chisquare(df, lam, n_paths)
        v = np.maximum(v, 0.0)
        VT += v * (h / 2.0 if s == n_steps else h)
    return v, VT


def _sim_var_milstein(v0, kappa, theta, nu, T, n_paths, n_steps, rng):
    """Milstein discretisation of the CIR variance process (reflect at 0)."""
    h = T / n_steps
    v = np.full(n_paths, v0, dtype=float)
    VT = v * (h / 2.0)
    for s in range(1, n_steps + 1):
        Z = rng.standard_normal(n_paths)
        sv = np.sqrt(np.maximum(v, 0.0))
        v = v + kappa * (theta - v) * h + nu * sv * Z * np.sqrt(h) \
              + 0.25 * nu * nu * (Z * Z * h - h)
        v = np.maximum(v, 0.0)
        VT += v * (h / 2.0 if s == n_steps else h)
    return v, VT


def heston_conditional_mc_calls(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    mc: HestonMCScheme,
) -> np.ndarray:
    """Heston conditional MC (Module 8): simulate (v_T, V_T), then BS-analytic per path.

    log(S_T/F0) | v_T, V_T ~ N( mu_cond, sigma_cond^2 ) with
      mu_cond    = (rho/nu)*(v_T - v0 + kappa*(V_T - theta*T)) - 0.5*(1-rho^2)*V_T
      sigma_cond = sqrt( (1 - rho^2) * V_T )
    Then price each strike by BS-style integration across paths.
    """
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    rng = np.random.default_rng(mc.seed)
    sim = _sim_var_exact if mc.scheme == "exact" else _sim_var_milstein
    v_T, V_T = sim(p.v0, p.kappa, p.theta, p.nu, T, mc.n_paths, mc.n_steps, rng)

    F0 = S0 * np.exp((r - q) * T)
    # log(S_T/F0) | (v_T, V_T) ~ N(mu, sigma^2) with
    #   mu    = (rho/nu)*(v_T - v0 - kappa*theta*T + kappa*V_T) - 0.5*V_T
    #   sigma = sqrt((1 - rho^2)*V_T)
    adj = (p.rho / p.nu) * (v_T - p.v0 + p.kappa * (V_T - p.theta * T))
    mu = adj - 0.5 * V_T
    sigma2 = (1.0 - p.rho * p.rho) * V_T
    sigma = np.sqrt(np.maximum(sigma2, 1e-16))

    # conditional BS call per path, per strike
    logFK = np.log(F0 / K)                       # (nK,)
    # per path (nPaths,), per strike: d1 = (log(F0/K) + mu + sigma^2) / sigma
    d1 = (logFK[None, :] + mu[:, None] + sigma[:, None] ** 2) / sigma[:, None]
    d2 = d1 - sigma[:, None]
    call_paths = F0 * np.exp(mu[:, None] + 0.5 * sigma[:, None] ** 2) * norm.cdf(d1) \
                 - K[None, :] * norm.cdf(d2)
    return np.exp(-r * T) * call_paths.mean(axis=0)
