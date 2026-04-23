from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional
from scipy.stats import norm
from ..models.base import ForwardSpec
from ..models.heston import HestonParams


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


# ---------------------------------------------------------------------------
# PyFENG-backed Heston MC baseline.
#
# We keep the in-house conditional MC above as the project's MC *story*
# (exact chi-squared variance + analytic BS conditional on (v_T, V_T)), and
# expose the PyFENG MC engines here as a parallel baseline so the scoreboard
# can quote an externally maintained reference MC without us shipping extra
# scheme implementations.
# ---------------------------------------------------------------------------

#: Short engine name → PyFENG class attribute. The default is Andersen2008
#: (the classic Heston QE scheme), which is accurate and fast.
_PYFENG_HESTON_MC_ENGINES: dict[str, str] = {
    "Andersen2008":       "HestonMcAndersen2008",
    "GlassermanKim2011":  "HestonMcGlassermanKim2011",
    "TseWan2013":         "HestonMcTseWan2013",
    "ChoiKwok2023PoisGe": "HestonMcChoiKwok2023PoisGe",
    "ChoiKwok2023PoisTd": "HestonMcChoiKwok2023PoisTd",
}

_PYFENG_MC_CACHE: dict[tuple, Any] = {}


def _get_pyfeng_mc_model(engine: str, fwd: ForwardSpec, p: HestonParams):
    if engine not in _PYFENG_HESTON_MC_ENGINES:
        raise ValueError(
            f"unknown engine {engine!r}; choose one of "
            f"{sorted(_PYFENG_HESTON_MC_ENGINES)}"
        )
    key = (engine, p, fwd)
    m = _PYFENG_MC_CACHE.get(key)
    if m is not None:
        return m
    try:
        import pyfeng as pf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "heston_mc_pyfeng_price_strip requires pyfeng; install with "
            "`pip install pyfeng`."
        ) from exc
    cls = getattr(pf, _PYFENG_HESTON_MC_ENGINES[engine])
    m = cls(
        sigma=p.v0,     # PyFENG: sigma = v0 (variance)
        vov=p.nu,
        rho=p.rho,
        mr=p.kappa,
        theta=p.theta,
        intr=fwd.r,
        divr=fwd.q,
    )
    _PYFENG_MC_CACHE[key] = m
    return m


def heston_mc_pyfeng_price_strip(
    strikes,
    fwd: ForwardSpec,
    p: HestonParams,
    n_paths: int,
    *,
    engine: str = "Andersen2008",
    seed: int = 7,
    dt: Optional[float] = None,
    antithetic: bool = True,
    cp: int = 1,
) -> np.ndarray:
    """Price a strike strip with a PyFENG Heston MC engine.

    Parameters
    ----------
    strikes :
        1-D iterable of strikes.
    fwd, p :
        Forward spec and :class:`HestonParams` (translated internally to
        PyFENG's ``sigma=v0, vov=nu, mr=kappa, theta=theta, rho=rho``
        keyword layout).
    n_paths :
        Number of MC paths.
    engine :
        One of :data:`_PYFENG_HESTON_MC_ENGINES` keys. Default
        ``"Andersen2008"`` (Heston QE).
    seed :
        Integer RNG seed passed through to PyFENG (``rn_seed``).
    dt :
        Simulation time step. ``None`` → ``T/100``, matching
        :func:`heston_conditional_mc_calls`'s 100-step discretisation so
        the two baselines are comparable.
    antithetic :
        Forwarded to PyFENG's ``set_num_params``. Default ``True``.
    cp :
        ``+1`` calls, ``-1`` puts.

    Returns
    -------
    np.ndarray
        Strip prices, one per strike.
    """
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    m = _get_pyfeng_mc_model(engine, fwd, p)
    step = float(fwd.T) / 100.0 if dt is None else float(dt)
    m.set_num_params(n_path=int(n_paths), dt=step,
                     rn_seed=int(seed), antithetic=antithetic)
    return np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=cp),
                      dtype=np.float64)
