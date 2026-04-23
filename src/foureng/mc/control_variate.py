"""Monte Carlo with control variates.

Two constructions are provided:

1. Underlying-as-control for plain BS MC:
       X = exp(-rT) * (S_T - K)^+
       Y = exp(-rT) * S_T         (E[Y] = S_0 * exp(-qT) in general; we use
                                   the discounted forward so E[Y] is exact)
   Optimal coefficient c* = Cov(X, Y) / Var(Y) estimated from the same
   paths (this introduces a tiny bias O(1/n_paths) compared to the plug-in
   coefficient, but is the standard Glasserman chapter-4 recipe).

2. Integrated-variance-as-control for Heston:
       X = Heston MC conditional-BS call payoff
       Y = V_T  (path-integrated variance int_0^T v_s ds)
   The expectation E[V_T] is available in closed form from the CIR
   dynamics,
       E[V_T] = theta*T + (v0 - theta) * (1 - exp(-kappa*T)) / kappa,
   which makes Y a bias-free control with known mean. The correlation
   with the call payoff is high because the conditional call is monotone
   increasing in V_T (it is a BS call at total variance V_T). This is the
   philosophy of a "Fourier-price-style" control variate adapted to the
   conditional-MC scheme: a cheap statistic of the simulated path whose
   expectation is already known from the model, subtracted without bias.

Both routines return a ``CVResult`` with the raw MC mean, the CV-adjusted
mean, both standard errors, and the variance-reduction ratio. The expected
reduction is regime-dependent: for the BS call with moderate moneyness the
underlying control typically cuts variance 2x-5x; the Fourier control on
Heston can do substantially better when the Heston is close to BS.

PyFENG-backed variance sampler (Heston side)
--------------------------------------------
The Heston control variate now samples ``(v_T, V_T)`` through PyFENG's
:class:`pyfeng.HestonMcGlassermanKim2011` in a **single endpoint call**:
``cond_states_step(T, v_0)`` returns ``(v_T, avgvar, _extra)`` where
``avgvar * T`` is an exact gamma-series draw of
``V_T = int_0^T v_s ds`` conditional on ``v_0`` (Glasserman & Kim
2011, formula 2.7). The joint law of ``(v_T, V_T)`` produced this way
has **zero discretisation error**, so we do not sub-step -- one call
is both correct and cheaper than the previous ``n_steps``-step
in-house scheme (exact NCX² for ``v_t`` + trapezoidal ``V_T``).
This also removes the only cross-module dependency on
``mc.heston_conditional_mc._sim_var_exact``; the in-house conditional
MC is still available in :mod:`foureng.mc.heston_conditional_mc` as a
pedagogical baseline, but the control variate no longer needs it.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple

from ..models.base import ForwardSpec
from ..models.heston import HestonParams
from ..iv.implied_vol import bs_price_from_fwd, BSInputs


@dataclass(frozen=True)
class CVResult:
    price_plain: float
    price_cv: float
    se_plain: float
    se_cv: float
    var_reduction: float
    n_paths: int


def _sample_cv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """c* = Cov(X, Y) / Var(Y), guarded against zero-variance controls."""
    vy = float(np.var(Y, ddof=1))
    if vy < 1e-30:
        return 0.0
    return float(np.cov(X, Y, ddof=1)[0, 1] / vy)


def bs_call_cv(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    vol: float,
    n_paths: int,
    seed: int | None = None,
) -> CVResult:
    """BS European call by Monte Carlo with S_T as control variate.

    Uses exact GBM in a single time step; the control is the discounted
    terminal price, which is a martingale with known expectation F_0*exp(-rT).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * vol * vol) * T + vol * np.sqrt(T) * Z)

    disc = np.exp(-r * T)
    X = disc * np.maximum(ST - K, 0.0)
    Y = disc * ST
    EY = S0 * np.exp(-q * T)  # exact expectation

    c = _sample_cv_coefficient(X, Y)
    Xcv = X - c * (Y - EY)

    mean_plain = float(np.mean(X))
    mean_cv = float(np.mean(Xcv))
    se_plain = float(np.std(X, ddof=1) / np.sqrt(n_paths))
    se_cv = float(np.std(Xcv, ddof=1) / np.sqrt(n_paths))
    vr = (se_plain / se_cv) ** 2 if se_cv > 0 else float("inf")

    return CVResult(
        price_plain=mean_plain, price_cv=mean_cv,
        se_plain=se_plain, se_cv=se_cv,
        var_reduction=vr, n_paths=n_paths,
    )


def heston_call_bs_control(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    n_paths: int,
    n_steps: int = 100,
    seed: int | None = None,
) -> CVResult:
    """Heston European call by conditional MC with V_T control variate.

    Simulates (v_T, V_T) under the CIR exact scheme (same as
    ``heston_conditional_mc_calls``) and forms X = per-path conditional BS
    call. The control variate is Y = V_T (the sample path-integrated
    variance), whose expectation under CIR is known analytically:

        E[V_T] = theta*T + (v0 - theta) * (1 - exp(-kappa*T)) / kappa.

    V_T is monotonically coupled to X (the conditional BS call is
    increasing in total variance), so the control correlation is high
    without introducing any bias -- in contrast to nonlinear per-path
    "BS-call control" attempts whose expectation is not known in closed
    form.

    Variance sampling is delegated to PyFENG's
    :class:`HestonMcGlassermanKim2011` in a **single** endpoint call:
    ``cond_states_step(T, v_0)`` returns ``(v_T, avg_var_on_[0,T],
    _extra)`` where ``avg_var_on_[0,T] * T`` is an exact gamma-series
    draw of ``V_T = int_0^T v_s ds`` conditional on ``v_0`` (Glasserman
    & Kim, 2011, formula 2.7). This is the natural use of that engine:
    with zero discretisation error, one call is both correct and
    cheaper than the previous ``n_steps``-step in-house loop, and
    removes the only cross-module dependency on
    ``heston_conditional_mc._sim_var_exact``.

    The ``n_steps`` parameter is retained in the signature for
    back-compatibility but is ignored -- the GK2011 endpoint sampler
    does not need sub-stepping.
    """
    from scipy.stats import norm

    try:
        import pyfeng as pf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "heston_call_bs_control requires pyfeng; install with "
            "`pip install pyfeng`."
        ) from exc

    del n_steps  # honoured via docstring; GK2011 endpoint sampler needs no substepping
    m = pf.HestonMcGlassermanKim2011(
        sigma=p.v0, vov=p.nu, rho=p.rho, mr=p.kappa, theta=p.theta,
        intr=r, divr=q,
    )
    m.set_num_params(
        n_path=int(n_paths),
        dt=float(T),
        rn_seed=seed,
        antithetic=False,  # preserve prior in-house behavior (no antithetic)
    )

    v_0_vec = np.full(int(n_paths), p.v0, dtype=float)
    v_T, avg_var, _extra = m.cond_states_step(float(T), v_0_vec)
    V_T = np.asarray(avg_var, dtype=float) * float(T)
    v_T = np.asarray(v_T, dtype=float)

    F0 = S0 * np.exp((r - q) * T)
    disc = np.exp(-r * T)

    # Conditional mean / vol of log(S_T / F_0) given (v_T, V_T).
    adj = (p.rho / p.nu) * (v_T - p.v0 + p.kappa * (V_T - p.theta * T))
    mu_cond = adj - 0.5 * V_T
    sigma2_cond = (1.0 - p.rho * p.rho) * V_T
    sigma_cond = np.sqrt(np.maximum(sigma2_cond, 1e-16))

    logFK = np.log(F0 / K)
    d1 = (logFK + mu_cond + sigma2_cond) / sigma_cond
    d2 = d1 - sigma_cond
    X = disc * (F0 * np.exp(mu_cond + 0.5 * sigma2_cond) * norm.cdf(d1)
                - K * norm.cdf(d2))

    # Integrated-variance control with exact mean.
    Y = V_T
    EY = p.theta * T + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / p.kappa

    c = _sample_cv_coefficient(X, Y)
    Xcv = X - c * (Y - EY)

    mean_plain = float(np.mean(X))
    mean_cv = float(np.mean(Xcv))
    se_plain = float(np.std(X, ddof=1) / np.sqrt(n_paths))
    se_cv = float(np.std(Xcv, ddof=1) / np.sqrt(n_paths))
    vr = (se_plain / se_cv) ** 2 if se_cv > 0 else float("inf")

    return CVResult(
        price_plain=mean_plain, price_cv=mean_cv,
        se_plain=se_plain, se_cv=se_cv,
        var_reduction=vr, n_paths=n_paths,
    )
