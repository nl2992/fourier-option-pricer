"""Calibrate any registry model to option quotes with CF-based gradients.

``calibrate(model, quotes, initial)`` fits the float parameters of any model in
``MODEL_REGISTRY`` to implied vols (or prices) across several maturities.

Objective
    Out-of-the-money option prices, each residual divided by the market Black
    vega, ``r_i = sqrt(w_i) (P_model - P_market) / vega_i``. To first order
    this is the implied-vol error, but it needs no implied-vol inversion inside
    the loop.
Pricing and gradient
    For each maturity a COS grid is fixed at the start of a solve. Prices are
    then linear in the CF values, so the Jacobian is one extra COS sum per
    parameter using ``d phi / d theta`` (analytic for BSM, Merton, Kou, VG,
    NIG, CGMY, Heston and Bates; central differences of the CF otherwise). With
    the grid fixed the objective is smooth, which is what gradient methods
    need; after convergence the grid is rebuilt at the solution and the solve
    repeated until the prices stop moving.
Solver
    ``scipy.optimize.least_squares`` with the trust-region reflective method,
    which handles the parameter bounds and behaves like Levenberg-Marquardt
    away from them.

Reference
---------
Cui, Y., del Bano Rollin, S. & Germano, G. (2017), Full and fast calibration of
the Heston stochastic volatility model, *European Journal of Operational
Research* 263(2), 625-638.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from ..iv.lets_be_rational import black_price, implied_vol_lets_be_rational
from ..models.base import ForwardSpec
from ..models.cf_gradients import (
    ANALYTIC_GRADIENT_MODELS,
    cf_and_gradient,
    param_names,
    params_from_vector,
    params_to_vector,
)
from ..models.registry import MODEL_REGISTRY

DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "bsm": {"sigma": (1e-4, 5.0)},
    "heston": {
        "kappa": (1e-3, 20.0),
        "theta": (1e-4, 2.0),
        "nu": (1e-3, 5.0),
        "rho": (-0.999, 0.999),
        "v0": (1e-4, 2.0),
    },
    "bates": {
        "kappa": (1e-3, 20.0),
        "theta": (1e-4, 2.0),
        "nu": (1e-3, 5.0),
        "rho": (-0.999, 0.999),
        "v0": (1e-4, 2.0),
        "lam_j": (0.0, 20.0),
        "mu_j": (-1.0, 1.0),
        "sigma_j": (1e-4, 1.0),
    },
    "vg": {"sigma": (1e-3, 2.0), "nu": (1e-4, 5.0), "theta": (-2.0, 2.0)},
    "nig": {"sigma": (1e-3, 2.0), "nu": (1e-3, 5.0), "theta": (-2.0, 2.0)},
    "cgmy": {"C": (1e-4, 20.0), "G": (1e-3, 50.0), "M": (1.0 + 1e-3, 50.0), "Y": (1e-4, 1.99)},
    "kou": {
        "sigma": (1e-3, 2.0),
        "lam": (0.0, 20.0),
        "p": (1e-3, 1.0 - 1e-3),
        "eta1": (1.0 + 1e-3, 50.0),
        "eta2": (1e-3, 50.0),
    },
    "merton_jd": {
        "sigma": (1e-3, 2.0),
        "lam": (0.0, 20.0),
        "muj": (-1.0, 1.0),
        "sigj": (1e-4, 1.0),
    },
}


@dataclass(frozen=True)
class MarketQuotes:
    """European option quotes on one underlying, as flat arrays.

    Give either ``ivs`` (Black implied vols) or ``prices`` with ``cp``
    (+1 call, -1 put). ``weights`` multiply the squared residuals.
    """

    S0: float
    r: float
    q: float
    maturities: np.ndarray
    strikes: np.ndarray
    ivs: np.ndarray | None = None
    prices: np.ndarray | None = None
    cp: np.ndarray | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        T = np.atleast_1d(np.asarray(self.maturities, dtype=float))
        K = np.atleast_1d(np.asarray(self.strikes, dtype=float))
        if T.shape != K.shape or T.ndim != 1:
            raise ValueError("MarketQuotes: maturities and strikes must be 1-D and the same length")
        if not (np.all(T > 0) and np.all(K > 0)):
            raise ValueError("MarketQuotes: maturities and strikes must be > 0")
        if (self.ivs is None) == (self.prices is None):
            raise ValueError("MarketQuotes: give exactly one of ivs or prices")
        for name in ("ivs", "prices", "cp", "weights"):
            val = getattr(self, name)
            if val is not None and np.shape(val) != T.shape:
                raise ValueError(f"MarketQuotes: {name} must have the same length as strikes")
        if self.prices is not None and self.cp is None:
            raise ValueError("MarketQuotes: prices need cp (+1 call, -1 put)")

    @classmethod
    def from_surface(cls, spec, market_ivs, weights=None) -> MarketQuotes:
        """Quotes from a :class:`~foureng.surface.vol_surface.SurfaceSpec` grid of vols."""
        TT, KK = np.meshgrid(np.asarray(spec.maturities), np.asarray(spec.strikes), indexing="ij")
        w = None if weights is None else np.asarray(weights, dtype=float).ravel()
        return cls(
            spec.S0,
            spec.r,
            spec.q,
            TT.ravel(),
            KK.ravel(),
            ivs=np.asarray(market_ivs, dtype=float).ravel(),
            weights=w,
        )


@dataclass
class CalibrationFit:
    """Result of :func:`calibrate`.

    ``iv_residuals`` are model minus market implied vols (in quote order),
    recomputed at the solution with Let's Be Rational.
    """

    model: str
    params: object
    values: dict
    iv_residuals: np.ndarray
    rmse_iv: float
    success: bool
    message: str
    nfev: int
    njev: int
    rounds: int
    gradient: str
    cost_history: list = field(default_factory=list)


def _truncation(phi, fwd, params, cumulants, log_tol: float = 34.0) -> tuple[float, float]:
    """COS interval for ``X = log(S_T / F)``: 12 cumulant widths, widened by a Chernoff bound.

    The cumulant rule can clip exponential jump tails at short maturities, so
    each end also satisfies ``P(X < a) <= E[exp(-s X)] exp(s a) <= e^-log_tol``
    (and the mirror bound above) for the best ``s`` at which the moment exists.
    Closed-form CFs continue past the moment-explosion point with finite values
    that are not moments, so ``s`` is walked on a fine grid and only the prefix
    where ``log E[exp(-s X)]`` is finite and convex is used.
    """
    c1, c2, c4 = cumulants
    half = 12.0 * np.sqrt(max(c2, 1e-12) + np.sqrt(abs(c4)))
    a0, b0 = c1 - half, c1 + half
    s = 0.25 * np.arange(1, 129)
    with np.errstate(all="ignore"):
        for sign in (1.0, -1.0):
            try:
                m = np.asarray(phi(1j * sign * s, fwd, params))  # E[exp(-sign s X)]
            except (ValueError, FloatingPointError, OverflowError):
                continue
            ok = np.isfinite(m) & (np.abs(np.imag(m)) <= 1e-8 * np.abs(m)) & (np.real(m) > 0)
            L = np.concatenate(([0.0], np.log(np.where(ok, np.real(m), 1.0))))
            convex = np.concatenate(([True], np.diff(L, 2) >= -1e-9 * (1.0 + np.abs(L[1:-1]))))
            valid = np.cumprod(ok & convex).astype(bool)
            n_ok = int(np.sum(valid)) - 2  # stay clear of a nearby pole
            if n_ok < 1:
                continue
            ends = (L[1 : n_ok + 1] + log_tol) / s[:n_ok]
            if sign > 0:
                a0 = min(a0, -float(np.min(ends)))
            else:
                b0 = max(b0, float(np.min(ends)))
    return a0, b0


class _Maturity:
    """COS pricing of OTM options at one maturity, linear in the CF values."""

    def __init__(self, model, fwd: ForwardSpec, strikes, is_call, params, n_max: int):
        self.fwd, self.strikes, self.is_call = fwd, strikes, is_call
        phi = MODEL_REGISTRY[model].cf
        a0, b0 = _truncation(phi, fwd, params, MODEL_REGISTRY[model].cumulants(fwd, params))
        width = b0 - a0
        # N from where |phi| has decayed: the COS series converges once u_N does
        t = np.geomspace(1.0, 4.0 * n_max * np.pi / width, 200)
        with np.errstate(all="ignore"):
            mag = np.abs(phi(t, fwd, params))
        # some closed forms overflow far out where the true value has underflowed
        mag = np.where(np.isfinite(mag), mag, 0.0)
        big = np.nonzero(~(mag < 1e-13))[0]
        u_cut = t[min(big[-1] + 1, len(t) - 1)] if len(big) else t[0]
        n = int(min(n_max, max(64, 2 ** np.ceil(np.log2(u_cut * width / np.pi + 1)))))
        k = np.arange(n)
        self.u = k * np.pi / width
        x = np.log(fwd.F0 / strikes)
        a, b = x + a0, x + b0  # interval for y = log(S_T / K), per strike
        # put payoff K (1 - e^y)^+ on [a, 0]: V_k = 2/(b-a) K (psi_k(a, 0) - chi_k(a, 0))
        lo = np.minimum(a, 0.0)
        w = self.u[:, None]
        arg_hi = w * (0.0 - a[None, :])
        arg_lo = w * (lo[None, :] - a[None, :])
        chi = (
            np.cos(arg_hi)
            - np.cos(arg_lo) * np.exp(lo)[None, :]
            + w * (np.sin(arg_hi) - np.sin(arg_lo) * np.exp(lo)[None, :])
        ) / (1.0 + w * w)
        psi = np.where(
            k[:, None] == 0,
            0.0 - lo[None, :],
            (np.sin(arg_hi) - np.sin(arg_lo)) / np.where(w == 0, 1.0, w),
        )
        V = 2.0 / width * strikes[None, :] * (psi - chi)
        V[0, :] *= 0.5
        V[:, b <= 0.0] = 0.0  # interval entirely below the kink: handled by parity below
        self.V = V  # (N, n_strikes)
        self.phase = np.exp(-1j * self.u * a0)
        self.deep_itm_put = b <= 0.0
        self.parity = fwd.disc * (fwd.F0 - strikes)

    def price(self, phi) -> np.ndarray:
        put = self.fwd.disc * (np.real(phi * self.phase) @ self.V)
        put = np.where(self.deep_itm_put, -self.parity, put)
        return np.where(self.is_call, put + self.parity, put)

    def gradient(self, dphi) -> np.ndarray:
        """``d price / d theta``, shape ``(n_params, n_strikes)``."""
        g = self.fwd.disc * (np.real(dphi * self.phase[None, :]) @ self.V)
        return np.where(self.deep_itm_put[None, :], 0.0, g)


def calibrate(
    model: str,
    quotes: MarketQuotes,
    initial,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    fixed: tuple[str, ...] | list[str] = (),
    gradient: str = "analytic",
    n_max: int = 4096,
    max_rounds: int = 4,
    max_nfev: int = 200,
    tol: float = 1e-12,
) -> CalibrationFit:
    """Fit ``model`` to ``quotes`` starting from the parameter dataclass ``initial``.

    Parameters
    ----------
    model : str
        Any ``MODEL_REGISTRY`` key whose parameters are plain floats.
    quotes : MarketQuotes
        Implied vols or prices, any mix of strikes and maturities.
    initial :
        Starting parameters (the model's parameter dataclass).
    bounds : dict, optional
        ``{name: (lo, hi)}``; overrides :data:`DEFAULT_BOUNDS` for the model.
        Parameters without bounds are unbounded.
    fixed : sequence of str
        Parameter names held at their ``initial`` values.
    gradient : {"analytic", "fd"}
        CF gradient source. ``"analytic"`` falls back to differences of the CF
        for models without closed-form gradients.
    n_max : int
        Cap on COS terms per maturity.
    max_rounds : int
        Grid rebuilds (the grid is fixed within a round).
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"calibrate: unknown model {model!r}")
    if gradient not in ("analytic", "fd"):
        raise ValueError("calibrate: gradient must be 'analytic' or 'fd'")
    names = param_names(model)
    x_all = params_to_vector(model, initial)
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(
            f"calibrate: unknown fixed parameters {sorted(unknown)}; model has {names}"
        )
    free = [j for j, n in enumerate(names) if n not in fixed]
    if not free:
        raise ValueError("calibrate: every parameter is fixed")
    bnd = dict(DEFAULT_BOUNDS.get(model, {}))
    bnd.update(bounds or {})
    lo = np.array([bnd.get(names[j], (-np.inf, np.inf))[0] for j in free])
    hi = np.array([bnd.get(names[j], (-np.inf, np.inf))[1] for j in free])
    use_analytic = gradient == "analytic" and model in ANALYTIC_GRADIENT_MODELS

    # ---- market side
    T_all = np.asarray(quotes.maturities, dtype=float)
    K_all = np.asarray(quotes.strikes, dtype=float)
    F_all = quotes.S0 * np.exp((quotes.r - quotes.q) * T_all)
    D_all = np.exp(-quotes.r * T_all)
    if quotes.ivs is not None:
        iv_mkt = np.asarray(quotes.ivs, dtype=float)
    else:
        iv_mkt = implied_vol_lets_be_rational(
            np.asarray(quotes.prices, dtype=float),
            F_all,
            K_all,
            T_all,
            disc=D_all,
            cp=np.asarray(quotes.cp),
        )
    if not np.all(np.isfinite(iv_mkt)):
        raise ValueError("calibrate: some quotes have no implied vol (outside no-arbitrage bounds)")
    is_call = K_all >= F_all
    cp_otm = np.where(is_call, 1, -1)
    p_mkt = black_price(F_all, K_all, T_all, iv_mkt, disc=D_all, cp=cp_otm)
    d1 = (np.log(F_all / K_all) + 0.5 * iv_mkt**2 * T_all) / (iv_mkt * np.sqrt(T_all))
    vega = np.maximum(D_all * F_all * norm.pdf(d1) * np.sqrt(T_all), 1e-10 * D_all * F_all)
    w = np.ones_like(T_all) if quotes.weights is None else np.asarray(quotes.weights, dtype=float)
    scale = np.sqrt(w) / vega
    groups = [(T, np.nonzero(T_all == T)[0]) for T in np.unique(T_all)]

    def build(params):
        return [
            (
                idx,
                _Maturity(
                    model,
                    ForwardSpec(quotes.S0, quotes.r, quotes.q, float(T)),
                    K_all[idx],
                    is_call[idx],
                    params,
                    n_max,
                ),
            )
            for T, idx in groups
        ]

    def full(z):
        x = x_all.copy()
        x[free] = z
        return x

    counts = {"nfev": 0, "njev": 0}
    history: list[float] = []
    # least_squares needs a start strictly inside the bounds
    z = np.clip(x_all[free], lo, hi)
    lo_f, hi_f = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 0.0)
    z = np.where(np.isfinite(lo) & (z <= lo), lo_f + 1e-9 * np.maximum(1.0, abs(lo_f)), z)
    z = np.where(np.isfinite(hi) & (z >= hi), hi_f - 1e-9 * np.maximum(1.0, abs(hi_f)), z)
    res = None
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        grid = build(params_from_vector(model, full(z), initial))
        cache: dict = {}

        def evaluate(zz):
            key = zz.tobytes()
            if key not in cache:
                cache.clear()
                try:
                    p = params_from_vector(model, full(zz), initial)
                    prices = np.empty_like(T_all)
                    jac = np.empty((len(T_all), len(free)))
                    for idx, mat in grid:
                        phi, dphi = cf_and_gradient(model, mat.u, mat.fwd, p, analytic=use_analytic)
                        prices[idx] = mat.price(phi)
                        jac[idx] = mat.gradient(dphi[free]).T
                    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(jac)):
                        raise ValueError("non-finite model prices")
                    cache[key] = (prices, jac)
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    cache[key] = None
            return cache[key]

        def fun(zz):
            counts["nfev"] += 1
            out = evaluate(zz)
            if out is None:  # infeasible parameters: a large residual rejects the step
                return np.full_like(T_all, 1e3)
            return scale * (out[0] - p_mkt)

        def jac(zz):
            counts["njev"] += 1
            out = evaluate(zz)
            if out is None:
                return np.zeros((len(T_all), len(free)))
            return scale[:, None] * out[1]

        res = least_squares(
            fun,
            z,
            jac=jac,
            bounds=(lo, hi),
            method="trf",
            x_scale="jac",
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_nfev,
        )
        z = res.x
        history.append(float(res.cost))
        # The grid was sized at the start of the round; rebuild it at the
        # solution and stop once that no longer changes the prices.
        out = evaluate(z)
        p_star = params_from_vector(model, full(z), initial)
        fresh = np.empty_like(T_all)
        for idx, mat in build(p_star):
            fresh[idx] = mat.price(MODEL_REGISTRY[model].cf(mat.u, mat.fwd, p_star))
        if out is not None and np.max(np.abs(out[0] - fresh) * scale) < 1e-10:
            break

    assert res is not None
    p_star = params_from_vector(model, full(z), initial)
    final = np.empty_like(T_all)
    for idx, mat in build(p_star):
        final[idx] = mat.price(MODEL_REGISTRY[model].cf(mat.u, mat.fwd, p_star))
    iv_model = implied_vol_lets_be_rational(final, F_all, K_all, T_all, disc=D_all, cp=cp_otm)
    iv_res = iv_model - iv_mkt
    return CalibrationFit(
        model=model,
        params=p_star,
        values={n: float(v) for n, v in zip(names, full(z))},
        iv_residuals=iv_res,
        rmse_iv=float(np.sqrt(np.nanmean(iv_res**2))),
        success=bool(res.success),
        message=str(res.message),
        nfev=counts["nfev"],
        njev=counts["njev"],
        rounds=rounds,
        gradient="analytic" if use_analytic else "fd",
        cost_history=history,
    )


__all__ = ["DEFAULT_BOUNDS", "CalibrationFit", "MarketQuotes", "calibrate"]
