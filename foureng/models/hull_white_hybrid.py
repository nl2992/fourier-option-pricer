"""Equity + one-factor Hull-White stochastic-rate hybrid (independent rates).

Under a one-factor Hull-White short rate independent of the equity driver,
European pricing moves to the T-forward measure: the discount factor becomes
the zero-coupon bond P(0,T) (equal to ``exp(-rT)`` on the flat curve used
throughout this package), and the forward-normalized log return picks up an
independent Gaussian from the bond-price volatility:

    X_T = Y_T + Z,   Z ~ N(-V_P/2, V_P)  independent of Y_T,
    phi(u) = phi_base(u) * exp(-0.5 * V_P * (u^2 + i u)),

where ``V_P = integral_0^T sigma_P(s)^2 ds`` with the Hull-White bond
volatility ``sigma_P(s) = (sigma_r / a) (1 - e^{-a (T - s)})``:

    V_P = (sigma_r^2 / a^2) [ T - 2 (1 - e^{-aT})/a + (1 - e^{-2aT})/(2a) ],
    V_P -> sigma_r^2 T^3 / 3  as  a -> 0.

Because the equity driver and the rate are independent, the equity law is
unchanged by the measure change, so *any* registry model can serve as the
base -- the hybrid CF is the product above and remains a martingale CF
(``phi(-i) = 1``). With a BSM base the hybrid is again lognormal with total
variance ``sigma^2 T + V_P``, which gives an exact closed-form test anchor
(Merton 1973's stochastic-rate Black-Scholes).

References
----------
Hull, J. & White, A. (1990). Pricing interest-rate-derivative securities.
*Review of Financial Studies*, 3(4), 573-592.

Merton, R.C. (1973). Theory of rational option pricing. *Bell Journal of
Economics and Management Science*, 4(1), 141-183. (Section 8: pricing with
stochastic interest rates via the bond numeraire.)

Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and
Practice*, 2nd ed., Springer. (Ch. 3, Hull-White bond volatility.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ForwardSpec, ModelSpec


def hw_bond_variance(mean_reversion: float, sigma_r: float, T: float) -> float:
    """Integrated Hull-White bond-price variance V_P over [0, T].

    Uses the exact expression for ``a T`` away from zero and the
    ``sigma_r^2 T^3 / 3`` Ho-Lee limit below ``a T < 1e-4``, where the exact
    form loses all significant digits to cancellation.
    """
    a, s2 = float(mean_reversion), float(sigma_r) ** 2
    if a < 0.0:
        raise ValueError(f"hw_bond_variance: mean_reversion must be >= 0; got {a}")
    if a * T < 1e-4:
        return s2 * T**3 / 3.0
    e1 = -np.expm1(-a * T)  # 1 - exp(-aT)
    e2 = -np.expm1(-2.0 * a * T)  # 1 - exp(-2aT)
    return s2 / (a * a) * (T - 2.0 * e1 / a + e2 / (2.0 * a))


@dataclass(frozen=True)
class HullWhiteHybridParams(ModelSpec):
    """Composite parameters: any base registry model + Hull-White rate factor.

    base_model     : registry key of the equity model (e.g. "bsm", "kou",
                     "heston"); may not itself be "hw_hybrid"
    base_params    : the base model's parameter dataclass
    mean_reversion : Hull-White mean-reversion speed a >= 0 (a = 0 is Ho-Lee)
    sigma_r        : short-rate volatility >= 0 (0 collapses to the base model)
    """

    base_model: str
    base_params: ModelSpec
    mean_reversion: float
    sigma_r: float

    def __init__(self, base_model, base_params, mean_reversion, sigma_r):
        object.__setattr__(self, "name", "hw_hybrid")
        object.__setattr__(self, "base_model", str(base_model))
        object.__setattr__(self, "base_params", base_params)
        object.__setattr__(self, "mean_reversion", float(mean_reversion))
        object.__setattr__(self, "sigma_r", float(sigma_r))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.base_model == "hw_hybrid":
            raise ValueError("HullWhiteHybridParams: base_model cannot be 'hw_hybrid'")
        if not isinstance(self.base_params, ModelSpec):
            raise TypeError(
                "HullWhiteHybridParams: base_params must be a model parameter "
                f"dataclass (ModelSpec); got {type(self.base_params).__name__!r}"
            )
        if not (np.isfinite(self.mean_reversion) and self.mean_reversion >= 0.0):
            raise ValueError(
                f"HullWhiteHybridParams: mean_reversion must be >= 0; got {self.mean_reversion}"
            )
        if not (np.isfinite(self.sigma_r) and self.sigma_r >= 0.0):
            raise ValueError(f"HullWhiteHybridParams: sigma_r must be >= 0; got {self.sigma_r}")
        # Registry membership is validated lazily in the CF/cumulant functions
        # to avoid a circular import with the registry module.


def _base_entry(p: HullWhiteHybridParams):
    from .registry import MODEL_REGISTRY

    if p.base_model not in MODEL_REGISTRY:
        raise ValueError(
            f"HullWhiteHybridParams: unknown base_model {p.base_model!r}; "
            f"choose from {sorted(k for k in MODEL_REGISTRY if k != 'hw_hybrid')}"
        )
    return MODEL_REGISTRY[p.base_model]


def hw_hybrid_cf(u: np.ndarray, fwd: ForwardSpec, p: HullWhiteHybridParams) -> np.ndarray:
    """CF of X_T = log(S_T/F0) under the equity + Hull-White hybrid."""
    entry = _base_entry(p)
    u_c = np.asarray(u, dtype=np.complex128)
    base = np.asarray(entry.cf(u_c, fwd, p.base_params), dtype=np.complex128)
    v_p = hw_bond_variance(p.mean_reversion, p.sigma_r, fwd.T)
    if v_p == 0.0:
        return base
    return base * np.exp(-0.5 * v_p * (u_c * u_c + 1j * u_c))


def hw_hybrid_cumulants(fwd: ForwardSpec, p: HullWhiteHybridParams) -> tuple[float, float, float]:
    """Cumulants: the independent Gaussian adds -V_P/2 to c1 and V_P to c2."""
    entry = _base_entry(p)
    c1, c2, c4 = entry.cumulants(fwd, p.base_params)
    v_p = hw_bond_variance(p.mean_reversion, p.sigma_r, fwd.T)
    return float(c1 - 0.5 * v_p), float(c2 + v_p), float(c4)
