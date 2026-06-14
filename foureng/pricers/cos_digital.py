"""COS-method pricer for digital (binary) options.

Implements Fourier-cosine payoff coefficients for cash-or-nothing and
asset-or-nothing digital options, as an extension of the Fang-Oosterlee
(2008) COS method.

For the COS expansion on [a, b] with variable x = log(S_T / F0):

    Cash-or-nothing call payoff: 1 if x > k_s, 0 otherwise
    Cash-or-nothing put payoff:  1 if x < k_s, 0 otherwise
    Asset-or-nothing call payoff: F0*exp(x) if x > k_s, 0 otherwise
    Asset-or-nothing put payoff:  F0*exp(x) if x < k_s, 0 otherwise

where k_s = log(K / F0).
"""

from __future__ import annotations

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import COSGrid

# ---------------------------------------------------------------------------
# Payoff coefficient helpers
# ---------------------------------------------------------------------------


def _psi_integral(a: float, b: float, N: int, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Integral psi_k = integral_{c}^{d} cos(k*pi*(x-a)/(b-a)) dx.

    For k=0: psi_0 = d - c
    For k>=1: psi_k = (b-a)/(k*pi) * [sin(k*pi*(d-a)/(b-a)) - sin(k*pi*(c-a)/(b-a))]

    Parameters
    ----------
    a, b  : COS truncation bounds
    N     : number of terms
    c, d  : lower and upper integration limits (shape: n_strikes,)

    Returns
    -------
    psi : shape (N, n_strikes)
    """
    k = np.arange(N)
    ba = b - a
    omega = k * np.pi / ba  # (N,)

    da = d[None, :] - a  # (1, n_strikes)
    ca = c[None, :] - a  # (1, n_strikes)

    sin_d = np.sin(omega[:, None] * da)  # (N, n_strikes)
    sin_c = np.sin(omega[:, None] * ca)

    psi = np.empty((N, len(c)))
    psi[0, :] = d - c
    with np.errstate(divide="ignore", invalid="ignore"):
        psi[1:, :] = (sin_d[1:, :] - sin_c[1:, :]) / omega[1:, None]
    psi = psi * (2.0 / ba)
    return psi


def _chi_integral(a: float, b: float, N: int, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Integral chi_k = integral_{c}^{d} exp(x)*cos(k*pi*(x-a)/(b-a)) dx * (2/(b-a)).

    Using integration by parts twice:
        integral exp(x) cos(omega*x) dx = exp(x)*(cos(omega*x) + omega*sin(omega*x))/(1+omega^2) + const

    where omega = k*pi/(b-a).

    Parameters
    ----------
    a, b  : COS truncation bounds
    N     : number of terms
    c, d  : lower and upper integration limits (shape: n_strikes,)

    Returns
    -------
    chi : shape (N, n_strikes)
    """
    k = np.arange(N)
    ba = b - a
    omega = k * np.pi / ba  # (N,)

    ec = np.exp(c)  # (n_strikes,)
    ed = np.exp(d)  # (n_strikes,)

    da = d[None, :] - a  # (1, n_strikes)
    ca = c[None, :] - a  # (1, n_strikes)

    cos_d = np.cos(omega[:, None] * da)
    sin_d = np.sin(omega[:, None] * da)
    cos_c = np.cos(omega[:, None] * ca)
    sin_c = np.sin(omega[:, None] * ca)

    denom = 1.0 + omega[:, None] ** 2  # (N, 1)

    chi = (
        (cos_d * ed[None, :] + omega[:, None] * sin_d * ed[None, :])
        - (cos_c * ec[None, :] + omega[:, None] * sin_c * ec[None, :])
    ) / denom
    chi = chi * (2.0 / ba)
    return chi


def _cash_or_nothing_call_coeffs(a: float, b: float, N: int, k_strike: np.ndarray) -> np.ndarray:
    """COS payoff coefficients for cash-or-nothing call: pays 1 if x > k_strike.

    Integration is over [max(k_strike,a), b]:
        G_k = psi_k integrated over [c, b]   where c = clip(k_strike, a, b)
    """
    c = np.clip(k_strike, a, b)
    d = np.full_like(c, b)
    return _psi_integral(a, b, N, c, d)


def _cash_or_nothing_put_coeffs(a: float, b: float, N: int, k_strike: np.ndarray) -> np.ndarray:
    """COS payoff coefficients for cash-or-nothing put: pays 1 if x < k_strike.

    Integration is over [a, min(k_strike,b)]:
        G_k = psi_k integrated over [a, d]   where d = clip(k_strike, a, b)
    """
    c = np.full_like(k_strike, a)
    d = np.clip(k_strike, a, b)
    return _psi_integral(a, b, N, c, d)


def _asset_or_nothing_call_coeffs(
    a: float, b: float, N: int, k_strike: np.ndarray, F0: float
) -> np.ndarray:
    """COS payoff coefficients for asset-or-nothing call: pays F0*exp(x) if x > k_strike.

    Integration is over [max(k_strike,a), b]:
        G_k = F0 * chi_k integrated over [c, b]   where c = clip(k_strike, a, b)
    """
    c = np.clip(k_strike, a, b)
    d = np.full_like(c, b)
    return F0 * _chi_integral(a, b, N, c, d)


def _asset_or_nothing_put_coeffs(
    a: float, b: float, N: int, k_strike: np.ndarray, F0: float
) -> np.ndarray:
    """COS payoff coefficients for asset-or-nothing put: pays F0*exp(x) if x < k_strike.

    Integration is over [a, min(k_strike,b)]:
        G_k = F0 * chi_k integrated over [a, d]   where d = clip(k_strike, a, b)
    """
    c = np.full_like(k_strike, a)
    d = np.clip(k_strike, a, b)
    return F0 * _chi_integral(a, b, N, c, d)


# ---------------------------------------------------------------------------
# Public pricing function
# ---------------------------------------------------------------------------


def cos_digital_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
    digital_type: str = "cash_or_nothing",
    cp: int = 1,
    cash: float = 1.0,
) -> np.ndarray:
    """COS-method prices for digital (binary) options.

    Parameters
    ----------
    phi          : characteristic function of log-return X_T = log(S_T/F0)
    fwd          : forward specification (spot, rate, div, maturity)
    strikes      : array of strike prices
    grid         : COSGrid with truncation interval [a, b] and N terms
    digital_type : "cash_or_nothing" or "asset_or_nothing"
    cp           : +1 call, -1 put
    cash         : cash amount for cash-or-nothing contracts (default 1.0)

    Returns
    -------
    np.ndarray
        Digital option prices, one per strike.
    """
    if digital_type not in {"cash_or_nothing", "asset_or_nothing"}:
        raise ValueError(
            f"cos_digital_prices: digital_type must be 'cash_or_nothing' or "
            f"'asset_or_nothing'; got {digital_type!r}"
        )
    if cp not in (1, -1):
        raise ValueError(f"cos_digital_prices: cp must be +1 or -1, got {cp}")

    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    center = float(getattr(grid, "center", 0.0))
    shifted_F0 = fwd.F0 * np.exp(center)

    k = np.arange(N)
    omega = k * np.pi / (b - a)

    # Characteristic function samples
    phi_vals = phi(omega)
    if center != 0.0:
        phi_vals = phi_vals * np.exp(-1j * omega * center)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5

    # Log-strike array
    k_strike = np.log(strikes / shifted_F0)

    if digital_type == "cash_or_nothing":
        if cp == 1:
            V = _cash_or_nothing_call_coeffs(a, b, N, k_strike)
        else:
            V = _cash_or_nothing_put_coeffs(a, b, N, k_strike)
        prices = cash * fwd.disc * (A[:, None] * V).sum(axis=0)
    else:
        # asset_or_nothing
        if cp == 1:
            V = _asset_or_nothing_call_coeffs(a, b, N, k_strike, shifted_F0)
        else:
            V = _asset_or_nothing_put_coeffs(a, b, N, k_strike, shifted_F0)
        prices = fwd.disc * (A[:, None] * V).sum(axis=0)

    return price


def cos_digital_price_strip(
    model: str,
    fwd: ForwardSpec,
    params,
    strikes: np.ndarray,
    maturity: float,
    cp: int = 1,
    payoff_type: Literal["cash_or_nothing", "asset_or_nothing"] = "cash_or_nothing",
    cash_amount: float = 1.0,
    *,
    grid=None,
    N: int = 256,
    L: float = 12.0,
) -> np.ndarray:
    """Price a strip of digital options at different strikes.

    Builds the grid and evaluates the CF once, then loops over strikes.
    """
    strikes = np.asarray(strikes, dtype=float)
    if model not in MODEL_REGISTRY:
        raise ValueError(f"cos_digital_price_strip: unknown model {model!r}")

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=maturity)
    if grid is None:
        cums = MODEL_REGISTRY[model].cumulants(fwd_T, params)
        grid = cos_auto_grid(cums, N=N, L=L)

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        product = DigitalOption(
            strike=float(K),
            maturity=maturity,
            cp=cp,
            payoff_type=payoff_type,
            cash_amount=cash_amount,
        )
        prices[i] = cos_digital_price(model, fwd, params, product, grid=grid)
    return prices


# ---------------------------------------------------------------------------
# Low-level CF-based batch pricer (used by Sprint 3 pipeline dispatch and tests)
# ---------------------------------------------------------------------------


def cos_digital_prices(
    phi,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid,
    digital_type: str = "cash_or_nothing",
    cp: int = 1,
    cash: float = 1.0,
) -> np.ndarray:
    """Low-level COS digital pricer operating directly on a CF and COSGrid.

    Parameters
    ----------
    phi          : callable(omega) → complex array, characteristic function
    fwd          : ForwardSpec
    strikes      : array of strike prices
    grid         : COSGrid with .a, .b, .N attributes
    digital_type : "cash_or_nothing" or "asset_or_nothing"
    cp           : +1 call, -1 put
    cash         : cash amount for cash-or-nothing (default 1.0)
    """
    a, b, N = grid.a, grid.b, grid.N
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    T = fwd.T
    S0, r, q = fwd.S0, fwd.r, fwd.q
    disc = np.exp(-r * T)
    F_T = S0 * np.exp((r - q) * T)

    omega = np.arange(N) * np.pi / (b - a)
    phi_vals = phi(omega)
    A = np.real(phi_vals * np.exp(-1j * omega * a))
    A[0] *= 0.5

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        k_star = np.log(K / F_T)
        k_star = max(a, min(b, k_star))
        if cp == 1:
            c1, c2 = k_star, b
        else:
            c1, c2 = a, k_star
        if digital_type == "cash_or_nothing":
            V_k = _chi(omega, c1, c2, a, b) * cash
        else:
            V_k = _psi(omega, c1, c2, a, b, F_T)
        prices[i] = disc * float(np.dot(A, V_k))
    return prices
