"""QuantoOption product dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantoOption:
    """Quanto option: foreign underlying, domestic payout.

    Parameters
    ----------
    S : float
        Spot price of the foreign asset in its own currency.
    K : float
        Strike in foreign currency.
    T : float
        Time to maturity (years).
    r_dom : float
        Domestic risk-free rate.
    r_for : float
        Foreign risk-free rate.
    q_for : float
        Continuous dividend yield of the foreign asset.
    rho : float
        Correlation between asset and FX log-returns, in [-1, 1].
    sigma_S : float
        Volatility of the foreign asset, >= 0.
    sigma_X : float
        Volatility of the FX rate, >= 0.
    cp : int
        +1 for call, -1 for put.
    """

    S: float
    K: float
    T: float
    r_dom: float
    r_for: float
    q_for: float
    rho: float
    sigma_S: float
    sigma_X: float
    cp: int = 1

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError(f"S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.sigma_S < 0:
            raise ValueError(f"sigma_S must be non-negative, got {self.sigma_S}")
        if self.sigma_X < 0:
            raise ValueError(f"sigma_X must be non-negative, got {self.sigma_X}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.cp not in (1, -1):
            raise ValueError(f"cp must be +1 or -1, got {self.cp}")
