from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FFTGrid:
    """Carr-Madan FFT grid. Nyquist binds eta and lam: eta*lam = 2*pi/N."""
    N: int
    eta: float
    alpha: float

    def u(self) -> np.ndarray:
        return np.arange(self.N) * self.eta

    @property
    def lam(self) -> float:
        return 2.0 * np.pi / (self.N * self.eta)

    @property
    def b(self) -> float:
        """Half-width of the log-strike grid, centered at k0."""
        return self.N * self.lam / 2.0

    def k_grid(self, k0: float = 0.0) -> np.ndarray:
        """Log-strike grid centered at k0. Spacing = lam."""
        return (k0 - self.b) + np.arange(self.N) * self.lam


@dataclass(frozen=True)
class FRFTGrid:
    """Fractional FFT grid — eta (freq step) and lam (log-strike step) independent."""
    N: int
    eta: float
    lam: float
    alpha: float

    def u(self) -> np.ndarray:
        return np.arange(self.N) * self.eta

    @property
    def zeta(self) -> float:
        """FRFT fraction: zeta = eta * lam / (2*pi)."""
        return self.eta * self.lam / (2.0 * np.pi)

    def k_grid(self, k0: float = 0.0) -> np.ndarray:
        return (k0 - self.N * self.lam / 2.0) + np.arange(self.N) * self.lam


@dataclass(frozen=True)
class COSGrid:
    """COS truncation interval [a,b] and number of cosine terms N."""
    N: int
    a: float
    b: float

    def u(self) -> np.ndarray:
        return np.arange(self.N) * np.pi / (self.b - self.a)
