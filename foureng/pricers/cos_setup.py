"""Strike-independent COS smile setup.

:class:`COSSmileSetup` evaluates the characteristic function **once** and
caches the density coefficients A_k for reuse across any number of strikes,
call/put flags, or parameter-sensitivity queries.

Motivation
----------
Pricing a full strike strip with the standard ``cos_prices`` call rebuilds
the frequency grid and evaluates the CF once per call.  ``COSSmileSetup``
separates setup (CF evaluation + phase shift) from pricing (payoff
coefficient dot product), letting callers amortise the CF cost across
hundreds of strikes.

Usage
-----
    setup = COSSmileSetup.build("heston", fwd, params)
    calls = setup.price(strikes, cp=1)   # vectorised over all strikes
    puts  = setup.price(strikes, cp=-1)

Both calls and puts produced from the same cached density coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..models.registry import MODEL_REGISTRY
from ..utils.grids import COSGrid
from ..utils.spectral_filters import COSFilterSpec, cos_filter_weights
from .cos import (
    COSResult,
    _call_payoff_coeffs,
    _put_payoff_coeffs,
    cos_auto_grid,
    recommended_cos_policy,
)


@dataclass(frozen=True)
class COSSmileSetup:
    """Cached COS density-coefficient object for a single model/fwd/params triplet.

    Attributes
    ----------
    model : str | None
        Registry key used to build the grid policy and retrieve cumulants.
    fwd : ForwardSpec
        Forward specification (spot, rates, maturity) at which the CF was
        evaluated.  A different maturity invalidates this setup.
    grid : COSGrid
        Truncation interval and cosine-term count.
    density_coeffs : np.ndarray
        Density COS coefficients A_k of shape (N,), where N = grid.N.
        These are the real parts of φ(kπ/(b-a)) * exp(-ikπ*a/(b-a)) with
        the k=0 term halved.
    """

    model: Optional[str]
    fwd: ForwardSpec
    grid: COSGrid
    density_coeffs: np.ndarray

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model: str,
        fwd: ForwardSpec,
        params,
        *,
        grid: Optional[COSGrid] = None,
        filter_spec: Optional[COSFilterSpec] = None,
    ) -> "COSSmileSetup":
        """Build a :class:`COSSmileSetup` by evaluating the CF once.

        Parameters
        ----------
        model :
            Model registry key, e.g. ``"heston"``.
        fwd :
            :class:`~foureng.models.base.ForwardSpec`.
        params :
            Model parameter dataclass.
        grid :
            Optional pre-built :class:`~foureng.utils.grids.COSGrid`.
            If ``None``, a grid is built from model cumulants using the
            recommended policy.
        filter_spec :
            Optional spectral filter applied to the density coefficients.
        """
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"COSSmileSetup.build: unknown model {model!r}; "
                f"choose from {sorted(MODEL_REGISTRY)}"
            )
        entry = MODEL_REGISTRY[model]

        if grid is None:
            cums = entry.cumulants(fwd, params)
            policy = recommended_cos_policy(model, params)
            from ..pricers.cos import cos_adaptive_decision
            decision = cos_adaptive_decision(cums, model=model, params=params, policy=policy)
            grid = decision.grid

        a, b, N = grid.a, grid.b, grid.N
        k = np.arange(N)
        omega = k * np.pi / (b - a)

        phi = lambda u: entry.cf(u, fwd, params)
        phi_vals = phi(omega)

        center = float(getattr(grid, "center", 0.0))
        if center != 0.0:
            phi_vals = phi_vals * np.exp(-1j * omega * center)

        A = np.real(phi_vals * np.exp(-1j * omega * a))
        A[0] *= 0.5

        if filter_spec is not None and filter_spec.name != "none":
            sigma = cos_filter_weights(N, filter_spec)
            A = A * sigma

        return cls(model=model, fwd=fwd, grid=grid, density_coeffs=A)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        strikes,
        cp: int = 1,
        *,
        payoff_mode: str = "put_parity",
    ) -> np.ndarray:
        """Price a strip of strikes using cached density coefficients.

        Parameters
        ----------
        strikes :
            1-D array-like of strike prices.
        cp :
            ``+1`` for calls, ``-1`` for puts.
        payoff_mode :
            ``"put_parity"`` (default), ``"call_direct"``, or ``"auto"``.

        Returns
        -------
        np.ndarray
            Prices at each strike, same shape as ``strikes``.
        """
        grid = self.grid
        a, b, N = grid.a, grid.b, grid.N
        A = self.density_coeffs
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

        center = float(getattr(grid, "center", 0.0))
        shifted_F0 = self.fwd.F0 * np.exp(center)

        if payoff_mode == "call_direct":
            V = _call_payoff_coeffs(a, b, N, strikes, shifted_F0)
            calls = self.fwd.disc * (A[:, None] * V).sum(axis=0)
        else:
            V_put = _put_payoff_coeffs(a, b, N, strikes, shifted_F0)
            puts = self.fwd.disc * (A[:, None] * V_put).sum(axis=0)
            calls = puts + self.fwd.disc * (self.fwd.F0 - strikes)

        if cp == 1:
            result = calls
        else:
            # put from parity: P = C - disc*(F0 - K)
            result = calls - self.fwd.disc * (self.fwd.F0 - strikes)

        return np.asarray(result, dtype=float)

    def price_call(self, strikes) -> np.ndarray:
        """Convenience: price calls via put+parity."""
        return self.price(np.asarray(strikes, dtype=float), cp=1)

    def price_put(self, strikes) -> np.ndarray:
        """Convenience: price puts via put+parity."""
        return self.price(np.asarray(strikes, dtype=float), cp=-1)

    def maturity(self) -> float:
        return self.fwd.T

    def validate_maturity(self, expected_T: float, tol: float = 1e-8) -> None:
        """Raise if this setup was built for a different maturity."""
        if abs(self.fwd.T - expected_T) > tol:
            raise ValueError(
                f"COSSmileSetup maturity mismatch: setup was built for T={self.fwd.T}, "
                f"but caller expects T={expected_T}."
            )
