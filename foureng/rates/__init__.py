"""Affine short-rate models for stochastic discounting.

This subpackage adds a research-grade short-rate CF family (Vasicek, CIR,
Hull-White) to the library.  The primary use case is stochastic discounting
of European claims: any equity-model CF ``phi_X(u)`` can be combined with a
short-rate model's CF ``phi_R(u) = E^Q[exp(-i u ∫_0^T r_s ds)]`` under the
independence assumption to yield a hybrid pricer.

For direct callable use, three closed-form quantities are exposed per model:

* ``discount_bond(params, T)``     -> zero-coupon bond price P(0, T)
* ``integrated_rate_cf(u, params, T)`` -> E^Q[exp(i u ∫_0^T r_s ds)]
* ``integrated_rate_cumulants(params, T)`` -> (c1, c2) of ∫_0^T r_s ds

All formulas follow Brigo & Mercurio (2006), *Interest Rate Models — Theory
and Practice*, 2nd ed., Springer.

LevFin bridge
-------------
For fixed-rate high-yield bonds, Federico Etchelecu's LevFin make-whole
calculator now uses a Hull-White trinomial tree plus analytic live-curve
zero-coupon prices for the make-whole strike.  This subpackage supplies the
curve-fitted short-rate machinery behind that reference implementation and
keeps the risk-neutral discounting identities explicit.
"""

from __future__ import annotations

from .cir import (
    CIRParams,
    cir_discount_bond,
    cir_integrated_rate_cf,
    cir_integrated_rate_cumulants,
)
from .hull_white import (
    HullWhiteParams,
    hull_white_discount_bond,
    hull_white_integrated_rate_cf,
    hull_white_integrated_rate_cumulants,
)
from .vasicek import (
    VasicekParams,
    vasicek_discount_bond,
    vasicek_integrated_rate_cf,
    vasicek_integrated_rate_cumulants,
)

__all__ = [
    "VasicekParams",
    "vasicek_discount_bond",
    "vasicek_integrated_rate_cf",
    "vasicek_integrated_rate_cumulants",
    "CIRParams",
    "cir_discount_bond",
    "cir_integrated_rate_cf",
    "cir_integrated_rate_cumulants",
    "HullWhiteParams",
    "hull_white_discount_bond",
    "hull_white_integrated_rate_cf",
    "hull_white_integrated_rate_cumulants",
]
