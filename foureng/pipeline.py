"""Unified pricing pipeline that dispatches a (model, method, strikes) request to the right engine.

The main entry points are:

* :func:`price_strip` — price a vector of strikes for a European option.
* :func:`price` — price a single :class:`~foureng.products.ProductSpec` object.

The internal phase helpers (phase2_carr_madan, phase3_frft, phase4_cos) are thin
wrappers used by the demo notebooks; end users will rarely call them directly.

The improved COS path (``method="cos_improved"``) builds the truncation interval
and cosine-term count adaptively from model cumulants, then routes wide-interval
cases to Lewis or Carr-Madan rather than forcing COS into an unfavorable geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analytics.bsm_asian import bsm_discrete_geometric_asian
from .analytics.bsm_barrier import bsm_barrier_price
from .analytics.bsm_exotics import bsm_forward_start, bsm_lookback_floating, margrabe_exchange
from .mc.engine import MCSpec, mc_price
from .models.base import CharFunc, ForwardSpec
from .models.registry import MODEL_REGISTRY
from .pricers.carr_madan import carr_madan_price_at_strikes
from .pricers.conv import conv_price_at_strikes
from .pricers.cos import (
    cos_adaptive_decision,
    cos_auto_grid,
    cos_prices,
    recommended_cos_policy,
)
from .pricers.cos_bermudan import cos_bermudan_price
from .pricers.filtered_cos import filtered_cos_prices
from .pricers.frft import frft_price_at_strikes
from .pricers.lattice import LatticeGrid, bsm_lattice_price, bsm_lattice_price_at_strikes
from .pricers.lewis import lewis_call_prices
from .pricers.mellin import MELLIN_SUPPORTED_MODELS, mellin_price_at_strikes
from .pricers.pde_fd import PDEGrid, bsm_pde_fd_price, bsm_pde_fd_price_at_strikes
from .pricers.proj import proj_european_price_at_strikes
from .pricers.sabr import sabr_hagan_price_at_strikes
from .utils.grids import CONVGrid, COSGrid, COSGridPolicy, FFTGrid, FRFTGrid
from .utils.spectral_filters import COSFilterSpec


@dataclass(frozen=True)
class PhaseOutputs:
    strikes: np.ndarray
    prices: np.ndarray


def phase2_carr_madan(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: FFTGrid
) -> PhaseOutputs:
    prices = carr_madan_price_at_strikes(phi, fwd, grid, strikes)
    return PhaseOutputs(strikes=np.asarray(strikes, float), prices=prices)


def phase3_frft(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: FRFTGrid
) -> PhaseOutputs:
    prices = frft_price_at_strikes(phi, fwd, grid, strikes)
    return PhaseOutputs(strikes=np.asarray(strikes, float), prices=prices)


def phase4_cos(phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: COSGrid) -> PhaseOutputs:
    res = cos_prices(phi, fwd, strikes, grid)
    return PhaseOutputs(strikes=res.strikes, prices=res.call_prices)


# ---------------------------------------------------------------------------
# Unified strip pricing  -  one call that the notebook / scoreboard goes
# through, with a ``backend=`` knob that switches the characteristic function
# between in-house analytic and PyFENG, and a ``method="pyfeng_fft"`` option
# that delegates to PyFENG's own pricer entirely.
# ---------------------------------------------------------------------------

_DIRECT_CALL_FRIENDLY_MODELS = {"heston", "ousv", "nig"}


def _cf_for(model: str, fwd: ForwardSpec, params):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODEL_REGISTRY)}")
    return lambda u: MODEL_REGISTRY[model].cf(u, fwd, params)


def _improved_cos_payoff_mode(model: str, grid: COSGrid) -> str:
    """Choose the more stable coefficient side for the improved COS path.

    On narrow centered intervals, Gaussian-like models are typically accurate
    enough on the direct call coefficients that we should avoid parity-based
    cancellation on ITM calls. For heavier-tailed models we stay with the more
    conservative mixed/put-side logic.
    """
    if model in _DIRECT_CALL_FRIENDLY_MODELS and grid.width <= 8.0:
        return "call_direct"
    return "auto"


def _pyfeng_fft_price(model: str, strikes, fwd: ForwardSpec, params, cp: int):
    """Call PyFENG's FFT pricer directly.

    Supported models (PyFENG ships native ``*Fft`` classes for these):
    ``bsm``, ``heston``, ``ousv``, ``vg``, ``cgmy``, ``nig``,
    ``sv32``, ``rough_heston``.

    Raises :class:`ValueError` for models where ``MODEL_REGISTRY[model].pyfeng_fft``
    is ``False``: ``kou``, ``bates``, ``heston_kou``, ``heston_cgmy``,
    ``garch_wmw2012``, ``merton_jd``, ``meixner``, ``bilateral_gamma``,
    ``generalized_hyperbolic``, ``fmls``.
    """
    if model not in MODEL_REGISTRY or not MODEL_REGISTRY[model].pyfeng_fft:
        raise ValueError(
            f"method='pyfeng_fft' is not supported for model={model!r} "
            " -  PyFENG has no FFT pricer for this model. Use "
            "'cos' / 'frft' / 'carr_madan'."
        )
    try:
        import pyfeng as pf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "method='pyfeng_fft' requires pyfeng; install with `pip install pyfeng`."
        ) from exc

    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    if model == "bsm":
        m = pf.BsmFft(sigma=params.sigma, intr=fwd.r, divr=fwd.q)
    elif model == "heston":
        # PyFENG's ``sigma`` kwarg is the *instantaneous variance* (v0),
        # not its square root. Verified in tests/test_pyfeng_cf_wrappers.py.
        m = pf.HestonFft(
            sigma=params.v0,
            vov=params.nu,
            rho=params.rho,
            mr=params.kappa,
            theta=params.theta,
            intr=fwd.r,
            divr=fwd.q,
        )
    elif model == "ousv":
        # Mirror the OUSV CF wrapper's kwarg translation (see
        # ``models/ousv.py``): our ``sigma0`` is PyFENG's ``sigma``,
        # our ``kappa`` is PyFENG's ``mr``, our ``nu`` is PyFENG's ``vov``.
        m = pf.OusvFft(
            sigma=params.sigma0,
            mr=params.kappa,
            theta=params.theta,
            vov=params.nu,
            rho=params.rho,
            intr=fwd.r,
            divr=fwd.q,
        )
    elif model == "vg":
        m = pf.VarGammaFft(
            sigma=params.sigma, nu=params.nu, theta=params.theta, intr=fwd.r, divr=fwd.q
        )
    elif model == "cgmy":
        m = pf.CgmyFft(C=params.C, G=params.G, M=params.M, Y=params.Y, intr=fwd.r, divr=fwd.q)
    elif model == "nig":
        m = pf.ExpNigFft(
            sigma=params.sigma, nu=params.nu, theta=params.theta, intr=fwd.r, divr=fwd.q
        )
    elif model == "sv32":
        m = pf.Sv32Fft(
            sigma=params.v0,
            vov=params.nu,
            mr=params.kappa,
            rho=params.rho,
            theta=params.theta,
            intr=fwd.r,
            divr=fwd.q,
        )
    elif model == "rough_heston":
        # RoughHestonFft lives in pyfeng.sv_fft; pyfeng.ex is broken under
        # newer SciPy (scipy.misc.derivative was removed).
        from pyfeng.sv_fft import RoughHestonFft as _RoughHestonFft  # type: ignore

        m = _RoughHestonFft(
            sigma=params.sigma,
            vov=params.vov,
            mr=params.mr,
            rho=params.rho,
            theta=params.theta,
            alpha=params.alpha,
            intr=fwd.r,
            divr=fwd.q,
        )
    else:
        raise ValueError(f"unknown model {model!r}")
    return np.asarray(m.price(K, spot=fwd.S0, texp=fwd.T, cp=cp), dtype=np.float64)


def price_strip(
    model: str,
    method: str,
    strikes,
    fwd: ForwardSpec,
    params,
    *,
    grid: Any = None,
    cp: int = 1,
) -> np.ndarray:
    """Unified strip pricer used by the scoreboard and demo notebook.

    Parameters
    ----------
    model :
        One of the supported model keys: ``"bsm"``, ``"heston"``, ``"ousv"``,
        ``"vg"``, ``"cgmy"``, ``"nig"``, ``"kou"``, ``"bates"``,
        ``"heston_kou"``, ``"heston_cgmy"``, ``"sv32"``.
    method :
        * ``"cos"``  -  in-house COS (Fang-Oosterlee 2008),
        * ``"cos_improved"``  -  adaptive COS policy with centered intervals,
          coupled N/L selection, and wide-interval fallback,
        * ``"frft"``  -  in-house FRFT (Chourdakis 2004),
        * ``"carr_madan"``  -  in-house Carr-Madan FFT (1999),
        * ``"pyfeng_fft"``  -  PyFENG's own native FFT pricer. Available for:
          BSM, Heston, OUSV, VG, CGMY, NIG, 3/2 SV (``sv32``), Rough Heston.
          Not available for Kou, Bates, Heston-Kou, Heston-CGMY, GARCH,
          Merton JD, Meixner, Bilateral Gamma, GH, or FMLS.
    strikes :
        1-D iterable of strikes.
    fwd, params :
        Forward spec and model-specific parameter dataclass.
    grid :
        Grid object appropriate to ``method``  -  :class:`FFTGrid` for
        ``"carr_madan"``, :class:`FRFTGrid` for ``"frft"``,
        :class:`COSGrid` for ``"cos"``. ``method='cos_improved'`` also accepts
        :class:`COSGridPolicy`. If ``None`` and ``method='cos'``, an auto grid
        is built from the model cumulants with :func:`cos_auto_grid`.
    cp :
        ``+1`` calls, ``-1`` puts (consulted only by ``pyfeng_fft``; the
        in-house pricers return calls and the caller applies parity).

    Returns
    -------
    np.ndarray
        Prices at ``strikes``.
    """
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    if K.size == 0:
        raise ValueError("strikes must be non-empty")
    if method == "pyfeng_fft":
        return _pyfeng_fft_price(model, K, fwd, params, cp=cp)

    if method == "sabr_hagan":
        if model != "sabr":
            raise ValueError("method='sabr_hagan' is currently implemented only for model='sabr'")
        return sabr_hagan_price_at_strikes(fwd, params, K, cp=cp)

    if method == "lattice":
        if model != "bsm":
            raise ValueError("method='lattice' is currently implemented only for model='bsm'")
        lattice_grid = grid if isinstance(grid, LatticeGrid) else LatticeGrid()
        return np.asarray(
            bsm_lattice_price_at_strikes(
                fwd, params, K, cp=cp, exercise="european", grid=lattice_grid
            ),
            dtype=np.float64,
        )

    phi = _cf_for(model, fwd, params)

    if method == "conv":
        conv_grid = grid if isinstance(grid, CONVGrid) else CONVGrid()
        return np.asarray(conv_price_at_strikes(phi, fwd, conv_grid, K, cp=cp), dtype=np.float64)

    if method == "mellin":
        if model not in MELLIN_SUPPORTED_MODELS:
            raise ValueError(
                f"method='mellin' is currently supported for {sorted(MELLIN_SUPPORTED_MODELS)}"
            )
        mellin_grid = grid if isinstance(grid, CONVGrid) else None
        return mellin_price_at_strikes(phi, fwd, K, cp=cp, grid=mellin_grid)

    if method == "proj":
        return proj_european_price_at_strikes(
            phi,
            fwd,
            MODEL_REGISTRY[model].cumulants(fwd, params),
            K,
            cp=cp,
        )

    if method == "pde_fd":
        if model != "bsm":
            raise ValueError("method='pde_fd' is currently implemented only for model='bsm'")
        pde_grid = grid if isinstance(grid, PDEGrid) else PDEGrid()
        return np.asarray(
            bsm_pde_fd_price_at_strikes(fwd, params, K, cp=cp, exercise="european", grid=pde_grid),
            dtype=np.float64,
        )

    if method == "cos":
        if isinstance(grid, COSGridPolicy):
            decision = cos_adaptive_decision(
                MODEL_REGISTRY[model].cumulants(fwd, params),
                model=model,
                params=params,
                policy=grid,
                strike_count=K.size,
            )
            if decision.method != "cos":
                if decision.method == "lewis":
                    return np.asarray(
                        lewis_call_prices(
                            phi,
                            K,
                            spot=fwd.S0,
                            texp=fwd.T,
                            intr=fwd.r,
                            divr=fwd.q,
                            method="trapz",
                            u_max=200.0,
                            n_u=max(4096, decision.grid.N),
                        ),
                        dtype=np.float64,
                    )
                if decision.method == "carr_madan":
                    eta = 0.10 if decision.grid.width > 48.0 else 0.25
                    cm_grid = FFTGrid(N=max(4096, decision.grid.N), eta=eta, alpha=1.5)
                    return np.asarray(
                        carr_madan_price_at_strikes(phi, fwd, cm_grid, K), dtype=np.float64
                    )
            payoff_mode = _improved_cos_payoff_mode(model, decision.grid)
            res = cos_prices(phi, fwd, K, decision.grid, payoff_mode=payoff_mode)
            return np.asarray(res.call_prices, dtype=np.float64)
        if grid is None:
            grid = cos_auto_grid(MODEL_REGISTRY[model].cumulants(fwd, params), N=256, L=10.0)
        res = cos_prices(phi, fwd, K, grid)
        return np.asarray(res.call_prices, dtype=np.float64)

    if method == "cos_improved":
        cums_fn = MODEL_REGISTRY[model].cumulants
        policy: COSGridPolicy = (
            grid
            if isinstance(grid, COSGridPolicy)
            else recommended_cos_policy(model, params, mode="benchmark")
        )
        improved_decision: Any | None = (
            None
            if isinstance(grid, COSGrid)
            else cos_adaptive_decision(
                cums_fn(fwd, params),
                model=model,
                params=params,
                policy=policy,
                strike_count=K.size,
            )
        )
        if isinstance(grid, COSGrid):
            payoff_mode = _improved_cos_payoff_mode(model, grid)
            res = cos_prices(phi, fwd, K, grid, payoff_mode=payoff_mode)
            return np.asarray(res.call_prices, dtype=np.float64)

        if improved_decision is None:
            raise RuntimeError("internal error: decision unexpectedly None after policy resolution")
        if improved_decision.method == "cos":
            payoff_mode = _improved_cos_payoff_mode(model, improved_decision.grid)
            res = cos_prices(phi, fwd, K, improved_decision.grid, payoff_mode=payoff_mode)
            return np.asarray(res.call_prices, dtype=np.float64)
        if improved_decision.method == "lewis":
            return np.asarray(
                lewis_call_prices(
                    phi,
                    K,
                    spot=fwd.S0,
                    texp=fwd.T,
                    intr=fwd.r,
                    divr=fwd.q,
                    method="trapz",
                    u_max=200.0,
                    n_u=max(4096, improved_decision.grid.N),
                ),
                dtype=np.float64,
            )
        if improved_decision.method == "carr_madan":
            eta = 0.10 if improved_decision.grid.width > 48.0 else 0.25
            cm_grid = FFTGrid(N=max(4096, improved_decision.grid.N), eta=eta, alpha=1.5)
            return np.asarray(carr_madan_price_at_strikes(phi, fwd, cm_grid, K), dtype=np.float64)
        raise ValueError(f"unsupported cos_improved fallback method {improved_decision.method!r}")

    if method == "cos_filtered":
        # ------------------------------------------------------------------
        # Adaptive filtered-COS path (extension beyond Junike-COS).
        #
        # ``grid`` can be one of:
        #   - None                         → auto policy + default exp filter
        #   - COSGridPolicy                → explicit policy + default exp filter
        #   - (COSGridPolicy, COSFilterSpec) → explicit policy + explicit filter
        #   - COSGrid                       → pre-built grid + default exp filter
        #
        # After resolving the grid the path mirrors cos_improved but calls
        # filtered_cos_prices (which passes filter_spec to cos_prices).
        # ------------------------------------------------------------------
        cums_fn = MODEL_REGISTRY[model].cumulants

        # --- unpack grid argument ------------------------------------------
        if isinstance(grid, tuple) and len(grid) == 2:
            policy_or_grid, filter_spec = grid
        else:
            policy_or_grid = grid
            filter_spec = COSFilterSpec("exponential", order=8)

        # Ensure filter_spec is a COSFilterSpec
        if not isinstance(filter_spec, COSFilterSpec):
            raise TypeError(
                f"cos_filtered: expected COSFilterSpec as second element of grid tuple, "
                f"got {type(filter_spec)}"
            )

        if isinstance(policy_or_grid, COSGrid):
            # Pre-built grid  -  use it directly
            cos_grid = policy_or_grid
            payoff_mode = _improved_cos_payoff_mode(model, cos_grid)
            res = filtered_cos_prices(
                phi,
                fwd,
                K,
                cos_grid,
                filter_spec=filter_spec,
                payoff_mode=payoff_mode,
            )
            return np.asarray(res.call_prices, dtype=np.float64)

        # Policy-based path (None → recommended policy)
        policy = (
            policy_or_grid
            if isinstance(policy_or_grid, COSGridPolicy)
            else recommended_cos_policy(model, params, mode="benchmark")
        )
        decision = cos_adaptive_decision(
            cums_fn(fwd, params),
            model=model,
            params=params,
            policy=policy,
            strike_count=K.size,
        )

        # Wide-interval fallback: route to lewis / carr_madan exactly as
        # cos_improved does  -  the filter is irrelevant for these engines.
        if decision.method == "lewis":
            return np.asarray(
                lewis_call_prices(
                    phi,
                    K,
                    spot=fwd.S0,
                    texp=fwd.T,
                    intr=fwd.r,
                    divr=fwd.q,
                    method="trapz",
                    u_max=200.0,
                    n_u=max(4096, decision.grid.N),
                ),
                dtype=np.float64,
            )
        if decision.method == "carr_madan":
            eta = 0.10 if decision.grid.width > 48.0 else 0.25
            cm_grid = FFTGrid(N=max(4096, decision.grid.N), eta=eta, alpha=1.5)
            return np.asarray(
                carr_madan_price_at_strikes(phi, fwd, cm_grid, K),
                dtype=np.float64,
            )

        # Normal COS path with spectral filter applied.
        payoff_mode = _improved_cos_payoff_mode(model, decision.grid)
        res = filtered_cos_prices(
            phi,
            fwd,
            K,
            decision.grid,
            filter_spec=filter_spec,
            payoff_mode=payoff_mode,
        )
        return np.asarray(res.call_prices, dtype=np.float64)

    if method == "frft":
        if grid is None:
            raise ValueError("method='frft' requires an explicit FRFTGrid")
        return np.asarray(frft_price_at_strikes(phi, fwd, grid, K), dtype=np.float64)

    if method == "carr_madan":
        if grid is None:
            raise ValueError("method='carr_madan' requires an explicit FFTGrid")
        return np.asarray(carr_madan_price_at_strikes(phi, fwd, grid, K), dtype=np.float64)

    raise ValueError(
        f"unknown method {method!r}; choose 'cos' | 'cos_improved' | 'cos_filtered' | "
        "'frft' | 'carr_madan' | 'conv' | 'mellin' | 'proj' | 'lattice' | "
        "'pde_fd' | 'sabr_hagan' | 'pyfeng_fft'"
    )


# ---------------------------------------------------------------------------
# Product-level pricing dispatcher
# ---------------------------------------------------------------------------


def price(
    product,
    model: str,
    method: str,
    fwd: "ForwardSpec",
    params,
    *,
    grid: Any = None,
) -> float | np.ndarray:
    """Price a :class:`~foureng.products.ProductSpec` under the given model and method.

    This is the product-aware counterpart to :func:`price_strip`. It routes
    vanilla Europeans plus the currently implemented product-specific engines
    for digitals, Americans, barriers, Asians, double-barriers, Bermudans,
    forward-starts, lookbacks, and selected variance-linked contracts.

    Parameters
    ----------
    product :
        A frozen product dataclass from ``foureng.products``.
    model :
        Registry key, e.g. ``"heston"``.
    method :
        Pricing engine, e.g. ``"cos_improved"``.
    fwd :
        :class:`~foureng.models.base.ForwardSpec` — spot / rate / div / maturity.
        The ``fwd.T`` is overridden by the product's own maturity where
        applicable.
    params :
        Model-specific parameter dataclass.
    grid :
        Optional grid override (passed to :func:`price_strip`).

    Returns
    -------
    float | np.ndarray
        Scalar price for single-product calls.  Returns an ndarray only when
        the product bundles multiple pay-offs (not yet implemented).
    """
    from .products.base import ProductSpec  # local import to avoid circular

    if not isinstance(product, ProductSpec):
        raise TypeError(f"price(): expected a ProductSpec subclass, got {type(product).__name__!r}")

    pt = product.product_type

    if pt == "european":
        # Override fwd.T with the product's own maturity.
        from .models.base import ForwardSpec as _FwdSpec
        from .products.european import EuropeanOption

        if not isinstance(product, EuropeanOption):
            raise TypeError(
                "price(): product_type='european' must be represented by "
                f"EuropeanOption, got {type(product).__name__!r}"
            )
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        results = price_strip(
            model, method, [product.strike], fwd_t, params, grid=grid, cp=product.cp
        )
        return float(results[0])

    if pt == "digital":
        from .models.base import ForwardSpec as _FwdSpec
        from .models.bsm import bsm_asset_or_nothing, bsm_cash_or_nothing
        from .pricers.cos_digital import cos_digital_price
        from .products.digital import DigitalOption

        if not isinstance(product, DigitalOption):
            raise TypeError(
                "price(): product_type='digital' must be represented by "
                f"DigitalOption, got {type(product).__name__!r}"
            )
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        if method == "digital_bsm":
            if model != "bsm":
                raise NotImplementedError(
                    "method='digital_bsm' is currently implemented only for model='bsm'."
                )
            if product.payoff_type == "cash_or_nothing":
                return bsm_cash_or_nothing(
                    fwd_t,
                    params,
                    product.strike,
                    cp=product.cp,
                    cash_amount=product.cash_amount,
                )
            return bsm_asset_or_nothing(fwd_t, params, product.strike, cp=product.cp)
        if method == "cos_digital":
            return cos_digital_price(model, fwd_t, params, product, grid=grid)
        raise NotImplementedError(
            "Digital pricing currently supports method='digital_bsm' or method='cos_digital'."
        )

    if pt == "american":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.american import AmericanOption

        if not isinstance(product, AmericanOption):
            raise TypeError(
                "price(): product_type='american' must be represented by "
                f"AmericanOption, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                "American pricing is currently implemented only for model='bsm'."
            )
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        if method == "lattice":
            lattice_grid = grid if isinstance(grid, LatticeGrid) else LatticeGrid()
            return bsm_lattice_price(
                fwd_t,
                params,
                product.strike,
                cp=product.cp,
                exercise="american",
                grid=lattice_grid,
            )
        if method == "pde_fd":
            pde_grid = grid if isinstance(grid, PDEGrid) else PDEGrid()
            return bsm_pde_fd_price(
                fwd_t,
                params,
                product.strike,
                cp=product.cp,
                exercise="american",
                grid=pde_grid,
            )
        raise NotImplementedError(
            "American pricing currently supports method='lattice' or method='pde_fd'."
        )

    if pt == "bermudan":
        from .products.bermudan import BermudanOption

        if not isinstance(product, BermudanOption):
            raise TypeError(
                "price(): product_type='bermudan' must be represented by "
                f"BermudanOption, got {type(product).__name__!r}"
            )
        if method != "cos_bermudan":
            raise NotImplementedError("Bermudan pricing currently supports method='cos_bermudan'.")
        return cos_bermudan_price(model, fwd, params, product, grid=grid)

    if pt == "barrier":
        from .products.barrier import BarrierOption

        if not isinstance(product, BarrierOption):
            raise TypeError(
                "price(): product_type='barrier' must be represented by "
                f"BarrierOption, got {type(product).__name__!r}"
            )
        if method != "barrier_bsm":
            raise NotImplementedError(
                "Barrier pricing currently supports method='barrier_bsm' for "
                "closed-form BSM single-barrier contracts."
            )
        if model != "bsm":
            raise NotImplementedError(
                "method='barrier_bsm' is currently implemented only for model='bsm'."
            )
        if product.monitoring != "continuous":
            raise NotImplementedError(
                "method='barrier_bsm' currently supports only continuous monitoring."
            )
        if product.rebate != 0.0:
            raise NotImplementedError("method='barrier_bsm' currently supports only zero rebates.")
        return bsm_barrier_price(
            fwd.S0,
            product.strike,
            product.barrier,
            fwd.r,
            fwd.q,
            product.maturity,
            params.sigma,
            product.barrier_type,
            cp=product.cp,
        )

    if pt == "asian":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.asian import AsianOption

        if not isinstance(product, AsianOption):
            raise TypeError(
                "price(): product_type='asian' must be represented by "
                f"AsianOption, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                "Asian pricing is currently implemented only for model='bsm'."
            )
        if method == "asian_bsm":
            if product.average_type != "geometric" or product.strike_type != "fixed":
                raise NotImplementedError(
                    "method='asian_bsm' currently supports fixed-strike geometric Asians only."
                )
            return bsm_discrete_geometric_asian(
                fwd.S0,
                product.strike,
                fwd.r,
                fwd.q,
                product.monitoring_times,
                params.sigma,
                cp=product.cp,
            )
        if method == "asian_mc":
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Asian pricing currently supports method='asian_bsm' or method='asian_mc'."
        )

    if pt == "double_barrier":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.barrier import DoubleBarrierOption

        if not isinstance(product, DoubleBarrierOption):
            raise TypeError(
                "price(): product_type='double_barrier' must be represented by "
                f"DoubleBarrierOption, got {type(product).__name__!r}"
            )
        if method != "double_barrier_mc":
            raise NotImplementedError(
                "Double-barrier pricing currently supports method='double_barrier_mc'."
            )
        if model != "bsm":
            raise NotImplementedError(
                "method='double_barrier_mc' is currently implemented only for model='bsm'."
            )
        if product.rebate != 0.0:
            raise NotImplementedError("double_barrier_mc currently supports only zero rebates.")
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    if pt == "forward_start":
        from .products.forward_start import ForwardStartOption

        if not isinstance(product, ForwardStartOption):
            raise TypeError(
                "price(): product_type='forward_start' must be represented by "
                f"ForwardStartOption, got {type(product).__name__!r}"
            )
        if method != "forward_start_bsm":
            raise NotImplementedError(
                "Forward-start pricing currently supports method='forward_start_bsm'."
            )
        if model != "bsm":
            raise NotImplementedError(
                "method='forward_start_bsm' is currently implemented only for model='bsm'."
            )
        return bsm_forward_start(
            fwd.S0,
            product.alpha,
            product.start_time,
            product.maturity,
            fwd.r,
            fwd.q,
            params.sigma,
            cp=product.cp,
        )

    if pt == "exchange":
        from .products.multi_asset import ExchangeOption

        if not isinstance(product, ExchangeOption):
            raise TypeError(
                "price(): product_type='exchange' must be represented by "
                f"ExchangeOption, got {type(product).__name__!r}"
            )
        if method != "exchange_bsm":
            raise NotImplementedError("Exchange pricing currently supports method='exchange_bsm'.")
        if model != "bsm":
            raise NotImplementedError(
                "method='exchange_bsm' is currently implemented only for model='bsm'."
            )
        return margrabe_exchange(
            fwd.S0,
            product.spot2,
            fwd.q,
            product.q2,
            product.maturity,
            params.sigma,
            product.sigma2,
            product.rho,
        )

    if pt == "lookback":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.lookback import LookbackOption

        if not isinstance(product, LookbackOption):
            raise TypeError(
                "price(): product_type='lookback' must be represented by "
                f"LookbackOption, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        if method == "lookback_bsm":
            if product.monitoring != "continuous":
                raise NotImplementedError(
                    "method='lookback_bsm' currently supports only continuous monitoring."
                )
            if product.strike_type != "floating":
                raise NotImplementedError(
                    "method='lookback_bsm' currently supports only floating-strike lookbacks."
                )
            return bsm_lookback_floating(
                fwd.S0,
                S_min=fwd.S0,
                S_max=fwd.S0,
                r=fwd.r,
                q=fwd.q,
                T=product.maturity,
                sigma=params.sigma,
                cp=product.cp,
            )
        if method == "lookback_mc":
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Lookback pricing currently supports method='lookback_bsm' or method='lookback_mc'."
        )

    if pt == "variance_swap":
        from .analytics.bsm_variance import bsm_variance_swap
        from .models.base import ForwardSpec as _FwdSpec
        from .products.variance import VarianceSwap

        if not isinstance(product, VarianceSwap):
            raise TypeError(
                "price(): product_type='variance_swap' must be represented by "
                f"VarianceSwap, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        if method == "variance_analytic_bsm":
            return bsm_variance_swap(fwd_t, params, product)
        if method != "variance_mc":
            raise NotImplementedError(
                "Variance-swap pricing currently supports method='variance_analytic_bsm' "
                "or method='variance_mc'."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec(n_steps=len(product.sampling_times))
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    if pt == "variance_option":
        from .analytics.bsm_variance import bsm_variance_option_integrated
        from .models.base import ForwardSpec as _FwdSpec
        from .products.variance import VarianceOption

        if not isinstance(product, VarianceOption):
            raise TypeError(
                "price(): product_type='variance_option' must be represented by "
                f"VarianceOption, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        if method == "variance_analytic_bsm":
            if product.variance_type != "integrated":
                raise NotImplementedError(
                    "method='variance_analytic_bsm' currently supports integrated-variance "
                    "options only."
                )
            return bsm_variance_option_integrated(fwd_t, params, product)
        if method != "variance_mc":
            raise NotImplementedError(
                "Variance-option pricing currently supports method='variance_analytic_bsm' "
                "or method='variance_mc'."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec(n_steps=len(product.sampling_times))
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    if pt == "cliquet":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.cliquet import CliquetOption

        if not isinstance(product, CliquetOption):
            raise TypeError(
                "price(): product_type='cliquet' must be represented by "
                f"CliquetOption, got {type(product).__name__!r}"
            )
        if method != "cliquet_mc":
            raise NotImplementedError("Cliquet pricing currently supports method='cliquet_mc'.")
        if model != "bsm":
            raise NotImplementedError(
                "method='cliquet_mc' is currently implemented only for model='bsm'."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec(n_steps=len(product.reset_times))
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    # For future products, provide a capability hint instead of a bare NotImplementedError.
    _HINTS: dict[str, str] = {
        "digital": "Use method='cos_digital' or method='digital_bsm' for BSM closed form.",
        "barrier": "Use barrier_bsm for continuous zero-rebate BSM barriers.",
        "asian": "Use asian_bsm for geometric Asians or asian_mc for BSM Monte Carlo.",
        "bermudan": "Use cos_bermudan for supported 1-D Levy models.",
        "american": "Use american_lattice / american_pde (Phase 4.4).",
        "lookback": "Use lookback_bsm for continuous floating lookbacks or lookback_mc for BSM Monte Carlo.",
        "forward_start": "Use forward_start_bsm for BSM forward-start options.",
        "variance_swap": "Use variance_analytic_bsm or variance_mc for BSM variance swaps.",
        "variance_option": (
            "Use variance_analytic_bsm for integrated-variance options or "
            "variance_mc for realised/integrated variance options."
        ),
        "cliquet": "Use cliquet_mc for BSM Monte Carlo cliquets.",
        "exchange": "Use exchange_bsm for the two-asset Margrabe closed form.",
        "basket": "Use multi_asset_mc (Phase 4.11).",
        "spread": "Use multi_asset_mc / Kirk approximation (Phase 4.11).",
        "best_of": "Use multi_asset_mc (Phase 4.11).",
        "double_barrier": "Use double_barrier_mc for BSM Monte Carlo double barriers.",
    }
    hint = _HINTS.get(pt, f"No pricer is registered for product_type={pt!r}.")
    raise NotImplementedError(f"price(): product_type={pt!r} is not yet implemented.\n{hint}")
