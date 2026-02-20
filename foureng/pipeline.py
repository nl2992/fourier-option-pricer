from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any
from .models.base import ForwardSpec, CharFunc
from .models.registry import MODEL_REGISTRY
from .utils.grids import FFTGrid, FRFTGrid, COSGrid, COSGridPolicy
from .pricers.carr_madan import carr_madan_price_at_strikes
from .pricers.frft import frft_price_at_strikes
from .pricers.cos import (
    cos_adaptive_decision,
    cos_auto_grid,
    cos_prices,
    recommended_cos_policy,
)
from .pricers.filtered_cos import FilteredCOSDecision, filtered_cos_prices
from .pricers.lewis import lewis_call_prices
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


def phase4_cos(
    phi: CharFunc, fwd: ForwardSpec, strikes: np.ndarray, grid: COSGrid
) -> PhaseOutputs:
    res = cos_prices(phi, fwd, strikes, grid)
    return PhaseOutputs(strikes=res.strikes, prices=res.call_prices)


# ---------------------------------------------------------------------------
# Unified strip pricing — one call that the notebook / scoreboard goes
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
            "— PyFENG has no FFT pricer for this model. Use "
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
        m = pf.HestonFft(sigma=params.v0, vov=params.nu, rho=params.rho,
                          mr=params.kappa, theta=params.theta,
                          intr=fwd.r, divr=fwd.q)
    elif model == "ousv":
        # Mirror the OUSV CF wrapper's kwarg translation (see
        # ``models/ousv.py``): our ``sigma0`` is PyFENG's ``sigma``,
        # our ``kappa`` is PyFENG's ``mr``, our ``nu`` is PyFENG's ``vov``.
        m = pf.OusvFft(sigma=params.sigma0, mr=params.kappa,
                        theta=params.theta, vov=params.nu, rho=params.rho,
                        intr=fwd.r, divr=fwd.q)
    elif model == "vg":
        m = pf.VarGammaFft(sigma=params.sigma, vov=params.nu, theta=params.theta,
                            intr=fwd.r, divr=fwd.q)
    elif model == "cgmy":
        m = pf.CgmyFft(C=params.C, G=params.G, M=params.M, Y=params.Y,
                        intr=fwd.r, divr=fwd.q)
    elif model == "nig":
        m = pf.ExpNigFft(sigma=params.sigma, vov=params.nu, theta=params.theta,
                          intr=fwd.r, divr=fwd.q)
    elif model == "sv32":
        m = pf.Sv32Fft(sigma=params.v0, vov=params.nu, mr=params.kappa,
                        rho=params.rho, theta=params.theta,
                        intr=fwd.r, divr=fwd.q)
    elif model == "rough_heston":
        # RoughHestonFft lives in pyfeng.sv_fft; pyfeng.ex is broken under
        # newer SciPy (scipy.misc.derivative was removed).
        from pyfeng.sv_fft import RoughHestonFft as _RoughHestonFft  # type: ignore
        m = _RoughHestonFft(
            sigma=params.sigma, vov=params.vov, mr=params.mr,
            rho=params.rho, theta=params.theta, alpha=params.alpha,
            intr=fwd.r, divr=fwd.q,
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
        * ``"cos"`` — in-house COS (Fang-Oosterlee 2008),
        * ``"cos_improved"`` — adaptive COS policy with centered intervals,
          coupled N/L selection, and wide-interval fallback,
        * ``"frft"`` — in-house FRFT (Chourdakis 2004),
        * ``"carr_madan"`` — in-house Carr-Madan FFT (1999),
        * ``"pyfeng_fft"`` — PyFENG's own native FFT pricer. Available for:
          BSM, Heston, OUSV, VG, CGMY, NIG, 3/2 SV (``sv32``), Rough Heston.
          Not available for Kou, Bates, Heston-Kou, Heston-CGMY, GARCH,
          Merton JD, Meixner, Bilateral Gamma, GH, or FMLS.
    strikes :
        1-D iterable of strikes.
    fwd, params :
        Forward spec and model-specific parameter dataclass.
    grid :
        Grid object appropriate to ``method`` — :class:`FFTGrid` for
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

    phi = _cf_for(model, fwd, params)

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
                    return np.asarray(carr_madan_price_at_strikes(phi, fwd, cm_grid, K), dtype=np.float64)
            payoff_mode = _improved_cos_payoff_mode(model, decision.grid)
            res = cos_prices(phi, fwd, K, decision.grid, payoff_mode=payoff_mode)
            return np.asarray(res.call_prices, dtype=np.float64)
        if grid is None:
            grid = cos_auto_grid(MODEL_REGISTRY[model].cumulants(fwd, params), N=256, L=10.0)
        res = cos_prices(phi, fwd, K, grid)
        return np.asarray(res.call_prices, dtype=np.float64)

    if method == "cos_improved":
        cums_fn = MODEL_REGISTRY[model].cumulants
        policy = (
            grid
            if isinstance(grid, COSGridPolicy)
            else recommended_cos_policy(model, params, mode="benchmark")
        )
        decision = (
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

        if decision is None:
            raise RuntimeError("internal error: decision unexpectedly None after policy resolution")
        if decision.method == "cos":
            payoff_mode = _improved_cos_payoff_mode(model, decision.grid)
            res = cos_prices(phi, fwd, K, decision.grid, payoff_mode=payoff_mode)
            return np.asarray(res.call_prices, dtype=np.float64)
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
            return np.asarray(carr_madan_price_at_strikes(phi, fwd, cm_grid, K), dtype=np.float64)
        raise ValueError(f"unsupported cos_improved fallback method {decision.method!r}")

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
            # Pre-built grid — use it directly
            cos_grid = policy_or_grid
            payoff_mode = _improved_cos_payoff_mode(model, cos_grid)
            res = filtered_cos_prices(
                phi, fwd, K, cos_grid,
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
        # cos_improved does — the filter is irrelevant for these engines.
        if decision.method == "lewis":
            return np.asarray(
                lewis_call_prices(
                    phi, K,
                    spot=fwd.S0, texp=fwd.T, intr=fwd.r, divr=fwd.q,
                    method="trapz", u_max=200.0,
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
            phi, fwd, K, decision.grid,
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
        "'frft' | 'carr_madan' | 'pyfeng_fft'"
    )
