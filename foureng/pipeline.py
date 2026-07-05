"""Unified pricing pipeline that dispatches a (model, method, strikes) request to the right engine.

The main entry points are:

* :func:`price_strip` — price a vector of strikes for a European option.
* :func:`price` — price a single :class:`~foureng.products.ProductSpec` object.

The improved COS path (``method="cos_improved"``) builds the truncation interval
and cosine-term count adaptively from model cumulants, then routes wide-interval
cases to Lewis or Carr-Madan rather than forcing COS into an unfavorable geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analytics.bsm_asian import bsm_discrete_geometric_asian
from .analytics.bsm_barrier import bsm_barrier_price
from .analytics.bsm_exotics import (
    bsm_forward_start,
    bsm_lookback_floating,
    kirk_spread,
    margrabe_exchange,
)
from .mc.engine import MCSpec, mc_price
from .models.base import ForwardSpec
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
from .pricers.hilbert import hilbert_price_at_strikes
from .pricers.lattice import LatticeGrid, bsm_lattice_price, bsm_lattice_price_at_strikes
from .pricers.lewis import lewis_call_prices
from .pricers.mellin import MELLIN_SUPPORTED_MODELS, mellin_price_at_strikes
from .pricers.pde_fd import PDEGrid, bsm_pde_fd_price, bsm_pde_fd_price_at_strikes
from .pricers.proj import (
    proj_asian_price_cv,
    proj_auto_grid,
    proj_barrier_price,
    proj_bermudan_put,
    proj_european_price_at_strikes,
)
from .pricers.sabr import sabr_hagan_price_at_strikes
from .utils.grids import CONVGrid, COSGrid, COSGridPolicy, FFTGrid, HilbertGrid
from .utils.spectral_filters import COSFilterSpec

# ---------------------------------------------------------------------------
# Unified strip pricing  -  one call that the notebook / scoreboard goes
# through, with a ``backend=`` knob that switches the characteristic function
# between in-house analytic and PyFENG, and a ``method="pyfeng_fft"`` option
# that delegates to PyFENG's own pricer entirely.
# ---------------------------------------------------------------------------

_DIRECT_CALL_FRIENDLY_MODELS = {"heston", "ousv", "nig"}

# Engines whose numerical core produces call prices only; puts are recovered
# by put-call parity at the top of ``price_strip``.
_CALL_ONLY_METHODS = {"cos", "cos_improved", "cos_filtered", "frft", "carr_madan"}


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
        Any key in :data:`~foureng.models.registry.MODEL_REGISTRY` (e.g.
        ``"bsm"``, ``"heston"``, ``"vg"``, ``"cgmy"``, ``"nig"``, ``"kou"`` …),
        or ``"sabr"`` when ``method="sabr_hagan"``.
    method :
        Characteristic-function engines:
        ``"cos"`` / ``"cos_improved"`` / ``"cos_filtered"`` (Fang-Oosterlee 2008
        and the adaptive/filtered extensions), ``"carr_madan"`` (FFT, 1999),
        ``"frft"`` (Chourdakis 2004), ``"conv"`` (Fourier inversion),
        ``"mellin"`` (Mellin transform, selected Lévy models), ``"hilbert"``
        (Feng-Linetsky 2008 discrete Hilbert transform), ``"proj"`` (PROJ
        frame projection, Kirkby 2015/2017), and ``"pyfeng_fft"`` (PyFENG native
        FFT for BSM/Heston/OUSV/VG/CGMY/NIG/3-2 SV/Rough Heston).
        Non-CF baselines: ``"lattice"`` and ``"pde_fd"`` (BSM only),
        ``"sabr_hagan"`` (``model="sabr"``).
    strikes :
        1-D iterable of strikes.
    fwd, params :
        Forward spec and model-specific parameter dataclass.
    grid :
        Grid object appropriate to ``method`` — :class:`FFTGrid` for
        ``"carr_madan"``, :class:`FRFTGrid` for ``"frft"``, :class:`CONVGrid`
        for ``"conv"``, :class:`COSGrid` for ``"cos"`` (with
        :class:`COSGridPolicy` also accepted by ``"cos_improved"`` /
        ``"cos_filtered"``), :class:`LatticeGrid` / :class:`PDEGrid` for the
        BSM baselines. If ``None``, a sensible engine-specific grid is built
        (e.g. ``cos_auto_grid`` / ``proj_auto_grid`` from the model cumulants).
    cp :
        ``+1`` calls, ``-1`` puts. The CF engines return calls and apply
        put-call parity internally where needed.

    Returns
    -------
    np.ndarray
        Prices at ``strikes``.
    """
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    if K.size == 0:
        raise ValueError("strikes must be non-empty")

    # The COS/FFT family computes calls; convert once here via put-call parity
    # so ``cp=-1`` behaves identically across every engine.
    if cp == -1 and method in _CALL_ONLY_METHODS:
        calls = price_strip(model, method, K, fwd, params, grid=grid, cp=1)
        return np.asarray(calls - fwd.disc * (fwd.F0 - K), dtype=np.float64)

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

    if method == "hilbert":
        hilbert_grid = grid if isinstance(grid, HilbertGrid) else None
        return np.asarray(
            hilbert_price_at_strikes(phi, fwd, K, cp=cp, grid=hilbert_grid), dtype=np.float64
        )

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


def _proj_bermudan_put_price(model: str, fwd: ForwardSpec, params, product) -> float:
    """PROJ Bermudan **put** for 1-D Lévy models on a uniform monitoring grid.

    Builds the one-step risk-neutral CF from the model registry and drives the
    PROJ Toeplitz-FFT recursion (:func:`~foureng.pricers.proj.proj_bermudan_put`).
    Supports the same 1-D Lévy family as ``cos_bermudan``; calls and arbitrary
    (non-uniform) exercise schedules are deferred to ``method='cos_bermudan'``.
    """
    from .pricers.cos_bermudan import _SUPPORTED_MODELS

    if product.cp != -1:
        raise NotImplementedError(
            "method='proj' for Bermudans currently supports puts (cp=-1); "
            "use method='cos_bermudan' for calls."
        )
    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"method='proj' Bermudan supports 1-D Lévy models {sorted(_SUPPORTED_MODELS)}; "
            f"got model={model!r}. Use method='cos_bermudan' or method='monte_carlo'."
        )

    T = float(product.maturity)
    ex = np.sort(np.asarray(product.exercise_times, dtype=float))
    M = ex.size
    # PROJ assumes a uniform monitoring grid t = dt, 2dt, ..., M*dt = T.
    expected = np.arange(1, M + 1) * (T / M)
    if not np.allclose(ex, expected, rtol=1e-6, atol=1e-9):
        raise NotImplementedError(
            "method='proj' Bermudan requires a uniform monitoring schedule "
            "(t = dt, 2dt, ..., T); use method='cos_bermudan' for arbitrary dates."
        )

    dt = T / M
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt)
    drift = (fwd.r - fwd.q) * dt

    def step_cf(u):
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=np.complex128)

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
    grid = proj_auto_grid(MODEL_REGISTRY[model].cumulants(fwd_T, params), N=1 << 14)
    return float(
        proj_bermudan_put(
            step_cf,
            S0=fwd.S0,
            r=fwd.r,
            T=T,
            W=float(product.strike),
            M=M,
            N=grid.N,
            alph=grid.alph,
        )
    )


def _proj_barrier_price_dispatch(model: str, fwd: ForwardSpec, params, product) -> float:
    """PROJ single-barrier pricer for 1-D Lévy models.

    Builds the one-step risk-neutral CF and drives ``proj_barrier_price``.
    Supports all 4 barrier types (knock-in via in-out parity) for the same
    1-D Lévy model family as the PROJ Bermudan pricer.
    """
    from .pricers.cos_bermudan import _SUPPORTED_MODELS

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"method='proj_barrier' supports 1-D Lévy models {sorted(_SUPPORTED_MODELS)}; "
            f"got model={model!r}. Use method='barrier_bsm' or method='monte_carlo'."
        )
    if product.rebate != 0.0:
        raise NotImplementedError("method='proj_barrier' currently supports only zero rebates.")

    T = float(product.maturity)
    M = 252  # default: approximately continuous (daily monitoring)
    dt = T / M
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt)
    drift = (fwd.r - fwd.q) * dt

    def step_cf(u: np.ndarray) -> np.ndarray:
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=np.complex128)

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
    grid = proj_auto_grid(MODEL_REGISTRY[model].cumulants(fwd_T, params), N=1 << 14)

    return proj_barrier_price(
        step_cf,
        S0=fwd.S0,
        r=fwd.r,
        T=T,
        K=float(product.strike),
        H=float(product.barrier),
        M=M,
        barrier_type=product.barrier_type,
        cp=product.cp,
        N=grid.N,
        alph=grid.alph,
    )


def _proj_double_barrier_price_dispatch(model: str, fwd: ForwardSpec, params, product) -> float:
    """PROJ double-barrier pricer for 1-D Lévy models.

    Same one-step-CF construction as the single-barrier dispatch; knock-in
    handled inside the pricer via same-engine in-out parity.
    """
    from .pricers.cos_bermudan import _SUPPORTED_MODELS
    from .pricers.proj import proj_double_barrier_price

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"method='proj_double_barrier' supports 1-D Lévy models "
            f"{sorted(_SUPPORTED_MODELS)}; got model={model!r}. "
            "Use method='double_barrier_bsm' or method='monte_carlo'."
        )
    if product.rebate != 0.0:
        raise NotImplementedError(
            "method='proj_double_barrier' currently supports only zero rebates."
        )

    T = float(product.maturity)
    M = 252
    dt = T / M
    cf = MODEL_REGISTRY[model].cf
    fwd_dt = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt)
    drift = (fwd.r - fwd.q) * dt

    def step_cf(u: np.ndarray) -> np.ndarray:
        return np.exp(1j * u * drift) * np.asarray(cf(u, fwd_dt, params), dtype=np.complex128)

    fwd_T = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T)
    grid = proj_auto_grid(MODEL_REGISTRY[model].cumulants(fwd_T, params), N=1 << 14)

    return proj_double_barrier_price(
        step_cf,
        S0=fwd.S0,
        r=fwd.r,
        T=T,
        K=float(product.strike),
        L=float(product.lower_barrier),
        U=float(product.upper_barrier),
        M=M,
        knockout=product.knockout,
        cp=product.cp,
        q=fwd.q,
        N=grid.N,
        alph=grid.alph,
    )


def _proj_asian_price_dispatch(model: str, fwd: ForwardSpec, params, product) -> float:
    """PROJ arithmetic Asian pricer (geometric control variate) for 1-D Lévy models.

    Estimates the arithmetic Asian price using Monte Carlo paths with a
    PROJ-computed geometric Asian as control variate. Only fixed-strike
    arithmetic Asians on uniform monitoring grids are supported.
    """
    from .pricers.cos_bermudan import _SUPPORTED_MODELS

    if model not in _SUPPORTED_MODELS:
        raise NotImplementedError(
            f"method='proj_asian' supports 1-D Lévy models {sorted(_SUPPORTED_MODELS)}; "
            f"got model={model!r}. Use method='asian_mc' or method='monte_carlo'."
        )
    if product.average_type != "arithmetic":
        raise NotImplementedError(
            "method='proj_asian' currently supports arithmetic average Asians only."
        )
    if product.strike_type != "fixed":
        raise NotImplementedError(
            "method='proj_asian' currently supports fixed-strike Asians only."
        )

    T = float(product.maturity)
    mon_times = np.asarray(product.monitoring_times, dtype=float)
    M = len(mon_times)

    phi = _cf_for(model, ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T), params)

    return proj_asian_price_cv(
        phi,
        ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T),
        params,
        model,
        K=float(product.strike),
        T=T,
        M=M,
        cp=product.cp,
        n_paths=20_000,
        seed=42,
    )


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
        if method == "monte_carlo":
            if model != "bsm":
                raise NotImplementedError(
                    "method='monte_carlo' is currently implemented only for model='bsm'."
                )
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
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
        if method == "monte_carlo":
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
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
            "American pricing currently supports method='lattice', method='pde_fd', "
            "or method='monte_carlo'."
        )

    if pt == "bermudan":
        from .products.bermudan import BermudanOption

        if not isinstance(product, BermudanOption):
            raise TypeError(
                "price(): product_type='bermudan' must be represented by "
                f"BermudanOption, got {type(product).__name__!r}"
            )
        if method == "cos_bermudan":
            return cos_bermudan_price(model, fwd, params, product, grid=grid)
        if method == "proj":
            return _proj_bermudan_put_price(model, fwd, params, product)
        if method == "monte_carlo":
            if model != "bsm":
                raise NotImplementedError(
                    "method='monte_carlo' is currently implemented only for model='bsm'."
                )
            from .models.base import ForwardSpec as _FwdSpec

            mc_spec = (
                grid if isinstance(grid, MCSpec) else MCSpec(n_steps=len(product.exercise_times))
            )
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Bermudan pricing currently supports method='cos_bermudan', method='proj' "
            "(1-D Lévy puts), or method='monte_carlo'."
        )

    if pt == "barrier":
        from .products.barrier import BarrierOption

        if not isinstance(product, BarrierOption):
            raise TypeError(
                "price(): product_type='barrier' must be represented by "
                f"BarrierOption, got {type(product).__name__!r}"
            )
        if method == "monte_carlo":
            if model != "bsm":
                raise NotImplementedError(
                    "method='monte_carlo' is currently implemented only for model='bsm'."
                )
            from .models.base import ForwardSpec as _FwdSpec

            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        if method == "proj_barrier":
            return _proj_barrier_price_dispatch(model, fwd, params, product)
        if method != "barrier_bsm":
            raise NotImplementedError(
                "Barrier pricing currently supports method='barrier_bsm' for "
                "closed-form BSM single-barrier contracts, method='proj_barrier' "
                "for 1-D Lévy models, or method='monte_carlo'."
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
        if method == "proj_asian":
            return _proj_asian_price_dispatch(model, fwd, params, product)
        if method == "asian_cf":
            from .pricers.geometric_asian import levy_geometric_asian_price

            if product.average_type != "geometric" or product.strike_type != "fixed":
                raise NotImplementedError(
                    "method='asian_cf' supports fixed-strike geometric Asians only "
                    "(the arithmetic average has no closed CF; use 'proj_asian' or "
                    "'monte_carlo')."
                )
            return float(
                levy_geometric_asian_price(
                    model,
                    fwd,
                    params,
                    strikes=product.strike,
                    monitoring_times=product.monitoring_times,
                    cp=product.cp,
                )[0]
            )
        if model != "bsm":
            raise NotImplementedError(
                "Asian pricing is currently implemented only for model='bsm' "
                "(use method='proj_asian' for 1-D Lévy models)."
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
        if method == "monte_carlo":
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Asian pricing currently supports method='asian_bsm', method='asian_mc', "
            "method='proj_asian' (1-D Lévy arithmetic Asian with geometric CV), "
            "or method='monte_carlo'."
        )

    if pt == "double_barrier":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.barrier import DoubleBarrierOption

        if not isinstance(product, DoubleBarrierOption):
            raise TypeError(
                "price(): product_type='double_barrier' must be represented by "
                f"DoubleBarrierOption, got {type(product).__name__!r}"
            )
        if method == "proj_double_barrier":
            return _proj_double_barrier_price_dispatch(model, fwd, params, product)
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} for double-barrier options is currently implemented only "
                "for model='bsm' (use method='proj_double_barrier' for 1-D Lévy models)."
            )
        if method == "double_barrier_bsm":
            from .analytics.bsm_barrier import bsm_double_barrier_price

            if product.rebate != 0.0:
                raise NotImplementedError(
                    "method='double_barrier_bsm' currently supports only zero rebates."
                )
            return bsm_double_barrier_price(
                fwd.S0,
                product.strike,
                product.lower_barrier,
                product.upper_barrier,
                fwd.r,
                fwd.q,
                product.maturity,
                params.sigma,
                cp=product.cp,
                knockout=True,
            )
        if method not in {"double_barrier_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Double-barrier pricing currently supports method='double_barrier_bsm' "
                "(eigenfunction expansion, BSM), method='double_barrier_mc', "
                "or method='monte_carlo'."
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
        if method == "forward_start_cf":
            from .pricers.forward_start import levy_forward_start_price

            return levy_forward_start_price(
                model,
                fwd,
                params,
                alpha=product.alpha,
                start_time=product.start_time,
                maturity=product.maturity,
                cp=product.cp,
            )
        if method != "forward_start_bsm":
            raise NotImplementedError(
                "Forward-start pricing currently supports method='forward_start_bsm' "
                "or method='forward_start_cf' (exact for Levy models)."
            )
        if model != "bsm":
            raise NotImplementedError(
                "method='forward_start_bsm' is currently implemented only for model='bsm' "
                "(use method='forward_start_cf' for Levy jump models)."
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
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        if method == "exchange_bsm":
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
        if method in {"multi_asset_mc", "monte_carlo"}:
            from .models.base import ForwardSpec as _FwdSpec

            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Exchange pricing currently supports method='exchange_bsm', "
            "method='multi_asset_mc', or method='monte_carlo'."
        )

    if pt == "basket":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.multi_asset import BasketOption

        if not isinstance(product, BasketOption):
            raise TypeError(
                "price(): product_type='basket' must be represented by "
                f"BasketOption, got {type(product).__name__!r}"
            )
        if method not in {"multi_asset_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Basket pricing currently supports method='multi_asset_mc' or method='monte_carlo'."
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    if pt == "spread":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.multi_asset import SpreadOption

        if not isinstance(product, SpreadOption):
            raise TypeError(
                "price(): product_type='spread' must be represented by "
                f"SpreadOption, got {type(product).__name__!r}"
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        if method == "spread_bsm":
            return kirk_spread(
                fwd.S0,
                product.spot2,
                product.strike,
                fwd.r,
                fwd.q,
                product.q2,
                product.maturity,
                params.sigma,
                product.sigma2,
                product.rho,
                cp=product.cp,
            )
        if method in {"multi_asset_mc", "monte_carlo"}:
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Spread pricing currently supports method='spread_bsm', method='multi_asset_mc', "
            "or method='monte_carlo'."
        )

    if pt == "best_of":
        from .models.base import ForwardSpec as _FwdSpec
        from .products.multi_asset import BestOfOption

        if not isinstance(product, BestOfOption):
            raise TypeError(
                "price(): product_type='best_of' must be represented by "
                f"BestOfOption, got {type(product).__name__!r}"
            )
        if method not in {"multi_asset_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Best-of pricing currently supports method='multi_asset_mc' or method='monte_carlo'."
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm'."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

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
        if method in {"lookback_mc", "monte_carlo"}:
            mc_spec = grid if isinstance(grid, MCSpec) else MCSpec()
            fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
            return mc_price(fwd_t, params.sigma, product, mc_spec).price
        raise NotImplementedError(
            "Lookback pricing currently supports method='lookback_bsm', method='lookback_mc', "
            "or method='monte_carlo'."
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
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        if method == "variance_levy_analytic":
            from .analytics.levy_variance import levy_variance_swap

            return levy_variance_swap(model, fwd_t, params, product)
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm' "
                "(use method='variance_levy_analytic' for Levy jump models)."
            )
        if method == "variance_analytic_bsm":
            return bsm_variance_swap(fwd_t, params, product)
        if method not in {"variance_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Variance-swap pricing currently supports method='variance_analytic_bsm', "
                "method='variance_levy_analytic', or method='variance_mc' / "
                "method='monte_carlo'."
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
        if method not in {"variance_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Variance-option pricing currently supports method='variance_analytic_bsm' "
                "or method='variance_mc' / method='monte_carlo'."
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
        if method == "cliquet_cf":
            from .pricers.cliquet import levy_cliquet_price

            return levy_cliquet_price(model, fwd, params, product)
        if method not in {"cliquet_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Cliquet pricing currently supports method='cliquet_cf' (locally "
                "collared, Levy models), method='cliquet_mc', or method='monte_carlo'."
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} is currently implemented only for model='bsm' "
                "(use method='cliquet_cf' for Levy models without global collars)."
            )
        mc_spec = grid if isinstance(grid, MCSpec) else MCSpec(n_steps=len(product.reset_times))
        fwd_t = _FwdSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=product.maturity)
        return mc_price(fwd_t, params.sigma, product, mc_spec).price

    if pt == "step":
        from .products.step import StepOption

        if not isinstance(product, StepOption):
            raise TypeError(
                "price(): product_type='step' must be represented by "
                f"StepOption, got {type(product).__name__!r}"
            )
        if method != "proj_step":
            raise NotImplementedError("Step-option pricing currently supports method='proj_step'.")
        from .pricers.cos_bermudan import _SUPPORTED_MODELS
        from .pricers.proj import proj_step_price

        if model not in _SUPPORTED_MODELS:
            raise NotImplementedError(
                f"method='proj_step' supports 1-D Lévy models {sorted(_SUPPORTED_MODELS)}; "
                f"got model={model!r}."
            )
        T_step = float(product.maturity)
        M_step = int(product.n_monitoring)
        dt_step = T_step / M_step
        cf_step = MODEL_REGISTRY[model].cf
        fwd_dt_step = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=dt_step)
        drift_step = (fwd.r - fwd.q) * dt_step

        def step_cf_step(u: np.ndarray) -> np.ndarray:
            return np.exp(1j * u * drift_step) * np.asarray(
                cf_step(u, fwd_dt_step, params), dtype=np.complex128
            )

        fwd_T_step = ForwardSpec(S0=fwd.S0, r=fwd.r, q=fwd.q, T=T_step)
        grid_step = proj_auto_grid(MODEL_REGISTRY[model].cumulants(fwd_T_step, params), N=1 << 14)
        return proj_step_price(
            step_cf_step,
            S0=fwd.S0,
            r=fwd.r,
            T=T_step,
            K=float(product.strike),
            B=float(product.barrier),
            rho=float(product.rho),
            M=M_step,
            step_type=product.step_type,
            cp=product.cp,
            q=fwd.q,
            N=grid_step.N,
            alph=grid_step.alph,
        )

    if pt == "fader":
        from .pricers.fader import levy_fader_price
        from .products.fader import FaderOption

        if not isinstance(product, FaderOption):
            raise TypeError(
                "price(): product_type='fader' must be represented by "
                f"FaderOption, got {type(product).__name__!r}"
            )
        if method != "fader_cf":
            raise NotImplementedError("Fader pricing currently supports method='fader_cf'.")
        return levy_fader_price(model, fwd, params, product)

    if pt == "parisian":
        from .products.parisian import ParisianOption

        if not isinstance(product, ParisianOption):
            raise TypeError(
                "price(): product_type='parisian' must be represented by "
                f"ParisianOption, got {type(product).__name__!r}"
            )
        if method not in {"parisian_mc", "monte_carlo"}:
            raise NotImplementedError(
                "Parisian pricing currently supports method='parisian_mc' or method='monte_carlo'."
            )
        if model != "bsm":
            raise NotImplementedError(
                f"method={method!r} for Parisian options is currently implemented only for model='bsm'."
            )
        from .mc.parisian_mc import parisian_mc_price_from_product

        n_paths = getattr(grid, "n_paths", 50_000) if grid is not None else 50_000
        n_steps = getattr(grid, "n_steps", 500) if grid is not None else 500
        seed = getattr(grid, "seed", None) if grid is not None else None
        price_val, _ = parisian_mc_price_from_product(
            product,
            S0=fwd.S0,
            r=fwd.r,
            q=fwd.q,
            sigma=params.sigma,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        return price_val

    if pt == "compound":
        from .products.compound import CompoundOption

        if not isinstance(product, CompoundOption):
            raise TypeError(
                f"product_type='compound' requires a CompoundOption instance, got {type(product)}"
            )
        if model not in {"bsm"}:
            raise NotImplementedError(
                f"compound options are only supported for model='bsm', got {model!r}"
            )
        if method not in {"geske", "analytic", None}:
            raise NotImplementedError(
                f"compound options support method='geske'/'analytic', got {method!r}"
            )
        from .analytics.bsm_compound import geske_compound_price

        return geske_compound_price(
            S=fwd.S0,
            K1=product.strike_outer,
            K2=product.strike_inner,
            r=fwd.r,
            q=fwd.q,
            T1=product.maturity_outer,
            T2=product.maturity_inner,
            sigma=params.sigma,
            cp_outer=product.cp_outer,
            cp_inner=product.cp_inner,
        )

    if pt == "chooser":
        from .products.chooser import ChooserOption

        if not isinstance(product, ChooserOption):
            raise TypeError(
                f"product_type='chooser' requires a ChooserOption instance, got {type(product)}"
            )
        if model not in {"bsm"}:
            raise NotImplementedError(
                f"chooser options are only supported for model='bsm', got {model!r}"
            )
        from .analytics.bsm_chooser import bsm_chooser_price

        return bsm_chooser_price(
            S=fwd.S0,
            K=product.strike,
            r=fwd.r,
            q=fwd.q,
            T_choice=product.maturity_choice,
            T_exp=product.maturity_expiry,
            sigma=params.sigma,
        )

    # Every supported product_type is handled by an explicit branch above; reaching
    # here means the product_type is unknown / not yet routed.
    raise NotImplementedError(
        f"price(): product_type={pt!r} is not recognized or has no registered pricer."
    )
