from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any
from .models.base import ForwardSpec, CharFunc
from .models.bsm import BsmParams, bsm_cf, bsm_cumulants
from .models.heston import HestonParams, heston_cf, heston_cumulants
from .models.ousv import OusvParams, ousv_cf, ousv_cumulants
from .models.variance_gamma import VGParams, vg_cf, vg_cumulants
from .models.cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from .models.nig import NigParams, nig_cf, nig_cumulants
from .models.kou import KouParams, kou_cf, kou_cumulants
from .models.bates import BatesParams, bates_cf, bates_cumulants
from .models.heston_kou import (
    HestonKouParams,
    heston_kou_cf,
    heston_kou_cumulants,
)
from .models.heston_cgmy import (
    HestonCGMYParams,
    heston_cgmy_cf,
    heston_cgmy_cumulants,
)
from .utils.grids import FFTGrid, FRFTGrid, COSGrid
from .pricers.carr_madan import carr_madan_price_at_strikes
from .pricers.frft import frft_price_at_strikes
from .pricers.cos import cos_prices, cos_auto_grid


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

_MODELS: dict[str, tuple[type, Any, Any]] = {
    # model_name : (ParamsClass, cf_callable(u, fwd, p), cumulants_fn(fwd, p))
    "bsm":         (BsmParams,         bsm_cf,          bsm_cumulants),
    "heston":      (HestonParams,      heston_cf,       heston_cumulants),
    "ousv":        (OusvParams,        ousv_cf,         ousv_cumulants),
    "vg":          (VGParams,          vg_cf,           vg_cumulants),
    "cgmy":        (CgmyParams,        cgmy_cf,         cgmy_cumulants),
    "nig":         (NigParams,         nig_cf,          nig_cumulants),
    "kou":         (KouParams,         kou_cf,          kou_cumulants),
    "bates":       (BatesParams,       bates_cf,        bates_cumulants),
    "heston_kou":  (HestonKouParams,   heston_kou_cf,   heston_kou_cumulants),
    "heston_cgmy": (HestonCGMYParams,  heston_cgmy_cf,  heston_cgmy_cumulants),
}

# Models whose CF has no PyFENG FFT counterpart — ``method='pyfeng_fft'``
# raises for these. Kou / Bates / Heston-Kou / Heston-CGMY: PyFENG ships
# no FFT pricer for any of them. BSM, Heston, OUSV, VG, CGMY, NIG all
# have native PyFENG FFT pricers (BsmFft / HestonFft / OusvFft /
# VarGammaFft / CgmyFft / ExpNigFft).
_NO_PYFENG_FFT = {"kou", "bates", "heston_kou", "heston_cgmy"}


def _cf_for(model: str, fwd: ForwardSpec, params):
    if model not in _MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(_MODELS)}")
    _, cf_fn, _ = _MODELS[model]
    return lambda u: cf_fn(u, fwd, params)


def _pyfeng_fft_price(model: str, strikes, fwd: ForwardSpec, params, cp: int):
    """Call PyFENG's FFT pricer directly.

    Supported models: BSM, Heston, OUSV, VG — the four PyFENG ships
    native ``*Fft`` pricers for. Raises :class:`ValueError` for any
    entry in :data:`_NO_PYFENG_FFT` (Kou, Bates, Heston-Kou, Heston-CGMY).
    """
    if model in _NO_PYFENG_FFT:
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
        One of ``"heston"``, ``"vg"``, ``"kou"``, ``"bates"``.
    method :
        * ``"cos"`` — in-house COS (Fang-Oosterlee 2008),
        * ``"frft"`` — in-house FRFT (Chourdakis 2004),
        * ``"carr_madan"`` — in-house Carr-Madan FFT (1999),
        * ``"pyfeng_fft"`` — PyFENG's own pricer (``HestonFft`` or
          ``VarGammaFft``). Not available for Kou or Bates.
    strikes :
        1-D iterable of strikes.
    fwd, params :
        Forward spec and model-specific parameter dataclass.
    grid :
        Grid object appropriate to ``method`` — :class:`FFTGrid` for
        ``"carr_madan"``, :class:`FRFTGrid` for ``"frft"``,
        :class:`COSGrid` for ``"cos"``. If ``None`` and ``method='cos'``,
        an auto grid is built from the model cumulants with
        :func:`cos_auto_grid`.
    cp :
        ``+1`` calls, ``-1`` puts (consulted only by ``pyfeng_fft``; the
        in-house pricers return calls and the caller applies parity).

    Returns
    -------
    np.ndarray
        Prices at ``strikes``.
    """
    K = np.ascontiguousarray(np.asarray(strikes, dtype=np.float64))
    if method == "pyfeng_fft":
        return _pyfeng_fft_price(model, K, fwd, params, cp=cp)

    phi = _cf_for(model, fwd, params)

    if method == "cos":
        if grid is None:
            _, _, cums_fn = _MODELS[model]
            grid = cos_auto_grid(cums_fn(fwd, params), N=256, L=10.0)
        res = cos_prices(phi, fwd, K, grid)
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
        f"unknown method {method!r}; choose 'cos' | 'frft' | 'carr_madan' | 'pyfeng_fft'"
    )
