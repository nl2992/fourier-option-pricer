"""Single source of truth for the supported model catalogue.

Every model that ``price_strip`` and the COS/FFT/FRFT pricers understand
is declared here as a :class:`ModelEntry`.  ``foureng.pipeline`` imports
``MODEL_REGISTRY`` instead of maintaining its own local dict.

Usage
-----
    from foureng.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY["heston"]
    phi   = lambda u: entry.cf(u, fwd, params)
    cums  = entry.cumulants(fwd, params)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .bates import BatesParams, bates_cf, bates_cumulants
from .bilateral_gamma import BilateralGammaParams, bilateral_gamma_cf, bilateral_gamma_cumulants
from .bsm import BsmParams, bsm_cf, bsm_cumulants
from .cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from .double_heston import DoubleHestonParams, double_heston_cf, double_heston_cumulants
from .fmls import FMLSParams, fmls_cf, fmls_cumulants
from .garch_wmw2012 import GarchWMW2012Params, garch_wmw2012_cf, garch_wmw2012_cumulants
from .generalized_hyperbolic import GHParams, gh_cf, gh_cumulants
from .heston import HestonParams, heston_cf, heston_cumulants
from .heston_cgmy import HestonCGMYParams, heston_cgmy_cf, heston_cgmy_cumulants
from .heston_kou import HestonKouParams, heston_kou_cf, heston_kou_cumulants
from .heston_nig import HestonNIGParams, heston_nig_cf, heston_nig_cumulants
from .kou import KouParams, kou_cf, kou_cumulants
from .meixner import MeixnerParams, meixner_cf, meixner_cumulants
from .merton_jd import MertonJDParams, merton_jd_cf, merton_jd_cumulants
from .nig import NigParams, nig_cf, nig_cumulants
from .ousv import OusvParams, ousv_cf, ousv_cumulants
from .rough_heston import RoughHestonParams, rough_heston_cf, rough_heston_cumulants
from .sv32 import Sv32Params, sv32_cf, sv32_cumulants
from .variance_gamma import VGParams, vg_cf, vg_cumulants
from .vgsa import VGSAParams, vgsa_cf, vgsa_cumulants


@dataclass(frozen=True)
class ModelEntry:
    """Metadata and callables for one supported model.

    Attributes
    ----------
    key : str
        Registry key used by ``price_strip`` and related functions.
    params_cls : type
        Parameter dataclass for this model.
    cf : Callable
        Characteristic function ``cf(u, fwd, params) -> np.ndarray``.
    cumulants : Callable
        Cumulant function ``cumulants(fwd, params) -> (c1, c2, c4)``.
    pyfeng_fft : bool
        ``True`` if PyFENG ships a native ``*Fft`` pricer for this model.
    status : str
        ``"stable"`` for all production models.
    """

    key: str
    params_cls: type
    cf: Callable
    cumulants: Callable
    pyfeng_fft: bool
    status: str = "stable"


def _e(key, params_cls, cf, cumulants, pyfeng_fft):
    return ModelEntry(
        key=key, params_cls=params_cls, cf=cf, cumulants=cumulants, pyfeng_fft=pyfeng_fft
    )


MODEL_REGISTRY: dict[str, ModelEntry] = {
    # ── PyFENG-backed ─────────────────────────────────────────────────────────
    "bsm": _e("bsm", BsmParams, bsm_cf, bsm_cumulants, True),
    "heston": _e("heston", HestonParams, heston_cf, heston_cumulants, True),
    "ousv": _e("ousv", OusvParams, ousv_cf, ousv_cumulants, True),
    "vg": _e("vg", VGParams, vg_cf, vg_cumulants, True),
    "cgmy": _e("cgmy", CgmyParams, cgmy_cf, cgmy_cumulants, True),
    "nig": _e("nig", NigParams, nig_cf, nig_cumulants, True),
    "sv32": _e("sv32", Sv32Params, sv32_cf, sv32_cumulants, True),
    "rough_heston": _e(
        "rough_heston", RoughHestonParams, rough_heston_cf, rough_heston_cumulants, True
    ),
    # ── In-house native CFs ───────────────────────────────────────────────────
    "kou": _e("kou", KouParams, kou_cf, kou_cumulants, False),
    "bates": _e("bates", BatesParams, bates_cf, bates_cumulants, False),
    "heston_kou": _e("heston_kou", HestonKouParams, heston_kou_cf, heston_kou_cumulants, False),
    "heston_cgmy": _e(
        "heston_cgmy", HestonCGMYParams, heston_cgmy_cf, heston_cgmy_cumulants, False
    ),
    "garch_wmw2012": _e(
        "garch_wmw2012", GarchWMW2012Params, garch_wmw2012_cf, garch_wmw2012_cumulants, False
    ),
    "merton_jd": _e("merton_jd", MertonJDParams, merton_jd_cf, merton_jd_cumulants, False),
    "meixner": _e("meixner", MeixnerParams, meixner_cf, meixner_cumulants, False),
    "bilateral_gamma": _e(
        "bilateral_gamma",
        BilateralGammaParams,
        bilateral_gamma_cf,
        bilateral_gamma_cumulants,
        False,
    ),
    "generalized_hyperbolic": _e("generalized_hyperbolic", GHParams, gh_cf, gh_cumulants, False),
    "fmls": _e("fmls", FMLSParams, fmls_cf, fmls_cumulants, False),
    # ── New in-house models (20-model portfolio) ──────────────────────────────
    "double_heston": _e(
        "double_heston",
        DoubleHestonParams,
        double_heston_cf,
        double_heston_cumulants,
        False,
    ),
    "vgsa": _e("vgsa", VGSAParams, vgsa_cf, vgsa_cumulants, False),
    # ── Model 21 ──────────────────────────────────────────────────────────────
    "heston_nig": _e(
        "heston_nig",
        HestonNIGParams,
        heston_nig_cf,
        heston_nig_cumulants,
        False,
    ),
}

__all__ = ["ModelEntry", "MODEL_REGISTRY"]
