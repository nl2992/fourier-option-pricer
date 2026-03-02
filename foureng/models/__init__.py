"""Characteristic functions and parameter dataclasses for all 20 models.

PyFENG-backed (thin adapters around ``pyfeng.*Fft.logp_cf``, requires pyfeng>=0.4.0):

    bsm          BsmParams      — Black-Scholes-Merton
    heston       HestonParams   — Heston (1993) stochastic volatility
    ousv         OusvParams     — Schöbel-Zhu (1999) OU stochastic volatility
    vg           VGParams       — Variance Gamma (Madan-Carr-Chang 1998)
    cgmy         CgmyParams     — CGMY (Carr-Geman-Madan-Yor 2002)
    nig          NigParams      — Normal Inverse Gaussian (Barndorff-Nielsen 1997)
    sv32         Sv32Params     — 3/2 stochastic volatility (Lewis 2000)
    rough_heston RoughHestonParams — Rough Heston (El Euch & Rosenbaum 2019)

In-house native CFs:

    kou          KouParams             — double-exponential jump-diffusion
    bates        BatesParams           — Heston + Merton log-normal jumps
    heston_kou   HestonKouParams       — Heston + Kou jumps
    heston_cgmy  HestonCGMYParams      — Heston + CGMY jumps
    garch_wmw2012 GarchWMW2012Params   — discrete GARCH (Wu-Ma-Wang 2012)
    merton_jd    MertonJDParams        — Merton (1976) jump-diffusion
    meixner      MeixnerParams         — Meixner process (Schoutens 2002)
    bilateral_gamma BilateralGammaParams — Bilateral Gamma (Küchler-Tappe 2008)
    generalized_hyperbolic GHParams    — GH Lévy (Barndorff-Nielsen 1977)
    fmls           FMLSParams            — Finite Moment Log Stable (Carr-Wu 2003)
    double_heston  DoubleHestonParams    — Two-factor Heston SV (Christoffersen 2009)
    vgsa           VGSAParams            — VG with Stochastic Arrival (CGMY 2003)
"""

from .base import CharFunc, ForwardSpec, ModelSpec
from .bates import BatesParams, bates_cf, bates_cumulants
from .bilateral_gamma import BilateralGammaParams, bilateral_gamma_cf, bilateral_gamma_cumulants
from .bsm import BsmParams, bsm_cf, bsm_cumulants
from .cgmy import CgmyParams, cgmy_cf, cgmy_cumulants
from .double_heston import DoubleHestonParams, double_heston_cf, double_heston_cumulants
from .fmls import FMLSParams, fmls_cf, fmls_cumulants
from .garch_wmw2012 import GarchWMW2012Params, garch_wmw2012_cf, garch_wmw2012_cumulants
from .generalized_hyperbolic import GHParams, gh_cf, gh_cumulants
from .heston import HestonParams, heston_cf, heston_cf_form2, heston_cumulants, heston_riccati_cd
from .heston_cgmy import HestonCGMYParams, cgmy_levy_exponent, heston_cgmy_cf, heston_cgmy_cumulants
from .heston_kou import HestonKouParams, heston_kou_cf, heston_kou_cumulants
from .kou import KouParams, kou_cf, kou_cumulants
from .meixner import MeixnerParams, meixner_cf, meixner_cumulants
from .merton_jd import MertonJDParams, merton_jd_cf, merton_jd_cumulants
from .nig import NigParams, nig_cf, nig_cumulants
from .ousv import OusvParams, ousv_cf, ousv_cumulants
from .rough_heston import RoughHestonParams, rough_heston_cf, rough_heston_cumulants
from .sv32 import Sv32Params, sv32_cf, sv32_cumulants
from .variance_gamma import VGParams, vg_cf, vg_cumulants
from .vgsa import VGSAParams, vgsa_cf, vgsa_cumulants

__all__ = [
    # base
    "ForwardSpec",
    "ModelSpec",
    "CharFunc",
    # BSM
    "BsmParams",
    "bsm_cf",
    "bsm_cumulants",
    # Heston (+ Riccati helper)
    "HestonParams",
    "heston_cf",
    "heston_cf_form2",
    "heston_cumulants",
    "heston_riccati_cd",
    # OUSV
    "OusvParams",
    "ousv_cf",
    "ousv_cumulants",
    # Variance Gamma
    "VGParams",
    "vg_cf",
    "vg_cumulants",
    # CGMY
    "CgmyParams",
    "cgmy_cf",
    "cgmy_cumulants",
    # NIG
    "NigParams",
    "nig_cf",
    "nig_cumulants",
    # Kou
    "KouParams",
    "kou_cf",
    "kou_cumulants",
    # Bates
    "BatesParams",
    "bates_cf",
    "bates_cumulants",
    # Heston-Kou
    "HestonKouParams",
    "heston_kou_cf",
    "heston_kou_cumulants",
    # Heston-CGMY
    "HestonCGMYParams",
    "cgmy_levy_exponent",
    "heston_cgmy_cf",
    "heston_cgmy_cumulants",
    # 3/2 SV
    "Sv32Params",
    "sv32_cf",
    "sv32_cumulants",
    # GARCH
    "GarchWMW2012Params",
    "garch_wmw2012_cf",
    "garch_wmw2012_cumulants",
    # Rough Heston
    "RoughHestonParams",
    "rough_heston_cf",
    "rough_heston_cumulants",
    # Merton JD
    "MertonJDParams",
    "merton_jd_cf",
    "merton_jd_cumulants",
    # Meixner
    "MeixnerParams",
    "meixner_cf",
    "meixner_cumulants",
    # Bilateral Gamma
    "BilateralGammaParams",
    "bilateral_gamma_cf",
    "bilateral_gamma_cumulants",
    # Generalized Hyperbolic
    "GHParams",
    "gh_cf",
    "gh_cumulants",
    # FMLS
    "FMLSParams",
    "fmls_cf",
    "fmls_cumulants",
    # Double Heston
    "DoubleHestonParams",
    "double_heston_cf",
    "double_heston_cumulants",
    # VGSA
    "VGSAParams",
    "vgsa_cf",
    "vgsa_cumulants",
]
