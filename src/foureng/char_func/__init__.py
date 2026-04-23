"""Characteristic functions and parameter dataclasses for each model.

PyFENG-backed CFs (thin adapters around ``pyfeng.*Fft.charfunc_logprice``):

    * BSM          : :class:`pyfeng.BsmFft`
    * Heston       : :class:`pyfeng.HestonFft`
    * OUSV         : :class:`pyfeng.OusvFft`   (Schöbel-Zhu 1999)
    * VG           : :class:`pyfeng.VarGammaFft` (Madan-Carr-Chang 1998)
    * CGMY         : :class:`pyfeng.CgmyFft`   (Carr-Geman-Madan-Yor 2002)
    * NIG          : :class:`pyfeng.ExpNigFft` (Barndorff-Nielsen 1997)

In-house CFs (not provided by PyFENG):

    * Kou, Bates, Heston-Kou, Heston-CGMY — the SVJ composites layer
      an independent jump factor on top of the PyFENG-backed Heston CF:

        - Bates        : Heston + Merton lognormal jumps
        - Heston-Kou   : Heston + Kou double-exponential jumps
        - Heston-CGMY  : Heston + CGMY tempered-stable jumps
"""
from .base import ForwardSpec, ModelSpec, CharFunc
from .bsm import (
    BsmParams,
    bsm_cf,
    bsm_cumulants,
)
from .ousv import (
    OusvParams,
    ousv_cf,
    ousv_cumulants,
)
from .heston import (
    HestonParams,
    heston_cf,
    heston_cf_form2,  # back-compat alias for heston_cf
    heston_cumulants,
)
from .variance_gamma import (
    VGParams,
    vg_cf,
    vg_cumulants,
)
from .cgmy import (
    CgmyParams,
    cgmy_cf,
    cgmy_cumulants,
)
from .nig import (
    NigParams,
    nig_cf,
    nig_cumulants,
)
from .kou import (
    KouParams,
    kou_cf,
    kou_cumulants,
)
from .bates import (
    BatesParams,
    bates_cf,
    bates_cumulants,
)
from .heston_kou import (
    HestonKouParams,
    heston_kou_cf,
    heston_kou_cumulants,
)
from .heston_cgmy import (
    HestonCGMYParams,
    cgmy_levy_exponent,
    heston_cgmy_cf,
    heston_cgmy_cumulants,
)

__all__ = [
    "ForwardSpec",
    "ModelSpec",
    "CharFunc",
    "BsmParams",
    "bsm_cf",
    "bsm_cumulants",
    "OusvParams",
    "ousv_cf",
    "ousv_cumulants",
    "HestonParams",
    "heston_cf",
    "heston_cf_form2",
    "heston_cumulants",
    "VGParams",
    "vg_cf",
    "vg_cumulants",
    "CgmyParams",
    "cgmy_cf",
    "cgmy_cumulants",
    "NigParams",
    "nig_cf",
    "nig_cumulants",
    "KouParams",
    "kou_cf",
    "kou_cumulants",
    "BatesParams",
    "bates_cf",
    "bates_cumulants",
    "HestonKouParams",
    "heston_kou_cf",
    "heston_kou_cumulants",
    "HestonCGMYParams",
    "cgmy_levy_exponent",
    "heston_cgmy_cf",
    "heston_cgmy_cumulants",
]
