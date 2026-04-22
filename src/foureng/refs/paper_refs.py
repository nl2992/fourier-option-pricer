"""Paper-anchored reference values — **single source of truth**.

Two flavors live here:

1. ``PaperAnchor`` — published prices from the primary literature. The
   notebook and tests grade pricer output against these as hard gates.

       * :data:`FO2008_HESTON_ATM`  — Fang & Oosterlee (2008) Heston ATM call.
       * :data:`LEWIS_HESTON_STRIP` — Lewis (2001) Heston 5-strike strip.
       * :data:`CM1999_VG_CASE4`    — Carr & Madan (1999) VG Case 4 puts.

2. ``RegressionStrip`` — frozen *numerical* references for models whose
   published papers (Bates 1996, Kou 2002, CGMY 2002) do not ship a
   clean exact price table matching the parameterisation we use. For
   those models we fix a canonical parameter set + strike strip and
   freeze high-accuracy prices produced by a stable internal method
   (Carr-Madan FFT at N=32768, cross-verified against FRFT N=16384 and
   COS N=4096 to ~1e-10). This is standard practice for numerical
   methods work — the model still comes from the paper; we're just
   using a controlled oracle to regress against.

       * :data:`OUSV_REGRESSION_STRIP_V1`
       * :data:`CGMY_REGRESSION_STRIP_V1`
       * :data:`NIG_REGRESSION_STRIP_V1`
       * :data:`BATES_REGRESSION_STRIP_V1`
       * :data:`HESTON_KOU_REGRESSION_STRIP_V1`
       * :data:`HESTON_CGMY_REGRESSION_STRIP_V1`

The ``_V1`` suffix is a version tag — if we ever regenerate the
reference prices (e.g. with a different oracle), the new set becomes
``_V2`` so downstream tests can pin to a specific version explicitly.

All arrays are marked write-protected at construction so accidental
in-place mutation in a notebook won't silently corrupt later cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..char_func.base import ForwardSpec
from ..char_func.heston import HestonParams
from ..char_func.variance_gamma import VGParams
from ..char_func.ousv import OusvParams
from ..char_func.cgmy import CgmyParams
from ..char_func.nig import NigParams
from ..char_func.bates import BatesParams
from ..char_func.heston_kou import HestonKouParams
from ..char_func.heston_cgmy import HestonCGMYParams


def _frozen_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr.setflags(write=False)
    return arr


# ---------------------------------------------------------------------------
# Paper-published anchors
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class PaperAnchor:
    """One paper-anchored reference set.

    Attributes
    ----------
    name :
        Short human-readable label.
    citation :
        One-line citation pointer to the paper / table.
    source :
        Short source tag ("Fang & Oosterlee 2008", "Lewis 2001", ...).
    fwd :
        Forward spec (S0, r, q, T) as published.
    params :
        Model-parameter dataclass.
    strikes :
        Read-only 1-D float64 array.
    prices :
        Read-only 1-D float64 array, matched shape to strikes.
    is_call :
        True if prices are calls, False for puts.
    notes :
        Free-form remark (which table, caveats, etc.).
    """

    name: str
    citation: str
    source: str
    fwd: ForwardSpec
    params: Any
    strikes: np.ndarray
    prices: np.ndarray
    is_call: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "strikes", _frozen_array(self.strikes))
        object.__setattr__(self, "prices", _frozen_array(self.prices))
        if self.strikes.shape != self.prices.shape:
            raise ValueError(
                f"{self.name}: strikes {self.strikes.shape} and prices "
                f"{self.prices.shape} must match"
            )


FO2008_HESTON_ATM = PaperAnchor(
    name="FO2008 Heston ATM",
    citation=(
        "Fang & Oosterlee (2008), 'A Novel Pricing Method for European "
        "Options Based on Fourier-Cosine Series Expansions', Table 1."
    ),
    source="Fang & Oosterlee 2008",
    fwd=ForwardSpec(S0=100.0, r=0.0, q=0.0, T=1.0),
    params=HestonParams(kappa=1.5768, theta=0.0398, nu=0.5751, rho=-0.5711, v0=0.0175),
    strikes=[100.0],
    prices=[5.785155450],
    is_call=True,
    notes="ATM call, Feller violated (2*kappa*theta = 0.126 < nu^2 = 0.331).",
)


LEWIS_HESTON_STRIP = PaperAnchor(
    name="Lewis Heston 5-strike",
    citation=(
        "Lewis (2001), 'A Simple Option Formula for General Jump-Diffusion "
        "and other Exponential Levy Processes'; 15-digit Heston reference."
    ),
    source="Lewis 2001",
    fwd=ForwardSpec(S0=100.0, r=0.01, q=0.02, T=1.0),
    params=HestonParams(kappa=4.0, theta=0.25, nu=1.0, rho=-0.5, v0=0.04),
    strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
    prices=[
        26.774758743998854,
        20.933349000596710,
        16.070154917028834,
        12.132211516709845,
        9.024913483457836,
    ],
    is_call=True,
)


CM1999_VG_CASE4 = PaperAnchor(
    name="CM1999 VG Case 4",
    citation=(
        "Carr & Madan (1999), 'Option valuation using the fast Fourier "
        "transform', Table 4 Case 4 — Variance-Gamma puts."
    ),
    source="Carr & Madan 1999",
    fwd=ForwardSpec(S0=100.0, r=0.05, q=0.03, T=0.25),
    params=VGParams(sigma=0.25, nu=2.0, theta=-0.10),
    strikes=[77.0, 78.0, 79.0],
    prices=[0.6356, 0.6787, 0.7244],
    is_call=False,
    notes="OTM put wing; paper values rounded to 4 decimals.",
)


PAPER_ANCHORS: dict[str, PaperAnchor] = {
    "fo2008_heston_atm":  FO2008_HESTON_ATM,
    "lewis_heston_strip": LEWIS_HESTON_STRIP,
    "cm1999_vg_case4":    CM1999_VG_CASE4,
}


# ---------------------------------------------------------------------------
# Frozen regression strips (numerical oracle, not paper tables)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RegressionStrip:
    """A canonical parameter + strike set with frozen high-accuracy prices.

    Attributes
    ----------
    name :
        Short human-readable label.
    model :
        Dispatcher key — one of the entries in ``pipeline._MODELS``.
    fwd :
        Forward spec.
    params :
        Model-parameter dataclass (``BatesParams``, ``HestonKouParams``,
        ``HestonCGMYParams``, ...).
    strikes :
        Read-only 1-D float64 array of strikes.
    prices :
        Read-only 1-D float64 array of European call prices at ``strikes``.
    ref_method :
        Human-readable description of how ``prices`` were produced
        (oracle method + grid parameters). Not machine-parsed.
    version :
        Integer bump if/when the frozen array is regenerated.
    notes :
        Free-form remark.
    """

    name: str
    model: str
    fwd: ForwardSpec
    params: Any
    strikes: np.ndarray
    prices: np.ndarray
    ref_method: str
    version: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "strikes", _frozen_array(self.strikes))
        object.__setattr__(self, "prices", _frozen_array(self.prices))
        if self.strikes.shape != self.prices.shape:
            raise ValueError(
                f"{self.name}: strikes {self.strikes.shape} and prices "
                f"{self.prices.shape} must match"
            )


# 41-strike mid-strip: linspace(80, 120, 41). Common across all three
# regression families so the three fixtures share the same x-axis.
_REGRESSION_STRIKES_41 = np.linspace(80.0, 120.0, 41)


# Oracle method used for all three strips below — kept identical so the
# cross-model consistency of tolerances is meaningful.
_ORACLE = "CM FFT N=32768 eta=0.10 alpha=1.5 (cross-verified vs FRFT N=16384 and COS N=4096 to ~1e-10)"


# ---- Bates ------------------------------------------------------------------

BATES_REGRESSION_STRIP_V1 = RegressionStrip(
    name="Bates regression strip v1",
    model="bates",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0),
    params=BatesParams(
        kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
        lam_j=0.5, mu_j=-0.05, sigma_j=0.10,
    ),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        2.391764510547162e+01, 2.308990868519525e+01, 2.227232123823753e+01,
        2.146534849470873e+01, 2.066945819326813e+01, 1.988511855045293e+01,
        1.911279664751753e+01, 1.835295673997119e+01, 1.760605849504413e+01,
        1.687255516362735e+01, 1.615289169394880e+01, 1.544750279495918e+01,
        1.475681095814993e+01, 1.408122444762544e+01, 1.342113526835905e+01,
        1.277691712318526e+01, 1.214892337162802e+01, 1.153748500091474e+01,
        1.094290862411892e+01, 1.036547451768921e+01, 9.805434713777915e+00,
        9.263011161871999e+00, 8.738393975937795e+00, 8.231739783416234e+00,
        7.743170193275535e+00, 7.272770401000876e+00, 6.820587948930020e+00,
        6.386631660774865e+00, 5.970870769468167e+00, 5.573234257590096e+00,
        5.193610429463975e+00, 4.831846733359787e+00, 4.487749850009130e+00,
        4.161086065337933e+00, 3.851581941126052e+00, 3.558925289704937e+00,
        3.282766468536586e+00, 3.022719990034776e+00, 2.778366450149658e+00,
        2.549254766129163e+00, 2.334904713953307e+00,
    ],
    ref_method=_ORACLE,
    version=1,
    notes="Canonical Bates params; 41-strike linspace(80,120).",
)


# ---- Heston-Kou -------------------------------------------------------------

HESTON_KOU_REGRESSION_STRIP_V1 = RegressionStrip(
    name="Heston-Kou regression strip v1",
    model="heston_kou",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0),
    params=HestonKouParams(
        kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
        lam_j=0.5, p_j=0.4, eta1=10.0, eta2=5.0,
    ),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        2.474723813260509e+01, 2.394626965526092e+01, 2.315481402685817e+01,
        2.237323463567157e+01, 2.160189767633365e+01, 2.084117130607160e+01,
        2.009142473210601e+01, 1.935302722906242e+01, 1.862634708493157e+01,
        1.791175047495200e+01, 1.720960026321044e+01, 1.652025473225181e+01,
        1.584406624161724e+01, 1.518137981735854e+01, 1.453253167484005e+01,
        1.389784767781687e+01, 1.327764174023649e+01, 1.267221417409144e+01,
        1.208184999315890e+01, 1.150681717895609e+01, 1.094736492095333e+01,
        1.040372184185204e+01, 9.876094221919086e+00, 9.364664237875981e+00,
        8.869588233438797e+00, 8.390995040568576e+00, 7.928984372177442e+00,
        7.483625308583649e+00, 7.054954901380793e+00, 6.642976919518598e+00,
        6.247660763180118e+00, 5.868940571301724e+00, 5.506714547348596e+00,
        5.160844529652112e+00, 4.831155829665341e+00, 4.517437354902701e+00,
        4.219442041390113e+00, 3.936887602848212e+00, 3.669457610014692e+00,
        3.416802901546589e+00, 3.178543327170672e+00,
    ],
    ref_method=_ORACLE,
    version=1,
    notes="Heston block as Bates strip; Kou double-exp jumps p=0.4, eta1=10, eta2=5.",
)


# ---- Heston-CGMY ------------------------------------------------------------

HESTON_CGMY_REGRESSION_STRIP_V1 = RegressionStrip(
    name="Heston-CGMY regression strip v1",
    model="heston_cgmy",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0),
    params=HestonCGMYParams(
        kappa=2.0, theta=0.04, nu=0.3, rho=-0.7, v0=0.04,
        C=0.5, G=5.0, M=5.0, Y=0.7,
    ),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        2.750852967511184e+01, 2.684192745658426e+01, 2.618657186363549e+01,
        2.554253581509971e+01, 2.490987964389410e+01, 2.428865114989791e+01,
        2.367888569499454e+01, 2.308060633861046e+01, 2.249382401189532e+01,
        2.191853772854625e+01, 2.135473483012635e+01, 2.080239126360646e+01,
        2.026147188876850e+01, 1.973193081295701e+01, 1.921371175066447e+01,
        1.870674840559563e+01, 1.821096487211327e+01, 1.772627605423953e+01,
        1.725258809873293e+01, 1.678979884068096e+01, 1.633779825824911e+01,
        1.589646893466764e+01, 1.546568652508839e+01, 1.504532022585229e+01,
        1.463523324414780e+01, 1.423528326601529e+01, 1.384532292077014e+01,
        1.346520024004596e+01, 1.309475910979588e+01, 1.273383971373432e+01,
        1.238227896683865e+01, 1.203991093765486e+01, 1.170656725812129e+01,
        1.138207752027296e+01, 1.106626965889973e+01, 1.075897031884263e+01,
        1.046000520744615e+01, 1.016919943063585e+01, 9.886377812926590e+00,
        9.611365200773742e+00, 9.343986749496434e+00,
    ],
    ref_method=_ORACLE,
    version=1,
    notes="Finite-variation CGMY (Y=0.7); symmetric tempering G=M=5, activity C=0.5.",
)


# ---- OUSV (Schöbel-Zhu) -----------------------------------------------------

OUSV_REGRESSION_STRIP_V1 = RegressionStrip(
    name="OUSV (Schöbel-Zhu) regression strip v1",
    model="ousv",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.0, T=1.0),
    params=OusvParams(sigma0=0.2, kappa=2.0, theta=0.2, nu=0.3, rho=-0.5),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        2.444538433968169e+01, 2.363084747094450e+01, 2.282550631300497e+01,
        2.202974927921989e+01, 2.124397340919312e+01, 2.046858411094865e+01,
        1.970399483044148e+01, 1.895062663365215e+01, 1.820890768301755e+01,
        1.747927258754349e+01, 1.676216160272988e+01, 1.605801965305768e+01,
        1.536729514642891e+01, 1.469043854747967e+01, 1.402790067316194e+01,
        1.338013067145525e+01, 1.274757364808278e+01, 1.213066789964810e+01,
        1.152984172560125e+01, 1.094550978949217e+01, 1.037806902059576e+01,
        9.827894058250243e+00, 9.295332267284410e+00, 8.780698381493512e+00,
        8.284268865861904e+00, 7.806276127037690e+00, 7.346902741358771e+00,
        6.906275907299752e+00, 6.484462359912205e+00, 6.081464003391137e+00,
        5.697214519132898e+00, 5.331577186259133e+00, 4.984344106049840e+00,
        4.655236962377034e+00, 4.343909364096804e+00, 4.049950716072226e+00,
        3.772891496776031e+00, 3.512209712976281e+00, 3.267338263116664e+00,
        3.037672892823789e+00, 2.822580425841516e+00,
    ],
    ref_method=_ORACLE,
    version=1,
    notes="Canonical OUSV params (Schobel-Zhu 1999 family); same 41-strike strip.",
)


# ---- Pure CGMY (no stochastic variance) -------------------------------------

CGMY_REGRESSION_STRIP_V1 = RegressionStrip(
    name="CGMY regression strip v1",
    model="cgmy",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0),
    params=CgmyParams(C=0.5, G=5.0, M=5.0, Y=0.7),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        25.04797353729841,   24.34249855322074,   23.650084601874703,
        22.970942376767667,  22.305259543128226,  21.653199932036582,
        21.01490290349881,   20.390482884788533,  19.78002908797037,
        19.183605407789273,  18.601250498386825,  18.032978024665294,
        17.47877708163105,   16.938612772704293,  16.412426935381085,
        15.900139002113276,  15.401646981271167,  14.916828542722545,
        14.445542192160964,  13.987628517338624,  13.542911488925647,
        13.111199801032855,  12.692288234477337,  12.285959028401155,
        11.891983246546259,  11.510122125625383,  11.140128394592447,
        10.781747555047236,  10.434719114465823,  10.098777765383366,
        9.773654505044709,   9.459077691348908,   9.15477403201237,
        8.860469504874658,   8.575890209385612,   8.300763148298845,
        8.034816940536619,   7.7777824678077945,  7.529393454954949,
        7.289386988857969,   7.057503976446688,
    ],
    ref_method=_ORACLE,
    version=1,
    notes=(
        "Symmetric tempering G=M=5, activity C=0.5, fine-structure Y=0.7 — "
        "a mild, finite-variation CGMY regime with non-zero r and q."
    ),
)


# ---- NIG (Normal Inverse Gaussian) ------------------------------------------

NIG_REGRESSION_STRIP_V1 = RegressionStrip(
    name="NIG regression strip v1",
    model="nig",
    fwd=ForwardSpec(S0=100.0, r=0.03, q=0.01, T=1.0),
    params=NigParams(sigma=0.2, nu=0.5, theta=-0.10),
    strikes=_REGRESSION_STRIKES_41,
    prices=[
        22.745846902307814,  21.906387100373248,  21.077393391008176,
        20.259499250508043,  19.45335416975029,   18.659621064091944,
        17.878973219444088,  17.112090752903672,  16.359656576132434,
        15.622351860317503,  14.900851014301699,  14.19581620258766,
        13.507891447321086,  12.83769637781322,   12.185819711856725,
        11.552812572255224,  10.93918176476825,   10.345383159468962,
        9.771815333812318,   9.218813645610648,   8.686644906061366,
        8.175502823787005,   7.685504374356173,   7.21668723225317,
        6.76900837390622,    6.342343924741806,   5.936490282410085,
        5.55116650442601,    5.186017904050668,   4.840620756031376,
        4.514487976387311,   4.207075609982973,   3.917789937081757,
        3.645994997334775,   3.391020334953118,   3.152168763201474,
        2.9287239699028875,  2.7199578122047776,  2.5251371568620677,
        2.343530175639868,   2.17441200974703,
    ],
    ref_method=_ORACLE,
    version=1,
    notes=(
        "Classical equity-like NIG params: sigma=0.2 (vol), nu=0.5 "
        "(kurtosis/IG-rate), theta=-0.10 (negative skew)."
    ),
)


REGRESSION_STRIPS: dict[str, RegressionStrip] = {
    "ousv":        OUSV_REGRESSION_STRIP_V1,
    "cgmy":        CGMY_REGRESSION_STRIP_V1,
    "nig":         NIG_REGRESSION_STRIP_V1,
    "bates":       BATES_REGRESSION_STRIP_V1,
    "heston_kou":  HESTON_KOU_REGRESSION_STRIP_V1,
    "heston_cgmy": HESTON_CGMY_REGRESSION_STRIP_V1,
}


__all__ = [
    "PaperAnchor",
    "FO2008_HESTON_ATM",
    "LEWIS_HESTON_STRIP",
    "CM1999_VG_CASE4",
    "PAPER_ANCHORS",
    "RegressionStrip",
    "OUSV_REGRESSION_STRIP_V1",
    "CGMY_REGRESSION_STRIP_V1",
    "NIG_REGRESSION_STRIP_V1",
    "BATES_REGRESSION_STRIP_V1",
    "HESTON_KOU_REGRESSION_STRIP_V1",
    "HESTON_CGMY_REGRESSION_STRIP_V1",
    "REGRESSION_STRIPS",
]
