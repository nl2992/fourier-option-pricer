"""Pricing-method capability registry.

This module answers the question: *can method M price product P under model X?*

Usage
-----
    from foureng.core.capabilities import METHOD_REGISTRY, explain_capability

    spec = METHOD_REGISTRY["cos"]
    print(spec.supports_products)
    # {'european', 'digital'}

    print(explain_capability("heston", "asian", "cos"))
    # Not supported: ...

Design principles
-----------------
* Every method has a :class:`MethodSpec` that declares what it can price.
* ``explain_capability`` returns a human-readable string instead of raising,
  so callers can decide whether to raise, warn, or redirect.
* Unsupported pairs never silently produce wrong answers; they always
  surface through this registry before a pricer is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class MethodSpec:
    """Declares what a pricing engine supports.

    Attributes
    ----------
    requires_cf : bool
        Engine uses the model's characteristic function.
    requires_simulation : bool
        Engine requires path simulation (Monte Carlo).
    requires_markov_state : bool
        Engine requires a low-dimensional Markov state (e.g. PDE/CTMC).
    supports_products : FrozenSet[str]
        Product types the engine can price (product_type strings).
    supports_exercise : FrozenSet[str]
        Exercise styles supported (e.g. ``{"european", "bermudan"}``).
    supports_path_dependent : bool
        True if the engine handles path-dependent payoffs natively.
    notes : str
        Optional human-readable notes for explain_capability().
    """

    requires_cf: bool = False
    requires_simulation: bool = False
    requires_markov_state: bool = False
    supports_products: FrozenSet[str] = field(default_factory=frozenset)
    supports_exercise: FrozenSet[str] = field(default_factory=frozenset)
    supports_path_dependent: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, MethodSpec] = {
    "cos": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european", "digital"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "COS / Fang-Oosterlee (2008). Terminal-payoff transforms only. "
            "Requires a CF-equipped model with known cumulants for truncation."
        ),
    ),
    "cos_improved": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european", "digital"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "Adaptive COS with Junike-2024 policy and wide-interval fallback "
            "to Lewis / Carr-Madan. Same product coverage as 'cos'."
        ),
    ),
    "cos_filtered": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european", "digital"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "Filtered COS with spectral window (Ruijter et al. 2015). "
            "Same product coverage as 'cos'."
        ),
    ),
    "carr_madan": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="FFT / Carr-Madan (1999). European calls only.",
    ),
    "frft": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="FRFT / Chourdakis (2004). European calls only.",
    ),
    "pyfeng_fft": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "PyFENG native FFT. Supported models: BSM, Heston, OUSV, VG, "
            "CGMY, NIG, 3/2 SV, Rough Heston."
        ),
    ),
    "conv": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "CONV-style Fourier probability inversion for European calls/puts. "
            "First implementation covers terminal vanilla payoffs for CF-equipped models."
        ),
    ),
    "mellin": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "European Mellin-method façade for VG, NIG, FMLS, and bilateral gamma. "
            "First slice uses validated transform inversion pending model-specific contours."
        ),
    ),
    "hilbert": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "Feng-Linetsky (2008) discrete Hilbert transform on the half-integer "
            "sinc grid; exponentially convergent Gil-Pelaez probabilities for "
            "strip-analytic characteristic functions."
        ),
    ),
    "sabr_hagan": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"european"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="Hagan lognormal SABR implied-vol approximation plus Black pricing.",
    ),
    "barrier_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"barrier"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Closed-form BSM single-barrier pricer for continuously monitored "
            "zero-rebate knock-in/knock-out calls and puts."
        ),
    ),
    "digital_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"digital"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="Closed-form BSM cash-or-nothing and asset-or-nothing digital pricer.",
    ),
    "cos_digital": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"digital"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="COS digital pricer for cash-or-nothing and asset-or-nothing European digitals.",
    ),
    "asian_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"asian"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="Closed-form BSM discretely monitored fixed-strike geometric Asian pricer.",
    ),
    "asian_cf": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"asian"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Exact discrete geometric-Asian pricer for Levy models: the average's "
            "CF is the product of per-increment CFs (Fusai-Meucci 2008), inverted "
            "with COS. No lognormal approximation."
        ),
    ),
    "asian_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"asian"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="BSM Monte Carlo arithmetic/geometric Asian pricer.",
    ),
    "proj_double_barrier": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"double_barrier"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "PROJ double-barrier pricer for 1-D Levy models: Toeplitz-FFT "
            "backward induction with absorption on both sides of the corridor "
            "at each monitoring date; knock-in via same-engine in-out parity."
        ),
    ),
    "double_barrier_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"double_barrier"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="BSM Monte Carlo double-barrier knock-in/knock-out pricer with zero rebate.",
    ),
    "forward_start_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"forward_start"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="Closed-form BSM forward-start pricer with strike set to alpha * S_tstart.",
    ),
    "forward_start_cf": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"forward_start"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Exact Levy forward-start pricer: the homogeneity factorization "
            "V = S0 e^{-q t1} EuroPrice(S0=1, K=alpha, tau) holds exactly for "
            "stationary independent increments; European leg via COS."
        ),
    ),
    "exchange_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"exchange"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="Closed-form BSM Margrabe exchange-option pricer for two correlated assets.",
    ),
    "spread_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"spread"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes="BSM Kirk spread-option approximation for two correlated assets.",
    ),
    "lookback_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"lookback"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Closed-form BSM floating-strike lookback pricer for continuous monitoring "
            "at inception."
        ),
    ),
    "lookback_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"lookback"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="BSM Monte Carlo lookback pricer for fixed- or floating-strike contracts.",
    ),
    "variance_analytic_bsm": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"variance_swap", "variance_option"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "BSM analytic variance pricer for realised-variance swaps and "
            "deterministic integrated-variance options."
        ),
    ),
    "variance_levy_analytic": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"variance_swap"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Exact discrete variance-swap fair strike for Levy models from "
            "per-increment CF cumulants; reduces to the BSM closed form when "
            "jumps are switched off."
        ),
    ),
    "variance_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"variance_swap", "variance_option"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="BSM Monte Carlo pricer for realized-variance swaps and variance options.",
    ),
    "proj_swing": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"swing"}),
        supports_exercise=frozenset({"bermudan"}),
        supports_path_dependent=True,
        notes=(
            "PROJ swing pricer: dynamic programming over (date, rights) with "
            "one Toeplitz convolution per level; n_rights=1 is Bermudan, "
            "n_rights>=dates the sum of per-date Europeans (Carmona-Touzi 2008)."
        ),
    ),
    "proj_step": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"step"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "PROJ step-option pricer (Linetsky 1999 occupation-time damping, "
            "discretely monitored): backward induction with soft killing "
            "exp(-rho dt) beyond the barrier; rho=0 is the vanilla, "
            "rho->inf the knock-out barrier."
        ),
    ),
    "fader_cf": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"fader"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Fade-in/fade-out pricer for Levy models: per-date COS density of "
            "the monitoring marginal times the remaining-life European value "
            "(spot-shift homogeneity turns the conditional values into one COS "
            "strike strip per date)."
        ),
    ),
    "cliquet_cf": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"cliquet"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "Exact locally collared cliquet for Levy models: each period's "
            "collared return is a static call spread priced by COS; additive "
            "sums and multiplicative products factorize by independence. "
            "Global collars are rejected (use cliquet_mc)."
        ),
    ),
    "cliquet_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"cliquet"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes="BSM Monte Carlo pricer for additive or multiplicative cliquet contracts.",
    ),
    "monte_carlo": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset(
            {
                "european",
                "american",
                "bermudan",
                "asian",
                "barrier",
                "double_barrier",
                "lookback",
                "variance_swap",
                "variance_option",
                "cliquet",
                "exchange",
                "basket",
                "spread",
                "best_of",
            }
        ),
        supports_exercise=frozenset({"european", "american", "bermudan"}),
        supports_path_dependent=True,
        notes=(
            "BSM Monte Carlo / Longstaff-Schwartz engine for vanilla and exotic "
            "products, including American and Bermudan early exercise."
        ),
    ),
    "multi_asset_mc": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset({"exchange", "basket", "spread", "best_of"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "BSM Monte Carlo pricer for correlated multi-asset European payoffs "
            "(exchange, basket, spread, best-of)."
        ),
    ),
    "cos_bermudan": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"bermudan", "european"}),
        supports_exercise=frozenset({"european", "bermudan"}),
        supports_path_dependent=False,
        notes=(
            "COS Bermudan via Fang & Oosterlee (2009) backward induction. "
            "Implemented for supported 1-D Levy models; stochastic-volatility "
            "extensions remain out of scope."
        ),
    ),
    "proj": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"european", "bermudan"}),
        supports_exercise=frozenset({"european", "bermudan"}),
        supports_path_dependent=False,
        notes=(
            "PROJ / Frame Projection (Kirkby 2015, 2017). Real B-spline frame "
            "projection (Haar/linear/quadratic/cubic) for European vanilla "
            "calls/puts on CF-equipped Lévy models, validated against COS to "
            "~1e-7. Bermudan puts on a uniform monitoring grid are priced via "
            "the PROJ Toeplitz-FFT recursion (proj_bermudan_put) for the same "
            "1-D Lévy family; barrier and Asian extensions via proj_barrier and proj_asian."
        ),
    ),
    "proj_barrier": MethodSpec(
        requires_cf=True,
        supports_products=frozenset({"barrier"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "PROJ single-barrier pricer (Kirkby 2015) for 1-D Lévy models. "
            "All four barrier types (down-out, up-out, down-in, up-in) for "
            "calls and puts via backward induction with barrier absorption. "
            "Supports BSM, VG, CGMY, NIG, Kou, Merton JD models."
        ),
    ),
    "proj_asian": MethodSpec(
        requires_cf=True,
        requires_simulation=True,
        supports_products=frozenset({"asian"}),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "PROJ arithmetic Asian pricer with geometric control variate. "
            "MC-based arithmetic Asian price with PROJ-computed geometric "
            "Asian as control variate for variance reduction. Supports all "
            "1-D Lévy models in the registry (BSM, VG, CGMY, NIG, Kou, Merton JD)."
        ),
    ),
    # ── Planned methods (not yet implemented) ────────────────────────────
    "pde_fd": MethodSpec(
        requires_cf=False,
        requires_markov_state=True,
        supports_products=frozenset({"european", "american", "barrier"}),
        supports_exercise=frozenset({"european", "american"}),
        supports_path_dependent=False,
        notes=(
            "Crank-Nicolson finite-difference PDE. Requires a Markov-state "
            "model (BSM or 1-D diffusion). "
            "American via free-boundary / PSOR."
        ),
    ),
    "lattice": MethodSpec(
        requires_cf=False,
        supports_products=frozenset({"european", "american", "barrier"}),
        supports_exercise=frozenset({"european", "american", "bermudan"}),
        supports_path_dependent=False,
        notes=(
            "Binomial / trinomial tree (CRR, Jarrow-Rudd). "
            "American exercise via backward induction. "
            "BSM and simple local-vol only."
        ),
    ),
    "ctmc": MethodSpec(
        requires_cf=False,
        requires_markov_state=True,
        supports_products=frozenset({"european", "american", "barrier", "bermudan"}),
        supports_exercise=frozenset({"european", "american", "bermudan"}),
        supports_path_dependent=False,
        notes=(
            "Continuous-Time Markov Chain approximation (Kirkby). "
            "Discretises the state space; supports early exercise and barriers "
            "via matrix-exponent / generator methods."
        ),
    ),
    "bsm_analytic": MethodSpec(
        requires_cf=False,
        supports_products=frozenset(
            {"european", "digital", "barrier", "asian", "lookback", "forward_start"}
        ),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=False,
        notes=(
            "BSM closed-form reference pricers. "
            "European, cash/asset digitals, geometric Asian, forward-start, "
            "single-barrier (Reiner-Rubinstein 1991), and lookback "
            "(Goldman-Sosin-Gatto 1979). BSM model only."
        ),
    ),
    "mc_gbm": MethodSpec(
        requires_cf=False,
        requires_simulation=True,
        supports_products=frozenset(
            {"european", "asian", "barrier", "lookback", "variance_swap", "variance_option"}
        ),
        supports_exercise=frozenset({"european"}),
        supports_path_dependent=True,
        notes=(
            "GBM log-Euler path Monte Carlo with antithetic variates. "
            "Arithmetic Asian (geometric-average CV), single-barrier with "
            "BGK (1999) continuity correction, floating/fixed lookback, "
            "and variance swap/option pricing. BSM model only."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Model-capability hints
# ---------------------------------------------------------------------------

# Models with one-dimensional Lévy log-price process — eligible for
# standard COS Bermudan backward induction.
_LEVY_1D_MODELS = {
    "bsm",
    "vg",
    "cgmy",
    "nig",
    "kou",
    "merton_jd",
    "bilateral_gamma",
    "generalized_hyperbolic",
    "fmls",
    "meixner",
}

# Models with stochastic volatility — COS Bermudan requires 2-D extension.
_SV_MODELS = {
    "heston",
    "ousv",
    "sv32",
    "rough_heston",
    "garch_wmw2012",
    "double_heston",
    "vgsa",
}

# Jump-diffusion models that layer jumps on a SV base
_SVJD_MODELS = {
    "bates",
    "heston_kou",
    "heston_cgmy",
}


def explain_capability(
    model: str,
    product: str,
    method: str,
) -> str:
    """Return a human-readable explanation of whether (model, product, method) is supported.

    Parameters
    ----------
    model : str
        Model registry key (e.g. ``"heston"``).
    product : str
        Product type string (e.g. ``"asian"``).
    method : str
        Method key (e.g. ``"cos"``).

    Returns
    -------
    str
        ``"Supported: ..."`` or ``"Not supported: ..."`` with a reason and
        suggested alternative where applicable.
    """
    from foureng.models.registry import MODEL_REGISTRY  # avoid circular at import time

    # --- registry checks ---
    if method not in METHOD_REGISTRY:
        known = sorted(METHOD_REGISTRY)
        return (
            f"Not supported: method={method!r} is not in the method registry. "
            f"Known methods: {known}."
        )

    if model not in MODEL_REGISTRY:
        return f"Not supported: model={model!r} is not in the model registry."

    spec = METHOD_REGISTRY[method]

    # --- product check ---
    if product not in spec.supports_products:
        supported = sorted(spec.supports_products)
        hint = _product_hint(model, product, method)
        return (
            f"Not supported: method={method!r} does not support product_type={product!r}. "
            f"Supported products for this method: {supported}. {hint}"
        )

    # --- model-specific restrictions ---
    restriction = _model_restriction(model, product, method)
    if restriction:
        return f"Not supported: {restriction}"

    # --- success ---
    notes = spec.notes or f"method={method!r} supports product={product!r}."
    return f"Supported: {notes}"


def _product_hint(model: str, product: str, method: str) -> str:
    """Return a suggested alternative for an unsupported (product, method) pair."""
    hints: dict[str, str] = {
        "asian": (
            "Asian arithmetic payoff is path-dependent; COS implementation "
            "currently handles terminal-payoff transforms only. "
            "Use Monte Carlo or PROJ Asian when implemented."
        ),
        "barrier": (
            "Barrier monitoring requires path knowledge between payoff dates. "
            "Use barrier_analytic_bsm (BSM), barrier_mc, barrier_proj, "
            "pde_fd, lattice, or ctmc."
        ),
        "lookback": (
            "Lookback payoff is path-dependent. "
            "Use lookback_bsm (BSM closed-form), lookback_mc, or PROJ."
        ),
        "bermudan": (
            "Bermudan early-exercise requires backward induction. "
            "Use cos_bermudan (Lévy models only), pde_fd (BSM), "
            "lattice, or ctmc."
        ),
        "american": (
            "American exercise requires backward induction. Use pde_fd, lattice, lsmc, or ctmc."
        ),
        "variance_swap": (
            "Variance swap payoff is path-dependent. "
            "Use variance_analytic_bsm (BSM) or variance_heston (Heston)."
        ),
        "variance_option": ("Variance option payoff is path-dependent. Use variance_mc."),
        "cliquet": (
            "Cliquet payoff is path-dependent across reset periods. Use cliquet_mc or cliquet_proj."
        ),
        "exchange": "Use multi_asset_mc or the Margrabe closed-form.",
        "basket": "Use multi_asset_mc or log-normal basket approximation.",
        "spread": "Use multi_asset_mc or Kirk approximation.",
        "best_of": "Use multi_asset_mc.",
        "digital": (
            "Cash-or-nothing / asset-or-nothing digital. "
            "Use cos_digital (COS extended payoff) or analytic BSM formula."
        ),
    }
    return hints.get(product, f"Consult the capability matrix for product={product!r}.")


def _model_restriction(model: str, product: str, method: str) -> str:
    """Return a restriction string if a model-specific limitation applies, else ''."""
    if method == "cos_bermudan":
        if model in _SV_MODELS:
            return (
                f"model={model!r} is a stochastic-volatility model; the standard "
                "COS Bermudan backward induction (Fang & Oosterlee 2009) handles "
                "only one-dimensional Markov states. The 2-D state extension is "
                "required for SV models."
            )
        if model in _SVJD_MODELS:
            return (
                f"model={model!r} combines stochastic volatility with jumps; "
                "COS Bermudan is not currently supported for this model class. "
                "Use Monte Carlo or PROJ."
            )

    if method == "pyfeng_fft":
        from foureng.models.registry import MODEL_REGISTRY

        if model in MODEL_REGISTRY and not MODEL_REGISTRY[model].pyfeng_fft:
            return (
                f"model={model!r} is not backed by a PyFENG FFT pricer. "
                f"Use 'cos', 'cos_improved', 'frft', or 'carr_madan'."
            )

    if method in {
        "barrier_bsm",
        "digital_bsm",
        "asian_bsm",
        "asian_mc",
        "double_barrier_mc",
        "forward_start_bsm",
        "monte_carlo",
        "exchange_bsm",
        "spread_bsm",
        "lookback_bsm",
        "lookback_mc",
        "variance_analytic_bsm",
        "variance_mc",
        "cliquet_mc",
        "multi_asset_mc",
    }:
        if model != "bsm":
            return f"method={method!r} is currently implemented only for model='bsm'."

    return ""
