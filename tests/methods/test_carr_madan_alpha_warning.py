"""price_strip(method='carr_madan') warns when alpha makes the damped moment infinite.

foureng.utils.validity.check_alpha probes phi at the damping line u = -i*(alpha+1)
but was never called from the pipeline. For Kou, the analytic bound is
alpha < eta1 - 1 (see foureng.utils.validity.kou_alpha_max); past that bound
E[S_T^(alpha+1)] is infinite and the damped Carr-Madan integrand is not
integrable, so price_strip should warn rather than silently return numeric
garbage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from foureng.models.base import ForwardSpec
from foureng.models.kou import KouParams
from foureng.pipeline import price_strip
from foureng.utils.grids import FFTGrid


def test_carr_madan_warns_when_kou_alpha_at_or_past_eta1_minus_1():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=3.0, eta2=2.0)
    # eta1 - 1 == 2.0; alpha = 2.5 is past the admissible bound.
    grid = FFTGrid(N=4096, eta=0.25, alpha=2.5)

    with pytest.warns(UserWarning):
        price_strip("kou", "carr_madan", np.array([100.0]), fwd, params, grid=grid)


def test_carr_madan_does_not_warn_for_admissible_alpha():
    fwd = ForwardSpec(S0=100.0, r=0.05, q=0.0, T=1.0)
    params = KouParams(sigma=0.15, lam=1.0, p=0.4, eta1=3.0, eta2=2.0)
    grid = FFTGrid(N=4096, eta=0.25, alpha=1.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price_strip("kou", "carr_madan", np.array([100.0]), fwd, params, grid=grid)
    assert not any(issubclass(w.category, UserWarning) for w in caught)
