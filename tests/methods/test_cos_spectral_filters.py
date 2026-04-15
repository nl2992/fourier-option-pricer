"""Tests for foureng/utils/spectral_filters.py.

Covers:
- Shape and finiteness for every supported filter name.
- Identity: none-filter returns all-ones.
- Exponential filter damps high-frequency terms to near-zero.
- Fejér filter is monotone non-increasing.
- Invalid N raises ValueError.
- Unknown filter name raises ValueError.
- N=1 edge case: returns [1.0] for any filter.
- sigma[0] = 1 for every filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from foureng.utils.spectral_filters import COSFilterSpec, cos_filter_weights

pytestmark = [pytest.mark.numerical_stability]


# ---------------------------------------------------------------------------
# Parametrised shape / finiteness / sigma_0 = 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["none", "fejer", "lanczos", "raised_cosine", "exponential"])
def test_cos_filter_weights_shape_and_finiteness(name):
    """Every filter must return shape (N,) float64 with all finite values."""
    w = cos_filter_weights(64, COSFilterSpec(name=name))
    assert w.shape == (64,)
    assert w.dtype == float
    assert np.all(np.isfinite(w))
    assert w[0] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["none", "fejer", "lanczos", "raised_cosine", "exponential"])
def test_filter_sigma0_exactly_one(name):
    """sigma_0 must be exactly 1.0 (not just approximately) for all filters."""
    w = cos_filter_weights(128, COSFilterSpec(name=name))
    assert w[0] == 1.0


# ---------------------------------------------------------------------------
# None → identity
# ---------------------------------------------------------------------------


def test_none_filter_is_identity():
    """COSFilterSpec('none') must return np.ones(N)."""
    w = cos_filter_weights(32, COSFilterSpec("none"))
    assert np.allclose(w, np.ones(32))


def test_none_filter_via_spec_none():
    """spec=None is equivalent to COSFilterSpec('none')."""
    w_none = cos_filter_weights(32, None)
    w_explicit = cos_filter_weights(32, COSFilterSpec("none"))
    assert np.allclose(w_none, w_explicit)


# ---------------------------------------------------------------------------
# Exponential filter properties
# ---------------------------------------------------------------------------


def test_exponential_filter_damps_high_frequency_terms():
    """Exponential filter: w[0]=1, w[-1]<1e-8, all values in [0,1]."""
    w = cos_filter_weights(128, COSFilterSpec("exponential", order=8))
    assert w[0] == pytest.approx(1.0)
    assert w[-1] < 1e-8, f"Highest-frequency weight not damped: w[-1]={w[-1]:.3e}"
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0 + 1e-15)


def test_exponential_filter_order_4_less_aggressive_than_order_12():
    """Higher order → more aggressive high-frequency damping."""
    w4 = cos_filter_weights(128, COSFilterSpec("exponential", order=4))
    w12 = cos_filter_weights(128, COSFilterSpec("exponential", order=12))
    # At mid-range frequency (k≈N//2), higher order should damp less
    mid = 128 // 2
    assert w4[mid] < w12[mid], (
        f"Expected order-4 to damp more at k={mid} than order-12; "
        f"w4[mid]={w4[mid]:.4f}, w12[mid]={w12[mid]:.4f}"
    )


def test_exponential_filter_custom_alpha():
    """Custom alpha is honoured: very small alpha → near-identity filter."""
    w = cos_filter_weights(64, COSFilterSpec("exponential", order=8, alpha=0.01))
    # All weights should be very close to 1 with tiny alpha
    assert np.all(w > 0.99), f"min weight: {w.min():.6f}"


def test_exponential_filter_bad_order_raises():
    """Exponential filter with order ≤ 0 must raise ValueError."""
    with pytest.raises(ValueError, match="order"):
        cos_filter_weights(64, COSFilterSpec("exponential", order=0))

    with pytest.raises(ValueError, match="order"):
        cos_filter_weights(64, COSFilterSpec("exponential", order=-2))


# ---------------------------------------------------------------------------
# Fejér filter properties
# ---------------------------------------------------------------------------


def test_fejer_filter_is_monotone_decreasing():
    """Fejér (first-order Cesàro) weights are non-increasing."""
    w = cos_filter_weights(128, COSFilterSpec("fejer"))
    diffs = np.diff(w)
    assert np.all(diffs <= 1e-15), f"Fejér weights not monotone: max diff = {diffs.max():.3e}"


def test_fejer_filter_last_value():
    """Fejér last weight is 1/(N-1) from 1-x; for N=128, x[-1]=1 → w[-1]=0."""
    w = cos_filter_weights(128, COSFilterSpec("fejer"))
    assert w[-1] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Lanczos filter properties
# ---------------------------------------------------------------------------


def test_lanczos_filter_first_few_values():
    """Lanczos filter: w[0]=1, w[1]=sinc(1/(N-1)) close to 1."""
    N = 64
    w = cos_filter_weights(N, COSFilterSpec("lanczos"))
    assert w[0] == pytest.approx(1.0)
    expected_1 = float(np.sinc(1.0 / (N - 1)))
    assert w[1] == pytest.approx(expected_1, rel=1e-12)


# ---------------------------------------------------------------------------
# Raised-cosine filter properties
# ---------------------------------------------------------------------------


def test_raised_cosine_filter_endpoints():
    """Raised-cosine: w[0] = 1, w[-1] = 0."""
    w = cos_filter_weights(64, COSFilterSpec("raised_cosine"))
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# N=1 edge case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["none", "fejer", "lanczos", "raised_cosine", "exponential"])
def test_n1_returns_one(name):
    """N=1 should return [1.0] for every filter."""
    w = cos_filter_weights(1, COSFilterSpec(name=name))
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Invalid N
# ---------------------------------------------------------------------------


def test_invalid_N_raises():
    """N=0 and N<0 must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        cos_filter_weights(0, COSFilterSpec("none"))

    with pytest.raises(ValueError, match="positive"):
        cos_filter_weights(-1, COSFilterSpec("fejer"))


# ---------------------------------------------------------------------------
# Unknown filter name
# ---------------------------------------------------------------------------


def test_unknown_filter_name_raises():
    """Unrecognised filter name must raise ValueError."""
    spec = COSFilterSpec(name="butterworth")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown"):
        cos_filter_weights(64, spec)


# ---------------------------------------------------------------------------
# Large N stability
# ---------------------------------------------------------------------------


def test_large_N_no_overflow():
    """Filter weights for large N should remain finite."""
    for name in ["fejer", "lanczos", "raised_cosine", "exponential"]:
        w = cos_filter_weights(16384, COSFilterSpec(name=name))
        assert np.all(np.isfinite(w)), f"non-finite weights for {name} at N=16384"
