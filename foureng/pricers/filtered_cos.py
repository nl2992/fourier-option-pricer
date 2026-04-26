"""Spectral-filtered COS pricer — thin convenience wrapper.

This module exposes ``filtered_cos_prices``, a one-liner that calls the
standard :func:`~foureng.pricers.cos.cos_prices` with a non-trivial
``filter_spec``.  It exists so that calling code and the pipeline can
reference a named function rather than always assembling the keyword
argument inline.

The ``FilteredCOSDecision`` dataclass mirrors ``COSPolicyDecision`` for the
filtered path, recording both the resolved grid and the applied filter so
that the grid-search comparator can store and display them in a uniform table.

All pricing logic lives in :func:`~foureng.pricers.cos.cos_prices`; no
numerical code is duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.base import CharFunc, ForwardSpec
from ..utils.grids import COSGrid
from ..utils.spectral_filters import COSFilterSpec
from .cos import COSResult, cos_prices


@dataclass(frozen=True)
class FilteredCOSDecision:
    """Record of the resolved grid + filter for the filtered-COS path.

    Analogous to ``COSPolicyDecision`` but extended with a ``filter_spec``
    field so that grid-search results tables can show both axes.
    """

    grid: COSGrid
    filter_spec: COSFilterSpec
    method: str
    reason: str = ""


def filtered_cos_prices(
    phi: CharFunc,
    fwd: ForwardSpec,
    strikes: np.ndarray,
    grid: COSGrid,
    *,
    filter_spec: COSFilterSpec = COSFilterSpec("exponential", order=8),
    payoff_mode: str = "put_parity",
) -> COSResult:
    """COS pricer with spectral filtering applied to the CF samples.

    This is a direct thin wrapper around :func:`~foureng.pricers.cos.cos_prices`
    with the ``filter_spec`` keyword forwarded.  The default filter is the
    order-8 exponential filter, which damps high-frequency terms to
    machine-ε while leaving the low-frequency terms intact.

    Parameters
    ----------
    phi :
        Characteristic function callable, same as for :func:`cos_prices`.
    fwd :
        Forward specification (:class:`~foureng.models.base.ForwardSpec`).
    strikes :
        1-D strike array.
    grid :
        Resolved COS grid (:class:`~foureng.utils.grids.COSGrid`).
    filter_spec :
        Spectral filter to apply.  Defaults to
        ``COSFilterSpec("exponential", order=8)``.
    payoff_mode :
        Passed to :func:`cos_prices`; see its docstring.

    Returns
    -------
    COSResult
        Same structure as from :func:`cos_prices`.
    """
    return cos_prices(
        phi,
        fwd,
        strikes,
        grid,
        payoff_mode=payoff_mode,
        filter_spec=filter_spec,
    )
