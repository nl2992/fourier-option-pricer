"""Shared PyFENG-backend plumbing for CF wrappers.

Every PyFENG-backed CF in this package follows the same three-step
pattern:

    1. Take our project-side parameter dataclass + :class:`ForwardSpec`.
    2. Translate to PyFENG's constructor kwargs.
    3. Cache the constructed model per (params, forward), then call
       :meth:`charfunc_logprice` on every CF request.

Keeping the pattern in one helper module lets each model file (BSM,
Heston, VG, OUSV, …) stay a thin 2-line wrapper around its own kwarg
translation. No file imports this module at top level — PyFENG is
imported lazily on first use so the project still imports cleanly in
an environment where PyFENG is not installed (the CF just raises a
clear :class:`ImportError` when invoked).

Cache design
------------
Each model file owns a private ``dict`` keyed on ``(params, fwd)`` and
hands it to :func:`build_cached`. The cache is unbounded in principle,
but a typical run sees O(1) distinct ``(params, fwd)`` tuples — one
per calibration point, one per notebook example. Constructing a
:class:`pyfeng.HestonFft` is an order of magnitude slower than a
single :meth:`charfunc_logprice` call, so caching matters for COS's
two-pass usage (cumulant Cauchy integral + strip pricing).
"""
from __future__ import annotations

from typing import Any, Callable


def import_pyfeng():
    """Import :mod:`pyfeng` with a project-branded error message.

    We defer the import so ``from foureng.models import ...`` stays
    cheap in environments that never touch PyFENG. When the caller does
    need it, a missing install produces a single actionable line rather
    than a generic ``ModuleNotFoundError`` stack.
    """
    try:
        import pyfeng as pf  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise ImportError(
            "foureng requires pyfeng for this model's CF; install with "
            "`pip install pyfeng`."
        ) from exc
    return pf


def build_cached(
    cache: dict,
    key: Any,
    factory: Callable[[], Any],
):
    """Return ``cache[key]`` if present, else compute via ``factory`` and store.

    ``factory`` is a zero-arg callable that returns the PyFENG model
    instance. Passing a thunk (rather than a prebuilt instance) means
    we skip the :class:`pyfeng.*Fft` constructor entirely on cache
    hits — the whole point of caching here.
    """
    m = cache.get(key)
    if m is not None:
        return m
    m = factory()
    cache[key] = m
    return m
