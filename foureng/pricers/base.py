"""Pricer-layer base contract.

Most pricers in this package are free functions (``carr_madan_price_at_strikes``,
``cos_prices``, ``frft_price_at_strikes``, ...) dispatched directly from
``foureng.pipeline``. ``BasePricer`` is a minimal common base for pricers that
want a class-based wrapper around a free-function core; today it is subclassed
only by :class:`foureng.pricers.lewis.LewisPricer`, which the pipeline does not
call (``foureng.pipeline`` calls ``lewis_call_prices`` directly). Most methods
have no such wrapper and are not expected to grow one.
"""

from __future__ import annotations


class BasePricer:
    """Common base for a class-based pricer wrapper around a free-function core.

        class BasePricer:
            method_name: str

            def price(self, model, strike, spot, texp, cp=1, **kwargs):
                raise NotImplementedError

    Subclassing this is optional: the pipeline dispatches to free functions
    directly and does not require a ``BasePricer`` subclass to exist.
    """

    method_name: str = ""


__all__ = ["BasePricer"]
