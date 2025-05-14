"""Pricer-layer base contract.

**Pass 1: placeholder.** Today's pricers are free functions
(``carr_madan_call_prices``, ``cos_prices``, ``frft_price_at_strikes``)
dispatched from ``foureng.pipeline``. Pass 2/3 of the PyFENG-compat
restructure introduces one ``BasePricer``-derived class per method
(``CarrMadanPricer``, ``COSPricer``, ``FRFTPricer``) consumed by the
``sv_fft`` / ``sv_cos`` / ``sv_frft`` façades.

The contract is declared here ahead of time so that downstream modules
can start importing the symbol without churn later. Concrete subclasses
land in Pass 2.
"""
from __future__ import annotations


class BasePricer:
    """Abstract pricer contract — one class per Fourier method.

    **Pass 1: empty placeholder.** Intended Pass-2 surface:

        class BasePricer:
            method_name: str

            def price(self, model, strike, spot, texp, cp=1, **kwargs):
                raise NotImplementedError

    Today's free-function API in ``carr_madan.py`` / ``cos.py`` /
    ``frft.py`` keeps working as-is; this stub just reserves the name.
    """

    method_name: str = ""


__all__ = ["BasePricer"]
