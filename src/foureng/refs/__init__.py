"""Paper-anchored reference values (citations + parameters + prices).

Exposed so notebooks and tests import the canonical numbers from one place
instead of re-typing them. See :mod:`foureng.refs.paper_refs`.
"""
from .paper_refs import (
    PaperAnchor,
    FO2008_HESTON_ATM,
    HESTON_PUBLISHED_STRIP,
    LEWIS_HESTON_STRIP,
    CM1999_VG_CASE4,
    PAPER_ANCHORS,
)

__all__ = [
    "PaperAnchor",
    "FO2008_HESTON_ATM",
    "HESTON_PUBLISHED_STRIP",
    "LEWIS_HESTON_STRIP",
    "CM1999_VG_CASE4",
    "PAPER_ANCHORS",
]
