"""Re-export the method capability registry from foureng.core.capabilities.

This module is provided so that code can import from either
``foureng.core.capabilities`` or ``foureng.pricers.registry``; the latter
matches the directory layout described in the Phase 2 design.
"""

from foureng.core.capabilities import (  # noqa: F401
    METHOD_REGISTRY,
    MethodSpec,
    explain_capability,
)
