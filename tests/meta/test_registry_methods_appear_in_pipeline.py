"""Static check: every method key in METHOD_REGISTRY must appear in pipeline.py.

A registry entry with no matching branch in pipeline.py is a claim with no
route: explain_capability() would say "Supported" for something price()
cannot actually do. This is a simple string-presence check, not a full
call-graph analysis, but it catches the specific drift this module has had
before (registry entries for methods that were never wired into price() or
price_strip()).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foureng.core.capabilities import METHOD_REGISTRY

_PIPELINE_SOURCE = (Path(__file__).resolve().parents[2] / "foureng" / "pipeline.py").read_text()


@pytest.mark.parametrize("method", sorted(METHOD_REGISTRY))
def test_registry_method_key_appears_in_pipeline(method):
    needle = f'"{method}"'
    assert needle in _PIPELINE_SOURCE, (
        f"method={method!r} is in METHOD_REGISTRY but the literal {needle} does not "
        "appear in foureng/pipeline.py; this looks like a registry claim with no route."
    )
