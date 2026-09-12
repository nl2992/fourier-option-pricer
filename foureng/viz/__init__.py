"""Visualisation helpers for demo notebooks and saved figures.

Requires the optional ``viz`` extra: ``pip install "fourier-option-pricer[viz]"``.
"""

try:
    import matplotlib  # noqa: F401
    import pandas  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the install
    raise ImportError(
        "foureng.viz needs matplotlib and pandas. "
        'Install them with: pip install "fourier-option-pricer[viz]"'
    ) from exc

from .columbia import (
    CLOUD,
    COLUMBIA_BLUE,
    DARK,
    GREEN,
    NAVY,
    ORANGE,
    PANEL,
    SLATE,
    WHITE,
    apply_columbia_style,
    plot_convergence,
    plot_error_bar,
    plot_error_vs_runtime,
    plot_L_sensitivity,
    plot_price_strip,
)
from .notebook_runtime import (
    ensure_repo_root_on_path,
    error_zoom_bounds,
    locate_repo_root,
    night_style,
    sci,
    style_table,
    timed_call,
    timed_median_ms,
    timeit_strip,
)

__all__ = [
    "apply_columbia_style",
    "COLUMBIA_BLUE",
    "NAVY",
    "DARK",
    "WHITE",
    "SLATE",
    "ORANGE",
    "GREEN",
    "PANEL",
    "CLOUD",
    "plot_price_strip",
    "plot_error_bar",
    "plot_convergence",
    "plot_L_sensitivity",
    "plot_error_vs_runtime",
    "ensure_repo_root_on_path",
    "error_zoom_bounds",
    "locate_repo_root",
    "night_style",
    "sci",
    "style_table",
    "timed_call",
    "timed_median_ms",
    "timeit_strip",
]
