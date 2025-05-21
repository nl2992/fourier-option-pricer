"""Columbia-themed matplotlib style + a small library of plot helpers.

Goal: uniform look-and-feel for the demo notebook and any figures exported
to ``images/``. Call ``apply_columbia_style()`` once at the top of a notebook
and every subsequent plot inherits the theme (white chart background, navy
titles, dark-navy tick labels, light dotted grid, no top/right spines).

Palette is fixed to the Columbia-standard colours so slides and figures sit
naturally next to the slide deck.
"""
from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


# ---- Palette -----------------------------------------------------------------

COLUMBIA_BLUE = "#B9D9EB"
NAVY = "#0B3D91"
DARK = "#0B1F3A"
WHITE = "#FFFFFF"
SLATE = "#5E6B7A"
ORANGE = "#D9622C"
GREEN = "#2E7D5B"
PANEL = "#F4F8FC"
CLOUD = "#EAF0F8"

# Ordered cycle used when multiple series share a panel. Keeps the first two
# series in the "brand" colours; the later ones are tonal variations that
# still read clearly against a white chart background.
_COLOR_CYCLE = [NAVY, ORANGE, COLUMBIA_BLUE, GREEN, "#5B8FB9", "#08306B", "#7FA3C7", "#1F5CA6"]


def apply_columbia_style() -> None:
    """Install the Columbia theme into the current matplotlib session."""
    plt.rcParams.update({
        # backgrounds
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        # text
        "axes.titlecolor": NAVY,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "axes.edgecolor": DARK,
        "text.color": DARK,
        "figure.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10.0,
        # layout
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "#B8BDC4",
        "grid.alpha": 0.6,
        # lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.edgecolor": WHITE,
        # font sizes
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Avenir Next",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
        ],
        # color cycle
        "axes.prop_cycle": plt.cycler(color=_COLOR_CYCLE),
        # dpi for crisp savefigs
        "savefig.dpi": 150,
        "figure.dpi": 100,
    })


# ---- Plot helpers ------------------------------------------------------------

def _new_ax(ax=None, figsize=(6.4, 3.8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def plot_price_strip(
    strikes: np.ndarray,
    series: Mapping[str, np.ndarray],
    title: str,
    xlabel: str = "strike K",
    ylabel: str = "call price",
    ax=None,
):
    """Overlay several call-price curves across a strike strip."""
    fig, ax = _new_ax(ax)
    for name, y in series.items():
        ax.plot(strikes, y, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_error_bar(
    strikes: np.ndarray,
    abs_err: np.ndarray,
    title: str,
    xlabel: str = "strike K",
    ylabel: str = "|error|",
    ax=None,
):
    """Bar chart of absolute error per strike (log y-axis)."""
    fig, ax = _new_ax(ax)
    ax.bar(strikes, abs_err, color=NAVY, width=0.7 * (strikes[1] - strikes[0])
           if len(strikes) > 1 else 0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax


def plot_convergence(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str = "N",
    ylabel: str = "max |error|",
    xlog: bool = True,
    ylog: bool = True,
    ax=None,
    label: str | None = None,
):
    """Convergence curve (typically error vs N or error vs runtime)."""
    fig, ax = _new_ax(ax)
    ax.plot(x, y, marker="o", label=label, color=NAVY)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_error_vs_runtime(
    df,
    title: str,
    label_col: str = "method",
    x_col: str = "runtime_ms",
    y_col: str = "max_abs_err_vs_ref",
    xlabel: str = "strip runtime (ms)",
    ylabel: str = "max |error| vs reference",
    ax=None,
):
    """Log-log frontier: one marker per row, labelled by method.

    Pass a DataFrame with at least ``[label_col, x_col, y_col]`` columns.
    Rows with NaN in either axis are silently dropped (handy when a method,
    e.g. PyFENG, is unavailable and the row holds a sentinel NaN).
    """
    import numpy as np  # local import so the module has no hard numpy at top
    fig, ax = _new_ax(ax, figsize=(6.8, 4.2))
    d = df[[label_col, x_col, y_col]].copy()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for i, (_, row) in enumerate(d.iterrows()):
        ax.scatter(row[x_col], row[y_col],
                   marker=markers[i % len(markers)],
                   s=90, color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)],
                   edgecolor=DARK, linewidths=0.6,
                   label=str(row[label_col]), zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax


def plot_L_sensitivity(
    strikes: np.ndarray,
    curves_by_L: Mapping[float, np.ndarray],
    title: str,
    xlabel: str = "strike K",
    ylabel: str = "price",
    ax=None,
):
    """Overlay per-L price curves to visualise COS truncation-length stability."""
    fig, ax = _new_ax(ax)
    for L, y in curves_by_L.items():
        ax.plot(strikes, y, marker="o", label=f"L = {L:g}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return fig, ax
