"""Shared runtime helpers used across the project notebooks."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import time
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .columbia import DARK, NAVY


def timeit_strip(fn, *args, n_repeat: int = 3, warmup: bool = True, **kwargs):
    """Return ``(value, best_ms)`` for a strip-style pricing call."""
    if warmup:
        fn(*args, **kwargs)
    best_ms = float("inf")
    out = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1e3)
    return out, best_ms


def timed_median_ms(fn, *args, n_rep: int = 5, warmup: bool = True, **kwargs):
    """Return ``(value, median_ms)`` for a callable."""
    if warmup:
        fn(*args, **kwargs)
    times = []
    out = None
    for _ in range(n_rep):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1e3)
    return out, float(np.median(times))


def timed_call(fn, repeats: int = 5, warmup: bool = True):
    """Notebook-friendly wrapper for zero-argument callables."""
    return timed_median_ms(fn, n_rep=repeats, warmup=warmup)


def sci(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2e}"


def error_zoom_bounds(
    *arrays, pad_frac: float = 0.08, min_pad: float = 5e-6
) -> tuple[float, float]:
    """Tight linear y-limits for clustered error curves."""
    values = [np.ravel(np.asarray(arr, dtype=float)) for arr in arrays if np.asarray(arr).size]
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    pad = max(pad_frac * (hi - lo), min_pad)
    return max(0.0, lo - pad), hi + pad


def night_style(
    df: pd.DataFrame,
    *,
    caption: str | None = None,
    formats: dict[str, str | Callable] | None = None,
    highlight_min: Iterable[str] | None = None,
    highlight_max: Iterable[str] | None = None,
    hide_index: bool = True,
):
    """Dark Columbia-styled table used in the demo notebook."""
    soft = "#D7EAF5"
    mid = "#7FA3C7"
    styler = df.style
    if formats:
        styler = styler.format(formats)
    if highlight_min:
        styler = styler.highlight_min(
            subset=list(highlight_min),
            color=soft,
            props="color: #0B1F3A; font-weight: bold;",
        )
    if highlight_max:
        styler = styler.highlight_max(
            subset=list(highlight_max),
            color=mid,
            props="color: #0B1F3A; font-weight: bold;",
        )
    styler = styler.set_properties(
        **{
            "background-color": DARK,
            "color": "#F8FAFC",
            "border": "1px solid #4B6FA8",
            "font-size": "11px",
        }
    )
    styler = styler.set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("font-family", "Menlo, Monaco, monospace"),
                ],
            },
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("color", NAVY),
                    ("font-size", "13px"),
                    ("font-weight", "bold"),
                    ("padding", "6px 0"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", NAVY),
                    ("color", "#F8FAFC"),
                    ("border", "1px solid #4B6FA8"),
                    ("padding", "6px 8px"),
                ],
            },
            {"selector": "td", "props": [("padding", "6px 8px")]},
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#10294B")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#0B1F3A")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#1F5CA6")],
            },
        ],
        overwrite=False,
    )
    if caption is not None:
        styler = styler.set_caption(caption)
    if hide_index:
        styler = styler.hide(axis="index")
    return styler


def style_table(
    df: pd.DataFrame,
    caption: str,
    *,
    gradient_cols: Iterable[str] | None = None,
    highlight_col: str | None = None,
):
    """Light Columbia-styled table used in the presentation notebooks."""
    styler = (
        df.style.format(precision=4, thousands=",")
        .set_caption(caption)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("color", NAVY),
                        ("font-weight", "700"),
                        ("font-size", "13.5px"),
                        ("padding", "4px 0 10px 0"),
                        ("text-align", "left"),
                        ("caption-side", "top"),
                    ],
                },
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "separate"),
                        ("border-spacing", "0"),
                        ("margin", "4px 0 10px 0"),
                        ("width", "auto"),
                        ("font-size", "12.5px"),
                        ("color", "#1a2230"),
                        ("border", "1px solid #CAD4E3"),
                        ("border-radius", "6px"),
                        ("overflow", "hidden"),
                    ],
                },
                {
                    "selector": "thead th",
                    "props": [
                        ("background", NAVY),
                        ("color", "white"),
                        ("font-weight", "700"),
                        ("padding", "8px 12px"),
                        ("border", "none"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "tbody th",
                    "props": [
                        ("background", "white"),
                        ("color", NAVY),
                        ("font-weight", "700"),
                        ("padding", "6px 10px"),
                        ("border-right", "2px solid #012169"),
                        ("border-bottom", "1px solid #E1E7EF"),
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("padding", "6px 12px"),
                        ("border-bottom", "1px solid #E1E7EF"),
                        ("color", "#1a2230"),
                    ],
                },
                {"selector": "tbody tr:nth-child(odd)  td", "props": [("background", "#FFFFFF")]},
                {"selector": "tbody tr:nth-child(even) td", "props": [("background", "#EAF0F8")]},
                {"selector": "tbody tr:nth-child(odd)  th", "props": [("background", "#FFFFFF")]},
                {"selector": "tbody tr:nth-child(even) th", "props": [("background", "#EAF0F8")]},
                {"selector": "tbody tr:hover td", "props": [("background", "#D6E2F2")]},
            ]
        )
    )
    if gradient_cols:
        styler = styler.background_gradient(
            cmap="Blues",
            subset=list(gradient_cols),
            axis=None,
            low=0.02,
            high=0.85,
        )
    if highlight_col:
        styler = styler.apply(
            lambda s: [
                "font-weight:700;color:#012169" if c == highlight_col else "" for c in s.index
            ],
            axis=0,
        )
    return styler


def iter_repo_candidates():
    """Yield plausible repo roots based on cwd, sys.path, and installed foureng."""
    seen = set()

    def _expand(pathlike):
        if not pathlike:
            return []
        path = pathlib.Path(pathlike).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        return [path, *path.parents]

    candidates = []
    for raw in (os.environ.get("PWD"), os.environ.get("OLDPWD")):
        candidates.extend(_expand(raw))
    try:
        candidates.extend(_expand(pathlib.Path.cwd()))
    except FileNotFoundError:
        pass
    for raw in sys.path:
        if raw:
            candidates.extend(_expand(raw))
    spec = importlib.util.find_spec("foureng")
    if spec is not None and spec.origin:
        candidates.extend(_expand(spec.origin))

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def locate_repo_root(
    *,
    required_paths: tuple[str, ...] = ("foureng", "benchmarks"),
    home_roots: tuple[str, ...] = ("Desktop", "Documents", "Projects", "Code"),
) -> pathlib.Path:
    """Locate the repo root for notebooks launched from mixed environments."""
    for candidate in iter_repo_candidates():
        if all((candidate / name).exists() for name in required_paths):
            return candidate

    for root_name in home_roots:
        base = pathlib.Path.home() / root_name
        if not base.exists():
            continue
        try:
            for match in base.rglob("foureng/__init__.py"):
                candidate = match.parent.parent
                if all((candidate / name).exists() for name in required_paths):
                    return candidate
        except Exception:
            continue

    required = ", ".join(required_paths)
    raise RuntimeError(
        f"Could not locate repo root containing: {required}. "
        "Launch the notebook from inside the project or install the package first."
    )


def ensure_repo_root_on_path(
    *,
    required_paths: tuple[str, ...] = ("foureng", "benchmarks"),
) -> pathlib.Path:
    """Locate the repo root and prepend it to ``sys.path`` if needed."""
    root = locate_repo_root(required_paths=required_paths)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
