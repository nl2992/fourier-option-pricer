"""Shared helpers for notebook builder and patch scripts."""
from __future__ import annotations

import json
from itertools import count
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:  # pragma: no cover - optional builder convenience only
    nbf = None


_CELL_IDS = count()


def _cell_id() -> str:
    return f"cell-{next(_CELL_IDS)}"


def md(source: str):
    """Create a markdown cell with stable fallback behaviour."""
    text = source.strip("\n")
    if nbf is not None:
        return nbf.v4.new_markdown_cell(text)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": _cell_id(),
        "source": text.splitlines(keepends=True),
    }


def code(source: str):
    """Create a code cell with stable fallback behaviour."""
    text = source.strip("\n")
    if nbf is not None:
        return nbf.v4.new_code_cell(text)
    return {
        "cell_type": "code",
        "metadata": {},
        "id": _cell_id(),
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(
    cells,
    *,
    python_version: str = "3.10.0",
    display_name: str = "Python 3",
    kernel_name: str = "python3",
):
    """Build a notebook payload with a consistent metadata block."""
    metadata = {
        "kernelspec": {
            "display_name": display_name,
            "language": "python",
            "name": kernel_name,
        },
        "language_info": {"name": "python", "version": python_version},
    }
    if nbf is not None:
        nb = nbf.v4.new_notebook(cells=cells)
        nb["metadata"].update(metadata)
        return nb
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata,
        "cells": list(cells),
    }


def read_notebook(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_notebook(path: str | Path, nb: dict) -> None:
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
