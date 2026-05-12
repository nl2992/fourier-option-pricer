"""Smoke tests: paper-replication notebooks must execute without errors.

Both tests are marked @pytest.mark.slow and are skipped in the default
fast-CI matrix. Run them explicitly with ``pytest -m slow``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_bates_replication_notebook_executes():
    nb = Path("notebooks/paper_replications/bates_mathworks_replication.ipynb")
    assert nb.exists(), f"Notebook not found: {nb}"
    subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=180", str(nb),
            "--output", "/tmp/bates_mathworks_replication.executed.ipynb",
        ],
        check=True,
    )


@pytest.mark.slow
def test_three_halves_replication_notebook_executes():
    nb = Path("notebooks/paper_replications/three_halves_replication.ipynb")
    assert nb.exists(), f"Notebook not found: {nb}"
    subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=180", str(nb),
            "--output", "/tmp/three_halves_replication.executed.ipynb",
        ],
        check=True,
    )
