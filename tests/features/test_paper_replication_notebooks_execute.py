"""Smoke tests: paper-replication notebooks must execute without errors.

Both tests are marked @pytest.mark.slow and are skipped in the default
fast-CI matrix. Run them explicitly with ``pytest -m slow``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest


def _execute_notebook_in_current_env(nb: Path, output_path: str) -> None:
    kernel_name = f"foureng-ci-{uuid.uuid4().hex[:8]}"
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmpdir:
        jupyter_path = str(Path(tmpdir) / "share" / "jupyter")
        env["JUPYTER_DATA_DIR"] = jupyter_path
        env["JUPYTER_PATH"] = jupyter_path
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ipykernel",
                "install",
                "--prefix",
                tmpdir,
                "--name",
                kernel_name,
                "--display-name",
                kernel_name,
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                f"--ExecutePreprocessor.kernel_name={kernel_name}",
                "--ExecutePreprocessor.timeout=180",
                str(nb),
                "--output",
                output_path,
            ],
            check=True,
            env=env,
        )


@pytest.mark.slow
def test_bates_replication_notebook_executes():
    pytest.importorskip(
        "ipykernel", reason="ipykernel not installed; skipping notebook execution tests"
    )
    nb = Path("notebooks/paper_replications/bates_mathworks_replication.ipynb")
    assert nb.exists(), f"Notebook not found: {nb}"
    _execute_notebook_in_current_env(nb, "/tmp/bates_mathworks_replication.executed.ipynb")


@pytest.mark.slow
def test_three_halves_replication_notebook_executes():
    pytest.importorskip(
        "ipykernel", reason="ipykernel not installed; skipping notebook execution tests"
    )
    nb = Path("notebooks/paper_replications/three_halves_replication.ipynb")
    assert nb.exists(), f"Notebook not found: {nb}"
    _execute_notebook_in_current_env(nb, "/tmp/three_halves_replication.executed.ipynb")
