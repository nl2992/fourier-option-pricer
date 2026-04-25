"""Rebuild the submission notebooks from their source builders.

Run from the repo root:
    python3 scripts/build_submission_notebooks.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "build_presentation_notebook.py",
    ROOT / "scripts" / "build_cos_method_improved_notebook.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"[build] {script.name}")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("[done] submission notebooks regenerated")


if __name__ == "__main__":
    main()
