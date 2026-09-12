"""Guard against version metadata drifting between release files."""

from __future__ import annotations

import re
from pathlib import Path

import foureng

ROOT = Path(__file__).resolve().parents[2]


def test_citation_version_matches_package():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    assert match is not None, "CITATION.cff has no version field"
    assert match.group(1) == foureng.__version__


def test_changelog_has_entry_for_package_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(foureng.__version__)}\b", text, flags=re.MULTILINE), (
        f"CHANGELOG.md has no '## {foureng.__version__}' section"
    )


def test_py_typed_marker_present():
    assert (Path(foureng.__file__).parent / "py.typed").is_file()
