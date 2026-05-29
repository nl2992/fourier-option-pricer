"""Verify that documentation files cross-referenced in key docs actually exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

# Documents that must exist (baseline set).
_REQUIRED_DOCS = [
    "README.md",
    "docs/paper_validation_matrix.md",
    "docs/validation_hierarchy.md",
    "docs/model_zoo.md",
    "docs/numerical_notes.md",
    "docs/current_capability_snapshot.md",
]


@pytest.mark.parametrize("relpath", _REQUIRED_DOCS)
def test_required_doc_exists(relpath):
    assert (ROOT / relpath).exists(), f"Required doc missing: {relpath}"


def test_readme_mentions_price_strip():
    text = (ROOT / "README.md").read_text()
    assert "price_strip" in text


def test_readme_mentions_model_count():
    text = (ROOT / "README.md").read_text()
    # Should mention some model count or model zoo
    assert any(word in text.lower() for word in ["model", "heston", "bsm"])


def test_validation_hierarchy_doc_has_levels():
    text = (DOCS / "validation_hierarchy.md").read_text()
    # Accepts either "L1" / "Level 1" style notation
    has_ln = any(f"L{i}" in text for i in range(1, 6))
    has_level_n = any(f"Level {i}" in text for i in range(1, 6))
    assert has_ln or has_level_n, "validation_hierarchy.md has no evidence-level notation"


def test_model_zoo_lists_heston():
    text = (DOCS / "model_zoo.md").read_text()
    assert "heston" in text.lower()


def _extract_md_links(text: str) -> list[str]:
    """Extract relative file paths from markdown links like [label](path)."""
    return re.findall(r"\[(?:[^\]]+)\]\(([^)#http][^)]*)\)", text)


def test_validation_matrix_internal_links_resolve():
    text = (DOCS / "paper_validation_matrix.md").read_text()
    for link in _extract_md_links(text):
        target = (DOCS / link).resolve()
        if not str(target).endswith(".py"):  # skip test file refs
            # Only check docs-to-docs links
            if "docs/" in link or link.endswith(".md"):
                assert target.exists(), f"Broken link in validation matrix: {link}"
