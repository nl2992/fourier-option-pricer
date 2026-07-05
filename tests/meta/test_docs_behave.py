"""Docs-behavior guards: examples must run and documented signatures must match.

Coverage audits (every symbol appears somewhere) live in the other meta
tests; these tests check that what the docs *say* is what the code *does*:
- every ```python block in README.md executes (blocks share one namespace,
  since later snippets continue earlier ones);
- every `name(args)` signature quoted in docs/api_reference.md uses only
  parameter names that exist on the real callable.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import foureng

ROOT = Path(__file__).resolve().parents[2]


def test_readme_python_blocks_execute():
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "README has no python blocks?"
    namespace: dict = {}
    for i, block in enumerate(blocks):
        try:
            exec(block, namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - the assert carries info
            raise AssertionError(f"README python block {i} failed: {exc}\n{block}") from exc


def test_api_reference_signatures_match_code():
    api = (ROOT / "docs" / "api_reference.md").read_text()
    mismatches = []
    for m in re.finditer(r"`(\w+)\(([^)`]*)\)`", api):
        name, doc_args = m.group(1), m.group(2)
        fn = getattr(foureng, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            real = set(inspect.signature(fn).parameters)
        except (ValueError, TypeError):
            continue
        doc_names = [
            a.split("=")[0].strip().lstrip("*")
            for a in doc_args.split(",")
            if a.strip() and a.strip() != "..."
        ]
        wrong = [d for d in doc_names if d and d not in real and d.rstrip(".") not in real]
        if wrong:
            mismatches.append(
                f"{name}: documented args {wrong} not in real signature {sorted(real)}"
            )
    assert not mismatches, "api_reference.md signatures drifted:\n" + "\n".join(mismatches)
