"""Fixtures shared by paper-replication tests."""

from __future__ import annotations

import pytest

from tests._reference_loaders import load_paper_case, load_paper_reference


@pytest.fixture
def paper_reference():
    return load_paper_reference


@pytest.fixture
def paper_case():
    return load_paper_case
