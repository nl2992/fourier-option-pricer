"""Shared loaders for versioned paper-replication fixtures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIXTURE_ROOT = ROOT / "benchmarks" / "paper_replications"


@dataclass(frozen=True)
class PaperReference:
    """Loaded metadata plus tabular rows for one paper fixture."""

    paper_key: str
    metadata: dict[str, str]
    rows: tuple[dict[str, str], ...]
    data_file: Path

    @property
    def label(self) -> str:
        table = self.metadata.get("table", "")
        citation = self.metadata.get("citation", self.paper_key)
        return f"{citation} / {table}".strip(" /")

    @property
    def atol(self) -> float:
        value = self.metadata.get("tolerance_abs", "0.0")
        return float(value) if value else 0.0

    @property
    def expected(self) -> list[float]:
        return [
            float(row["expected_price"])
            for row in self.rows
            if row.get("expected_price") not in ("", None)
        ]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_flat_yaml(path: Path) -> dict[str, str]:
    """Read the flat metadata.yaml files without adding a PyYAML dependency."""
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path}: invalid metadata line {raw_line!r}")
        metadata[key.strip()] = _unquote(value)
    return metadata


def _read_csv_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


@lru_cache(maxsize=None)
def load_paper_metadata(paper_key: str) -> dict[str, str]:
    path = PAPER_FIXTURE_ROOT / paper_key / "metadata.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No paper metadata fixture at {path}")
    return _read_flat_yaml(path)


@lru_cache(maxsize=None)
def load_paper_rows(
    paper_key: str, filename: str = "expected_prices.csv"
) -> tuple[dict[str, str], ...]:
    path = PAPER_FIXTURE_ROOT / paper_key / filename
    if not path.exists():
        raise FileNotFoundError(f"No paper row fixture at {path}")
    return _read_csv_rows(path)


def load_paper_reference(paper_key: str, filename: str = "expected_prices.csv") -> PaperReference:
    return PaperReference(
        paper_key=paper_key,
        metadata=load_paper_metadata(paper_key),
        rows=load_paper_rows(paper_key, filename),
        data_file=PAPER_FIXTURE_ROOT / paper_key / filename,
    )


def load_paper_case(
    paper_key: str,
    case_id: str,
    filename: str = "expected_prices.csv",
) -> PaperReference:
    reference = load_paper_reference(paper_key, filename)
    rows = tuple(row for row in reference.rows if row.get("case_id") == case_id)
    if not rows:
        raise KeyError(f"{paper_key}/{filename}: no rows with case_id={case_id!r}")
    return PaperReference(reference.paper_key, reference.metadata, rows, reference.data_file)


def numeric_column(rows: tuple[dict[str, str], ...], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw not in ("", None):
            values.append(float(raw))
    return values
