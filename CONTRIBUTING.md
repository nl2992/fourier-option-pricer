# Contributing

## Local setup

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Required checks before pushing

```bash
ruff check foureng/ tests/
ruff format --check foureng/ tests/
python -m mypy foureng
python -m pytest -q -m "not slow"
```

## Optional checks

```bash
python -m pip install -e ".[test,notebook]"
python -m pytest -q tests/features/test_paper_replication_notebooks_execute.py

python -m pip install -e ".[bench]"
rm -f benchmarks/results/pyperf.json
python benchmarks/pyperf_canonical_cases.py
```

## Contribution guidelines

- Keep new models and pricers in their own modules with clear separation of concerns.
- Prefer vectorized NumPy code over Python loops for strip pricing paths.
- Add tests for invariants, reductions, and paper anchors when introducing a numerical change.
- Keep benchmark-sensitive changes measurable with either saved outputs or the `pyperf` harness.
- Do not remove existing validation assets without replacing them with equivalent coverage.
