# Packaging and Release

This document describes how `fourier-option-pricer` is packaged, versioned, and released to PyPI.

---

## Package identity

| Field | Value |
|-------|-------|
| PyPI name | `fourier-option-pricer` |
| Python import name | `foureng` |
| Version source | `foureng/_version.py` (single source of truth; `pyproject.toml` reads it) |
| Python version support | ≥ 3.10 (CI: 3.10 – 3.14) |
| License | MIT (`LICENSE` ships in both the sdist and the wheel) |
| Typing | PEP 561 `py.typed` marker included |

---

## Project layout

```
pyproject.toml          # PEP 517/621/639 build config (setuptools backend)
foureng/                # source package
  __init__.py           # public API + __all__
  _version.py           # __version__
  py.typed              # PEP 561 marker
  models/               # 27 characteristic-function models
  pricers/              # COS, Carr-Madan, FRFT, Hilbert, CONV, Lewis, PROJ, CTMC, ...
  products/             # payoff dataclasses (barriers, Asians, cliquets, ...)
  analytics/            # closed forms (BSM exotics, Levy credit, variance)
  surface/              # SVI / SSVI / Dupire / calibration
  mc/                   # Monte Carlo engines and control variates
  greeks/  iv/  utils/  core/
  viz/                  # optional; needs the [viz] extra
  pipeline.py           # unified price / price_strip dispatcher
```

---

## Install from PyPI

```bash
pip install fourier-option-pricer          # core: numpy, scipy, pyfeng (+ statsmodels)
pip install "fourier-option-pricer[viz]"   # + matplotlib, pandas for foureng.viz
```

---

## Install from source (development)

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
python -m pip install -e ".[dev]"
```

The `[dev]` extra pulls in every other extra (`viz`, `test`, `notebook`, `typecheck`,
`bench`) plus `nbmake`, `build`, `twine`, and `ruff`.

---

## Runtime dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| numpy | ≥ 1.26 | Array operations |
| scipy | ≥ 1.10 | Numerical integration, special functions, optimisation |
| pyfeng | ≥ 0.4.0 | CF backends for 8 PyFENG-backed models |
| statsmodels | ≥ 0.14 | Not used by `foureng`; `pyfeng` imports it at import time without declaring it |

`matplotlib` and `pandas` are optional (`[viz]`) and are only imported by `foureng.viz`
and `foureng.experiments`. Minimum versions are exercised in CI through
`constraints/minimum.txt` on the Python 3.10 leg.

---

## Running the test suite

```bash
# Fast CI-style suite (excludes slow and notebook tests):
python -m pip install -e ".[test]"
python -m pytest -q -m "not slow"

# Full suite including Monte Carlo and notebook execution guards:
python -m pip install -e ".[test,notebook]"
python -m pytest -q

# Paper-replication tests only:
python -m pytest -q -m "paper"

# Software-reference tests only (MathWorks Bates):
python -m pytest -q -m "software_reference"
```

---

## Release checklist

Releases are uploaded by `.github/workflows/publish.yml` using PyPI Trusted Publishing
(OIDC, no API token). Do not upload with `twine` by hand.

1. Bump `__version__` in `foureng/_version.py`.
2. Add a dated section to `CHANGELOG.md`; update `version` / `date-released` in `CITATION.cff`.
3. Run the local gate:
   ```bash
   ruff check foureng tests && ruff format --check foureng tests
   mypy foureng
   pytest -q -m "not slow"
   python -m build && twine check --strict dist/*
   ```
4. Merge to `main`, then create a GitHub Release with tag `v<version>` (for example
   `v0.21.0`). Publishing the Release triggers the workflow, which:
   - fails if the tag does not equal `v` + `foureng/_version.py`,
   - builds the sdist and wheel and runs `twine check --strict`,
   - installs the wheel into a clean venv and prices a Heston strip (exercising the pyfeng backend) as a smoke test,
   - runs the fast test suite,
   - uploads to PyPI from the `pypi` environment (with PEP 740 attestations).
5. Verify the release installs cleanly:
   ```bash
   pip install --no-cache-dir "fourier-option-pricer==<version>"
   python -c "import foureng as fe; print(fe.__version__)"
   ```

---

## CI / GitHub Actions

`ci.yml` runs on pushes to `main`/`dev` and on pull requests to `main`: ruff lint and format,
mypy, the fast test suite on Python 3.10 (minimum dependency pins) through 3.14, the
paper/reference tests, and a packaging job that builds, inspects, and smoke-tests the wheel
with core dependencies only. `slow_tests.yml` runs the full suite weekly. Dependabot keeps
the GitHub Actions versions current.
