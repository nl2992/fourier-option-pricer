# Packaging and Release

This document describes how `fourier-option-pricer` is packaged, versioned, and released to PyPI.

---

## Package identity

| Field | Value |
|-------|-------|
| PyPI name | `fourier-option-pricer` |
| Python import name | `foureng` |
| Current version | 0.4.1 |
| Python version support | ≥ 3.9 |
| License | MIT |

---

## Project layout

```
pyproject.toml          # PEP 517 build config (setuptools backend)
foureng/                # source package
  __init__.py           # public API + __all__
  models/               # 20 characteristic-function models
  pricers/              # carr_madan / frft / cos / lewis
  utils/                # grids, cumulants, implied vol, spectral filters
  mc/                   # Monte Carlo baselines
  pipeline.py           # unified price_strip dispatcher
  iv/                   # implied volatility routines
```

---

## Install from PyPI

```bash
pip install fourier-option-pricer          # latest release
pip install "fourier-option-pricer==0.4.1" # pin to this release
```

---

## Install from source (development)

```bash
git clone https://github.com/nl2992/fourier-option-pricer.git
cd fourier-option-pricer
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, pytest-cov, and notebook dependencies.

---

## Runtime dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.10 | Numerical integration, special functions |
| pyfeng | ≥ 0.4.0 | CF backends for 8 PyFENG-backed models |

---

## Running the test suite

```bash
# Fast CI-style suite (excludes slow and notebook tests):
pytest -q -m "not slow"

# Full suite including Monte Carlo and notebook execution guards:
pytest -q

# Paper-replication tests only:
pytest -q -m "paper"

# Software-reference tests only (MathWorks Bates):
pytest -q -m "software_reference"
```

---

## Build and publish checklist

1. Bump the version in `pyproject.toml`.
2. Update `CHANGELOG.md` (if maintained).
3. Run the full test suite and confirm all tests pass:
   ```bash
   pytest -q
   ```
4. Build distribution artefacts:
   ```bash
   python -m build
   ```
5. Check the artefacts:
   ```bash
   twine check dist/*
   ```
6. Upload to TestPyPI first:
   ```bash
   twine upload --repository testpypi dist/*
   ```
7. Install from TestPyPI and smoke-test:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ fourier-option-pricer
   python -c "import foureng as fe; print(fe.__version__)"
   ```
8. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```
9. Tag the release in git:
   ```bash
   git tag v0.4.1 && git push origin v0.4.1
   ```

---

## CI / GitHub Actions

The CI workflow (`.github/workflows/`) runs the fast test suite on push to `main` and on
all pull requests. The matrix covers Python 3.9, 3.10, 3.11, and 3.12 on Ubuntu.

pyfeng 0.4.x is pinned in the CI environment. Tests that depend on PyFENG (Bates, Heston,
SV32, VG, CGMY, NIG, OUSV, Rough Heston) may fail locally if only pyfeng 0.3.x is
installed — this is expected and does not indicate a code bug.
