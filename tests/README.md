# Test Suite Organization

The suite is grouped by validation purpose. No tests should be removed during
reorganization; use `python -m pytest --collect-only -q` to compare collection
counts before and after any move.

| Folder | Purpose |
| --- | --- |
| `papers/` | Tests that replicate or benchmark against published-paper examples, especially Carr-Madan, Lewis, Fang-Oosterlee COS, FRFT, and Kou/COS workflows. |
| `models/` | Model adapter, regression-strip, and reduction-limit tests for all 18 supported models. Each in-house native-CF model has a paper-backed 3-layer suite: Layer 1 (analytic/limit benchmarks), Layer 2 (cross-engine agreement across Lewis, COS, Carr-Madan, FRFT), and Layer 3 (structural properties and no-arbitrage bounds). |
| `methods/` | Pricing-method behavior tests: COS policy, filters, grid/search logic, cross-method agreement, alpha validity, and broader robustness sweeps. |
| `features/` | Package features and public workflows: Monte Carlo, control variates, implied volatility, calibration, Greeks, public API, PyFENG wrappers, and generalized integration checks. |

`conftest.py` remains at the root so fixtures and reference data are shared by
all groups.
