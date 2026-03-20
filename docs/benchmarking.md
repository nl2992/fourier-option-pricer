# Benchmarking

The repository includes a small `pyperf` benchmark harness for canonical
pricing workloads. This makes the efficiency story testable as an engineering
process rather than only as a one-off report result.

## Canonical benchmark cases

- `heston_cos_improved_strip_21`: 21-strike Heston strip through the adaptive improved-COS path
- `vg_carr_madan_strip_25`: 25-strike Variance-Gamma strip through Carr-Madan FFT
- `fo2008_heston_cos_improved_5`: 5-strike FO2008-style Heston benchmark case

## Local usage

```bash
python -m pip install -e ".[bench]"
mkdir -p benchmarks/results
rm -f benchmarks/results/pyperf.json
python benchmarks/pyperf_canonical_cases.py
python benchmarks/pyperf_canonical_cases.py -o benchmarks/results/pyperf.json
```

## Suggested workflow

1. Run the benchmark harness before a release or major numerical rewrite.
2. Save the `pyperf` JSON output as an artifact.
3. Compare results against earlier runs using `pyperf compare_to`.
4. Investigate regressions before merging a performance-sensitive change.

## Scope

These benchmarks are intentionally **not** part of the main fast CI path.
They are intended for manual regression tracking or for a dedicated benchmark
workflow so that correctness CI remains fast and reliable.
