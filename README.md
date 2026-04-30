# fourier-option-pricer

Course project and research repo for Fourier-based European option pricing under characteristic-function models.

This repository compares Carr-Madan FFT, PyFENG-backed FFT references, COS, and an adaptive filtered-COS extension across diffusion, stochastic-volatility, Levy, and stochastic-volatility-with-jumps settings. It contains the package source, the replication notebooks, and the benchmark artifacts used in the project write-up.

## Repository guide

- `foureng/`: packaged pricing library and public API
- `notebooks/demo.ipynb`: Colab-friendly walkthrough of the main pricing workflow
- `notebooks/presentation_fourier_methods.ipynb`: presentation notebook version
- `notebooks/cos_method_improved.ipynb`: COS truncation and policy notebook
- `benchmarks/`: generated tables and figures used by the notebooks
- `tests/`: regression and public API tests

For the PyPI-facing package README, see [README-II.md](README-II.md).

## Methods in scope

| Method | Role in the repo |
|--------|-------------------|
| Carr-Madan FFT | Baseline transform method for pricing a strike strip on a fixed frequency grid |
| PyFENG FFT reference | External oracle for supported models in benchmarking and notebook comparisons |
| COS | Main Fourier-cosine pricing method |
| Improved COS | COS with model-aware truncation and adaptive grid selection |
| Adaptive filtered-COS | Extension that searches across no-filter and filtered COS policies under a tolerance target |

## Model families

| Family | Models |
|--------|--------|
| Pure diffusion | Black-Scholes-Merton |
| Stochastic volatility | Heston, OU-SV |
| Pure jump / Levy | Variance Gamma, NIG, CGMY |
| Jump diffusion | Kou |
| SV + jumps | Bates, Heston-Kou, Heston-CGMY |

## Project note

The main extension in this repo is the adaptive filtered-COS layer. Junike-style truncation improves interval selection, but short maturities and jump-heavy models can still show finite-series ringing. The extension keeps plain improved COS in the candidate set, then tests a small family of spectral filters and selects the cheapest candidate that still meets the target tolerance.

## References

- Carr, P., and Madan, D. (1999), *Option valuation using the fast Fourier transform*, Journal of Computational Finance.
- Lewis, A. L. (2001), *A simple option formula for general jump-diffusion and other exponential Levy processes*.
- Fang, F., and Oosterlee, C. W. (2008), *A novel pricing method for European options based on Fourier-cosine series expansions*.
- Junike, G., and Pankrashkin, K. (2022), *Precise option pricing by the COS method*.
- Ruijter, M. J., Versteegh, M., and Oosterlee, C. W. (2015), *On the application of spectral filters in a Fourier option pricing technique*.
