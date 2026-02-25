# References

All papers cited in the foureng codebase, listed by category.
Free-access links point to published versions or freely available preprints.

---

## Fourier Pricing Methods

**Carr, P. & Madan, D. (1999).** Option valuation using the fast Fourier transform.
*Journal of Computational Finance*, 2(4), 61–73.
DOI: [10.21314/JCF.1999.043](https://doi.org/10.21314/JCF.1999.043)
*(Foundational FFT pricer; the damped-call transform implemented in `pricers/carr_madan.py`.)*

**Fang, F. & Oosterlee, C. W. (2008).** A novel pricing method for European options based on Fourier-cosine series expansions.
*SIAM Journal on Scientific Computing*, 31(2), 826–848.
DOI: [10.1137/080718061](https://doi.org/10.1137/080718061)
*(COS method and cumulant-based truncation rule; implemented in `pricers/cos.py` and `utils/cumulants.py`.)*

**Chourdakis, K. (2004).** Option pricing using the fractional FFT.
*Journal of Computational Finance*, 8(2), 1–18.
DOI: [10.21314/JCF.2004.137](https://doi.org/10.21314/JCF.2004.137)
*(FRFT pricer allowing non-uniform log-strike grids; implemented in `pricers/frft.py`.)*

**Junike, G. & Pankrashkin, K. (2022).** Precise option pricing by the COS method — how to choose the truncation range.
*Applied Mathematics and Computation*, 421, 126935.
DOI: [10.1016/j.amc.2022.126935](https://doi.org/10.1016/j.amc.2022.126935)
Preprint: [arXiv:2004.02968](https://arxiv.org/abs/2004.02968)
*(Tolerance-based truncation; drives `cos_adaptive_decision` and `cos_improved` in `pricers/cos.py`.)*

**Ruijter, M. J., Versteegh, M. & Oosterlee, C. W. (2015).** On the application of spectral filters in a Fourier option pricing technique.
*Journal of Computational Finance*, 19(1), 75–106.
DOI: [10.21314/JCF.2015.306](https://doi.org/10.21314/JCF.2015.306)
*(Spectral (Lanczos, exponential) filters for COS; implemented in `utils/spectral_filters.py` and `pricers/filtered_cos.py`.)*

**Lewis, A. L. (2000).** *Option Valuation Under Stochastic Volatility.*
Finance Press, Newport Beach, CA.
*(Semi-closed-form formula via Fourier inversion; used as a benchmark in tests for GARCH and other models. `pricers/lewis.py`.)*

---

## Stochastic Volatility Models

**Heston, S. L. (1993).** A closed-form solution for options with stochastic volatility with applications to bond and currency options.
*Review of Financial Studies*, 6(2), 327–343.
DOI: [10.1093/rfs/6.2.327](https://doi.org/10.1093/rfs/6.2.327)
*(Affine SV CF via Riccati ODE; `models/heston.py`.)*

**Schöbel, R. & Zhu, J. (1999).** Stochastic volatility with an Ornstein–Uhlenbeck process: An extension.
*European Finance Review*, 3(1), 23–46.
DOI: [10.1023/A:1009803506170](https://doi.org/10.1023/A:1009803506170)
*(OU-SV CF; `models/ousv.py`.)*

**Lewis, A. L. (2000).** *Option Valuation Under Stochastic Volatility.*
Finance Press, Newport Beach, CA.
*(3/2 SV model CF derivation; `models/sv32.py`. See also above under Fourier methods.)*

**Carr, P. & Sun, J. (2007).** A new approach for option pricing under stochastic volatility.
*Review of Derivatives Research*, 10(2), 87–150.
DOI: [10.1007/s11147-007-9014-6](https://doi.org/10.1007/s11147-007-9014-6)
*(Alternative 3/2 SV characterisation.)*

**El Euch, O. & Rosenbaum, M. (2019).** The characteristic function of rough Heston models.
*Mathematical Finance*, 29(1), 3–38.
DOI: [10.1111/mafi.12173](https://doi.org/10.1111/mafi.12173)
Preprint: [arXiv:1609.02108](https://arxiv.org/abs/1609.02108)
*(Rough Heston CF via fractional Riccati equations; `models/rough_heston.py`.)*

**Callegaro, G., Grasselli, M. & Pagès, G. (2021).** Fast hybrid schemes for fractional Riccati equations (rough is not so tough).
*Mathematics of Operations Research*, 46(1), 221–254.
DOI: [10.1287/moor.2020.1054](https://doi.org/10.1287/moor.2020.1054)
Preprint: [arXiv:1902.01599](https://arxiv.org/abs/1902.01599)
*(Numerical solver for Rough Heston Riccati equation used by PyFENG.)*

**Nelson, D. B. (1990).** ARCH models as diffusion approximations.
*Journal of Econometrics*, 45(1–2), 7–38.
DOI: [10.1016/0304-4076(90)90092-8](https://doi.org/10.1016/0304-4076(90)90092-8)
*(Theoretical basis for the GARCH diffusion limit; referenced in `models/garch_wmw2012.py`.)*

**Wu, X.-Y., Ma, C.-Q. & Wang, S.-Y. (2012).** Warrant pricing under GARCH diffusion model.
*Economic Modelling*, 29(6), 2237–2244.
DOI: [10.1016/j.econmod.2012.06.013](https://doi.org/10.1016/j.econmod.2012.06.013)
*(Closed-form MGF/CF for the GARCH diffusion; `models/garch_wmw2012.py`.)*

---

## Jump-Diffusion Models

**Merton, R. C. (1976).** Option pricing when underlying stock returns are discontinuous.
*Journal of Financial Economics*, 3(1–2), 125–144.
DOI: [10.1016/0304-405X(76)90022-2](https://doi.org/10.1016/0304-405X(76)90022-2)
*(Log-normal jump-diffusion CF; `models/merton_jd.py`.)*

**Kou, S. G. (2002).** A jump-diffusion model for option pricing.
*Management Science*, 48(8), 1086–1101.
DOI: [10.1287/mnsc.48.8.1086.166](https://doi.org/10.1287/mnsc.48.8.1086.166)
*(Double-exponential jump sizes; `models/kou.py`.)*

**Bates, D. S. (1996).** Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche mark options.
*Review of Financial Studies*, 9(1), 69–107.
DOI: [10.1093/rfs/9.1.69](https://doi.org/10.1093/rfs/9.1.69)
*(Heston + log-normal jumps (SVJ); `models/bates.py`.)*

---

## Pure-Jump Lévy Models

**Madan, D. B., Carr, P. & Chang, E. C. (1998).** The Variance Gamma process and option pricing.
*European Finance Journal*, 2(1), 79–105.
DOI: [10.1023/A:1009703431535](https://doi.org/10.1023/A:1009703431535)
*(VG CF as a time-changed Brownian motion; `models/variance_gamma.py`.)*

**Carr, P., Geman, H., Madan, D. B. & Yor, M. (2002).** The fine structure of asset returns: An empirical investigation.
*Journal of Business*, 75(2), 305–332.
DOI: [10.1086/338705](https://doi.org/10.1086/338705)
*(CGMY Lévy CF; `models/cgmy.py` and `models/heston_cgmy.py`.)*

**Barndorff-Nielsen, O. E. (1997).** Normal inverse Gaussian distributions and stochastic volatility modelling.
*Scandinavian Journal of Statistics*, 24(1), 1–13.
DOI: [10.1111/1467-9469.00045](https://doi.org/10.1111/1467-9469.00045)
*(NIG CF; `models/nig.py`.)*

**Carr, P. & Wu, L. (2003).** The finite moment log stable process and option pricing.
*Journal of Finance*, 58(2), 753–777.
DOI: [10.1111/1540-6261.00544](https://doi.org/10.1111/1540-6261.00544)
*(FMLS CF for asset return distributions with finite upper moments; `models/fmls.py`.)*

**Schoutens, W. & Teugels, J. L. (1998).** Lévy processes, polynomials and martingales.
*Communications in Statistics — Stochastic Models*, 14(1–2), 335–349.
DOI: [10.1080/15326349808807475](https://doi.org/10.1080/15326349808807475)
*(Foundational Meixner process paper.)*

**Schoutens, W. (2002).** The Meixner process: Theory and applications in finance.
*EURANDOM Report 2002-004*, EURANDOM, Eindhoven.
Free access: [EURANDOM](https://www.eurandom.tue.nl/reports/2002/004-report.pdf)
*(Meixner CF and martingale correction; `models/meixner.py`.)*

**Küchler, U. & Tappe, S. (2008).** Bilateral Gamma distributions and processes in financial mathematics.
*Stochastic Processes and their Applications*, 118(2), 261–283.
DOI: [10.1016/j.spa.2007.04.006](https://doi.org/10.1016/j.spa.2007.04.006)
*(Bilateral Gamma CF; `models/bilateral_gamma.py`.)*

**Barndorff-Nielsen, O. E. (1977).** Exponentially decreasing distributions for the logarithm of particle size.
*Proceedings of the Royal Society of London A*, 353(1674), 401–419.
DOI: [10.1098/rspa.1977.0041](https://doi.org/10.1098/rspa.1977.0041)
*(Generalised hyperbolic (GH) distribution; `models/generalized_hyperbolic.py`.)*

**Eberlein, E. & Keller, U. (1995).** Hyperbolic distributions in finance.
*Bernoulli*, 1(3), 281–299.
DOI: [10.2307/3318481](https://doi.org/10.2307/3318481)
*(Hyperbolic Lévy motion as a special case of GH; referenced in `models/generalized_hyperbolic.py`.)*

**Carr, P., Geman, H., Madan, D. B. & Yor, M. (2003).** Stochastic volatility for Lévy processes.
*Mathematical Finance*, 13(3), 345–382.
DOI: [10.1111/1467-9965.00020](https://doi.org/10.1111/1467-9965.00020)
*(VGSA CF via CIR Laplace transform of the VG Lévy exponent; `models/vgsa.py`.)*

---

## Multi-Factor and Two-Factor SV Models

**Christoffersen, P., Heston, S. & Jacobs, K. (2009).** The shape and term structure of the index option smirk: Why multifactor stochastic volatility models work so well.
*Management Science*, 55(12), 1914–1932.
DOI: [10.1287/mnsc.1090.1065](https://doi.org/10.1287/mnsc.1090.1065)
*(Two independent Heston variance factors; `models/double_heston.py`. CF factorises as a product of two single-Heston CFs.)*

---

## Validation References

**Baldeaux, J. & Badran, A. (2012).** Consistent modelling of VIX and equity derivatives using a 3/2 plus jumps model.
*Applied Mathematical Finance*, 21(4), 299–312.
DOI: [10.1080/1350486X.2013.868631](https://doi.org/10.1080/1350486X.2013.868631)
*(3/2 SV parameter set used for qualitative figure replication and no-arbitrage smoke tests; `tests/refs/sv32_baldeaux_badran_figure_params.json` and `tests/papers/test_phase6_sv32_baldeaux_badran_smoke.py`.)*

---

## Monte Carlo Methods

**Andersen, L. B. G. (2008).** Simple and efficient simulation of the Heston stochastic volatility model.
*Journal of Computational Finance*, 11(3), 1–42.
DOI: [10.21314/JCF.2008.189](https://doi.org/10.21314/JCF.2008.189)
*(Quadratic-exponential (QE) scheme for Heston; `mc/heston_conditional_mc.py` via PyFENG `HestonMcAndersen2008`.)*

**Glasserman, P. & Kim, K.-K. (2011).** Gamma expansion of the Heston stochastic volatility model.
*Finance and Stochastics*, 15(2), 267–296.
DOI: [10.1007/s00780-009-0115-y](https://doi.org/10.1007/s00780-009-0115-y)
*(Exact gamma-series draw of integrated variance; used in `mc/control_variate.py` via PyFENG `HestonMcGlassermanKim2011`.)*

**Ballotta, L. & Kyriakou, I. (2014).** Monte Carlo simulation of the CGMY process and option pricing.
*Journal of Futures Markets*, 34(12), 1095–1121.
DOI: [10.1002/fut.21647](https://doi.org/10.1002/fut.21647)
*(Simulation methods for CGMY; background reference for `mc/` module.)*

---

## Reference Textbooks

**Cont, R. & Tankov, P. (2004).** *Financial Modelling with Jump Processes.*
Chapman & Hall / CRC Press. ISBN 978-1-584-88413-2.
*(Comprehensive reference for Lévy process CFs, cumulants, and martingale conditions; cited in `models/merton_jd.py` and `models/meixner.py`.)*

**Glasserman, P. (2004).** *Monte Carlo Methods in Financial Engineering.*
Springer. ISBN 978-0-387-00451-8.
*(Standard reference for variance reduction (control variates); cited in `mc/control_variate.py`.)*
