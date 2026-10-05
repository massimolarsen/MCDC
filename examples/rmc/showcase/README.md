# CE-RMC in MC/DC: showcase figures

Figures that show what the port of continuous-energy Residual Monte Carlo (RMC) into
MC/DC adds over the dissertation code, made from results that already exist.
`make_plots.py` only reads results and runs no simulation:
- the example outputs (`../<problem>/output*.h5`);
- the kernel verification statistics (`mcdc/rmc/writeups/figures/*/stats.json`);
- the OSU HPC timing tables copied into `hpc_timing/` from `../efficiency`.

```
python make_plots.py
```

## The figures

| Figure | What it shows |
|---|---|
| `architecture.png` | How RMC sits on MC/DC. MC/DC's nuclear data, samplers, fixed-source transport (Numba, MPI) and track-length tallies are reused. The new `mcdc.rmc` package adds the inverted kernels, the cached transfer moments, the residual source and the update. Core MC/DC changes are small hooks: energy window, all-prompt fission, sign-safe weights, and the new moment tally scores. |
| `kernel_verification.png` | Each ENDF law that RMC evaluates as a density f(E → E′, μ₀), checked against MC/DC's own sampler: 22 tests with 10⁶ samples each. All χ² values are within ±2 of the expected value. The yields agree to 10⁻⁷–10⁻³, with quadrature errors of 10⁻¹⁴ to 5×10⁻³. The laws include anisotropic CM elastic, levels, Kalbach–Mann, Law 61, N-body, evaporation, Maxwellian, fission, and free gas for H-1 and U-238. |
| `convergence_gallery.png` | ‖ε̃‖ per iteration for 5 synthetic problems with analytic references and 5 problems with real ENDF/B-VIII.1 data. The synthetic problems converge geometrically, falling 4–19 decades in their 10–30 iterations. The real-data problems level off at the statistical noise of one iteration. |
| `linear_basis.png` | The linear discontinuous energy basis against the dissertation's constant basis on the A = 1 absorber, at 100 histories per bin. The constant basis stops at its Galerkin bias (5×10⁻⁸ to 8×10⁻⁷). The linear basis goes on to 4–6×10⁻¹⁰, 80–1900× lower depending on G. |
| `flux_gallery.png` | Collision-only RMC on the constant basis against standard Monte Carlo on the same MC/DC model: O-16, Al-27, C-12 fast and thermal, H-2 with (n,2n), and an A = 1 fissile medium. On Al-27, RMC resolves the flux down to 10⁻¹⁰ where standard Monte Carlo has few or no histories. The few-percent differences in sharp features (the O-16 2.4 MeV peak, the C-12 free-gas thermal peak) match the expected bias of the collision-only iterations. That is what the correction passes are designed to remove, though these cases have not been rerun with corrections. |
| `hpc_efficiency.png` | Error against cost on the OSU HPC (A = 1 absorber, 8 ranks). At equal cost (24 CPU s), the constant-basis, collision-only configuration is about 750× more accurate than standard Monte Carlo. Across all levels, the RMC figure of merit is 2×10⁵–5×10⁶ times higher. That configuration stays at its bias (2.04×10⁻⁵). The current configuration (linear energy basis + corrections) gets below it, to 3.1×10⁻⁶. |
| `hpc_timing.png` | Where the time goes on the HPC, in seconds per run. Compiling, or loading cached compiled code in a fresh process (110–220 s), costs more than most runs. The transfer moments load from the cache in under 0.5 s. The integrated correction sampler takes 99% of the current runs, and the correction cost is now being cut. |
| `progress_bar_fix.png` | A bug found while profiling on the HPC: MC/DC passed its progress counter by value, so it printed the progress bar after every history. The bars compare the same runs before the fix and after it, with the bar also turned off and on different nodes, so they bound the fix's effect rather than isolate it. Standard Monte Carlo ran 40–57× faster and the constant-basis RMC runs 24–423× faster. A likely but unconfirmed reason that RMC slowed more at larger sizes is that each print converted a structure holding the source bank, which grows with the run size. |

## Caveats

- **HPC numbers:** these come from one seed per level, on `share` nodes of different CPU generations, so single timings vary by roughly ±20%.
- **Absorber correction sampler:** the current-configuration absorber results use the integrated correction sampler. The sweep now uses the pointwise sampler for 0D problems, so rerun `make_plots.py` after copying in new results from `../efficiency`.
- **Flux gallery problems:** each problem is the example as defined in its directory. Al-27 is a synthetic nuclide with a constant scattering fraction σ_s/σ_t = 1/4, run on a scaled energy grid (plotted in unscaled eV). C-12 thermal is real 293.6 K data with the free-gas kernel. See each `input.py` and `README.md`.
