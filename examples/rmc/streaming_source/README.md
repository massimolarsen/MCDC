# Streaming-source RMC: research and verification plan

The objective is a broadly useful 3D RMC method with a better figure of merit
than standard Monte Carlo (SMC), including the cost of constructing and evaluating
the trial. A convenient problem is not sufficient evidence. This prototype tests
a transport-adapted trial on a matrix of source, optical-depth, scattering-ratio
and material configurations. It is a one-group research model, not MC/DC's
continuous-energy production solver.

## Operational concept and requirements

Represent the isotropic collision source Q by one value per structured cell.
The trial is psi_t = T^-1((q + Q)/(4 pi)), with T = Omega.grad + Sigma_t.
Backward characteristic integration treats cell interfaces and attenuation
directly. The residual is isotropic, r = (Sigma_s phi_t - Q)/(4 pi), and ordinary
signed-weight Monte Carlo transports its particles. Both the trial's cell-average
tally and pointwise scalar flux are estimated with independently randomized
quadrature, so quadrature noise is measured rather than silently treated as zero.

| ID | Requirement | Component and verification |
|---|---|---|
| R1 | The estimator shall solve the same fixed-source transport equation as SMC. | Shared transport kernel; absorber and homogeneous-medium checks; repeated-seed agreement. |
| R2 | Every nonzero residual region shall have proposal support. | Strictly positive cell proposals; manufactured cancellation tests. |
| R3 | Randomized inner and trial-tally estimates shall have correct expectations. | Independent random shifts; exact characteristic tests; independent-seed uncertainty coverage. |
| R4 | FOM shall include source construction, trial tally, inner quadrature and correction transport. | Separate non-overlapping timers; cold compilation reported separately. |
| R5 | Conclusions shall cover a prespecified case matrix and report failures. | Thin/thick, weak/strong scatter, uniform/localized sources, homogeneous/heterogeneous media. |
| R6 | New algorithms and limitations shall be documented. | This file and `mcdc/rmc/writeups/cartesian_3d/streaming_source.tex`. |

## Interfaces and limits

Inputs: increasing x/y/z edges; total and scattering cross sections and isotropic
external source density per cell. Sources integrate to one. Initially all outer
faces are vacuum. Cross sections are piecewise constant, with 0 <= Sigma_s < Sigma_t
in material cells; voids have Sigma_s = Sigma_t = 0.
Outputs: scalar flux per cell, independent-run errors, phase times and FOM ratios.
Global seeds identify independent source construction, baseline and residual draws.
No production neutron physics, nuclear data or MC/DC transport settings are changed.

The frozen source is obtained from sampled first-flight cell-response coefficients,
then a small linear solve. It need not be exact: arbitrary frozen Q leaves the
residual identity valid. Sampling of that source solve is independent of final
estimation. Baseline cell averages use repeated randomized low-discrepancy
first-flight integration. The correction uses independent randomized inner rays
and a fixed positive spatial proposal; it never accepts or rejects particles
using a noisy residual estimate.

## Acceptance and iteration

Verify components first, then independent-run bias/error coverage, then compare
variance times *total* wall time on every case. Vary the inner ray budget and the
number of correction histories. Include pure SMC and report regressions as well
as improvements. An extension to reflective boundaries, energy-dependent kernels,
anisotropy or multiplying media requires its own derivation and verification;
success here is evidence about the algorithm, not validation of those extensions.

## Running

`python run.py --output results.json` runs every matrix case with eight independent
complete runs per method and inner budgets 4, 16 and 64. `--cells 4` changes the
mesh; `--histories`, `--smc-histories`, `--power` and `--repeats` set the effort.
Use an even `--cells` count (for example 2 or 4) to keep physical source and material
interfaces fixed. `--linear`, `--antithetic`, `--forward`, `--targeted`, `--split`
and `--unfitted` select the documented experimental comparisons. Forward,
targeted and split are mutually exclusive; linear currently uses backward only.
The JSON includes raw independent estimates, estimated and empirical variances,
phase times, agreement diagnostics and both versions of the FOM ratio. Both
methods use the same analog collision and track-length transport kernel.

Run `python verify.py` for the standalone experimental physics and statistical
checks. These cover exact rays, conservation, void limits and independent SMC
agreement for every estimator variant; they are deliberately outside the unit
suite. Only three focused regression tests were added to the unit suite, covering
production proposal support, setup timing and the study's P7 projection.
