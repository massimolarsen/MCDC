# What the first measured iterations establish

## Objective and scope

Find a generally useful 3D residual construction and sampler with a better
variance-times-total-cost tradeoff than SMC. These experiments are a one-group
research stage, using a shared analog kernel for the comparison. They do not
establish a production MC/DC speedup or validate continuous-energy scattering.

The algorithm and assumptions are derived in
`../../../mcdc/rmc/writeups/cartesian_3d/streaming_source.tex` (from the repository
root the file is `mcdc/rmc/writeups/cartesian_3d/streaming_source.tex`). The local
README records requirements R1-R6, interfaces and verification gates.

## Prespecified matrix and independent validation

The box is [0,2]^3 with vacuum boundaries. Homogeneous total cross sections are
0.3/cm (thin) and 3/cm (thick); the contrast case alternates 0.1/cm and 5/cm on
the eight unit macro-cubes. Scattering ratios are 0.2, 0.8 and 0.97. The source
is either uniform throughout the box or throughout [0,1]^3, always normalized
to one. The material and source interfaces remain fixed on mesh refinement.

All five final matrices use 24 complete independent runs and a fresh seed family
starting at integer 20261101. There are 18 cases and five variants, each with its
own matching SMC runs: 4,320 complete estimates in the saved validation data.
All RMC times include source construction, pilot allocation, randomized baseline
and residual transport. Both paths are warmed consistently for compilation.
The independently scrambled baseline uncertainty is included in reported errors.
The strongest agreement diagnostic across the fitted variants is max |z| = 3.56
over the many cell comparisons; empirical/reported aggregate variance ratios
range approximately 0.76-1.36. These are diagnostics, not a simultaneous coverage
proof. Pointwise bootstrap intervals exclude timing-environment uncertainty.

The refined variant scores its 64 cells, while other variants score eight.
Each RMC/SMC comparison uses identical responses, but cross-resolution changes
therefore do not isolate trial-space resolution alone.

## Outcomes, including failures

See `results.md`, `fom_matrix.png`, and raw records under `results/`.

| variant | empirical FOM above SMC in cases | pointwise 95% interval above SMC |
|---|---:|---:|
| Unfitted uncollided/collided control | 5/18 | 5/18 |
| Fitted flat collision source | 7/18 | 6/18 |
| Antipodal inner rays | 8/18 | 7/18 |
| Limited linear source reconstruction | 8/18 | 6/18 |
| Refined source/transport mesh | 8/18 | 8/18 |

The fitted flat source gives about 15x total-cost FOM on the thin c=0.8 uniform
case, 52x on the thick c=0.2 uniform case, and 2.2x on the contrast c=0.8 uniform
case. The unfitted control gives about 2.7x, 2.9x and 0.42x respectively: the
fitted source contributes benefits beyond uncollided integration alone.

The same flat method loses badly on thick c=0.8 corner-source transport (about
0.081x SMC) and contrast c=0.97 corner-source transport (about 0.052x). Neither
linear reconstruction nor uniform refinement repairs this pattern. Antipodal
integration removes odd angular components but is not a broad performance cure.
The complete matrices are retained; selecting the best variant separately for
each case would require an independent selection/validation protocol.

## Iterations before the held-out runs

- The initial estimator retained all randomized errors rather than using the
  oracle study's exact trial tally. This exposed localized-source baseline noise
  that residual-norm screening alone had hidden.
- A forward baseline, source-directed angular cones and external/collision-source
  splitting were implemented and checked for agreement. Exploratory matrices
  did not show a generally better tradeoff, so they remain explicit controls,
  not defaults.
- A positive pilot-based response proposal and shared local Sobol coordinates
  reduce sampling/generator overhead without treating correlated cell errors as
  independent. A fixed 10% volume mixture preserves support.
- Exploratory K=4 versus K=16 comparisons reduced source-weight second moments,
  but the extra rays often worsened total FOM. For example, thick c=0.8 corner
  transport changed its source-weight second moment from about 0.73 to 0.56
  while approximately doubling time. These exploratory settings differ from
  the final validation budget and are not the primary performance evidence.

## What to pursue next

1. Improve **residual transport variance**, especially cancellation and response
   importance for localized sources. A small source residual does not imply a
   small variance for a remote cell response. The current proposal only adapts
   by birth cell; it does not resolve full position-direction response structure.
2. Test correlated signed-history estimators or rigorously unbiased cancellation,
   with covariance included in uncertainties. Independent positive and negative
   transport is a concrete remaining cost. No such new method is implemented here.
3. Test an adaptively refined or genuinely projected higher-order collision source
   rather than only a clipped reconstruction. Hold physical responses fixed for
   that trial-space comparison; uniform mesh refinement alone did not solve the
   difficult cases.
4. Keep the unfitted hybrid and optimized SMC controls. A source-fit gain is not
   sufficient if its precompute, baseline tally or residual evaluation costs more
   than the gain saves.
5. Extend a demonstrated, adequately broad one-group improvement to energy and
   angle incrementally, verifying against production MC/DC physics. Do not infer
   CE efficiency from the large favorable one-group ratios.

The broad objective remains open. The present result is an implemented,
error-accounted research path with a useful failure map, not a claim that 3D RMC
is generally faster.

## Validation traceability

R1/R3: exact heterogeneous and linear-source rays, small-optical-depth and void
limits, first-flight conservation, independent SMC comparisons for all estimator
variants. R2: manufactured angular-corner cancellation, positive proposal mixture.
R4: disjoint timing regression in both production spatial paths, complete phase
times in the research records. R5: all 18 cases and all losses retained, fresh
seed family, independent-run bootstrap. R6: algorithm derivations and limitations
in the linked LaTeX writeups; standalone copies of all four changed/new 3D study
notes were compiled with their shared preamble inlined.

The experimental checks are retained in `verify.py`, outside the unit suite.
The final unit-suite addition is three focused regression tests for confirmed
bugs: angular proposal support, disjoint setup timing (both spatial paths), and
P7 quadrature reproduction. Existing unit tests are preserved.

Final check after consolidating the tests: all three new unit regressions passed
with `--mode=numba`; the standalone `verify.py` passed exact-ray, conservation,
void and independent SMC agreement checks for all seven estimator variants.
Earlier, the four existing production 3D transport tests passed after the support
and timing fixes. `git diff --check` also passed.
