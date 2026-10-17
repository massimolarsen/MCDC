# Which trial space would make RMC efficient? A projection study

A correction pass of RMC samples the residual r = q - L psi~ of its trial function.
- With ideal residual sampling its source-weight second moment scales like
  ||r||_1^2, against ||q||_1^2 for standard MC. Tally variance also depends on
  the transported response and the proposal.
- (||q||_1/||r||_1)^2 is an efficiency screening indicator, not a measured
  variance or FOM gain.
- Iterating the trial function by Monte Carlo needs histories in proportion to its
  number of unknowns.

This study measures ||r||_1/||q||_1 for many candidate trial spaces, without running
RMC. It takes a fine deterministic reference psi and computes the residual of its
projection Pi psi onto each trial space: an ideal approximation study, not the
noisy collision-only fixed point. On the thermal pin it reproduces the residual of an actual
RMC run with the constant basis (1.10 against 0.51 + 0.60 measured; see
`mcdc/rmc/NOTES.md`, section 13).

## Method

- **Reference** (`sn3d.py`): 3D Cartesian multigroup discrete ordinates.
  - Diamond difference with a negative fixup.
  - 2048 product-quadrature directions aligned with the patch edges (8 Gauss
    points per panel). The former 512-direction quadrature did not reproduce P7.
  - P7 scattering.
  - Fine groups (3 per coarse bin) and a fine mesh (4 per coarse cell and axis;
    8 for the absorber).
  - Checked against the exact ray-integral solution of the absorber box: 0.35% rms.
- **Physics** (`data.py`): elastic scattering on a target at rest, isotropic in the
  centre-of-mass frame, from the actual cross sections. Every other reaction is
  removal.
  - Thermal: the synthetic nuclides of the fuel rod, which have exactly this physics.
  - Fast: real elastic data with inelastic and fission as removal, an analogue of the
    real fast window.
- **Trial spaces** (`project.py`): tensor products of
  - energy: constant or linear per coarse bin (E0, E1);
  - angle: patches J = 8 or 32 carrying polynomials of degree p <= 2 (J8p0 ... J32p2),
    or spherical harmonics P1-P7;
  - space: dg0, dg1, dg2, cg1, cg2 (`mcdc.rmc.space3d`);
  - optionally a trial mesh refined r times per axis.
- **Residual:** volume part on the fine grid, plus the jumps across coarse faces,
  with fine-cell/group averaging before the absolute value. This can underestimate
  the pointwise L1 norm; it requires refinement checks.
- **Streaming-exact trial functions** (`project.streaming_exact`):
  psi~ = (Omega.grad + Sigma_t)^-1 (q + Pi S psi). The scattering source, not the
  flux, is projected onto a basis: spatial polynomial x spherical harmonics up to N.
  - Streaming is exact in the mathematical representation; the study approximates
    its inverse with diamond difference.
  - The code now constructs this trial directly and evaluates S psi~ - Pi S psi,
    avoiding the identity that assumed the fixed-up reference used the same inverse.
- **Problems:**
  - the absorber box (streaming, 2 x 2 x 2 cells of one mean free path);
  - the C5G7-style pin, thermal window, lattice and isolated;
  - the pin's fast-window analogue (lattice).

## Running

    python run.py PROBLEM [REFINE [N]] --mode=numba       # polynomial trial spaces
    python run.py PROBLEM [REFINE [N]] --sx --mode=numba  # streaming-exact trials
    python summarize.py                       # Pareto fronts -> summary.md

## Results

The existing JSON, logs, results.md and summary.md are historical results from
the former 512-direction quadrature and indirect streaming identity. They have
not been overwritten and must not be interpreted as results of the corrected
study. Regenerate every case before making updated Pareto comparisons.
Corrected records carry `study_version=2`, quadrature and evaluation metadata;
the table generators exclude unversioned historical records rather than mixing
the two methods in one Pareto comparison.
The separate `../streaming_source` prototype measures actual randomized estimators
and total-cost FOM on a prespecified 3D matrix.
