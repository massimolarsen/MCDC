# CE-RMC investigation notes

Running log of findings and deferred fixes for the continuous-energy Residual Monte
Carlo implementation in `mcdc/rmc`.

## 1. Why the scattering residual r_s stalls convergence (2026-10-02)

**Observation.** If every iteration includes the scattering correction r_s
(T_bar − s_bar, with T_bar the true in-scatter of psi~ and s_bar its flat per-block
estimate), RMC stalls after 1–2 iterations near plain-SMC accuracy. If r_s is left out
(collision residual only), RMC converges exponentially, about 10–20x per iteration in
||eps~||. The error then settles on a flat floor that shrinks with G.

A = 1, c = 1/2, collision residual only, analytic reference:

| G  | ||eps~|| per iteration      | error floor (L∞, relative) |
|----|-----------------------------|----------------------------|
| 20 | 3.5 → 1e-2 → ... → 1e-14    | 1.3e-3                     |
| 50 | 3.5 → 2e-2 → ... → 6e-10    | 4.2e-4                     |

**Cause 1: the floor does not shrink.** RMC noise is proportional to the residual
norm. The collision residual can reach zero (at the fixed point of the binned
in-scatter operator). r_s cannot: even at the exact bin averages it holds the in-bin
shape of the in-scatter source. Measured exactly for A = 1:

| G   | ||r_s||_1 / in-scatter |
|-----|------------------------|
| 20  | 2.9%                   |
| 50  | 1.2%                   |
| 100 | 0.59%                  |

**Cause 2: pointwise E_in sampling inflates the variance about 485x.** The sampler
draws (E_in, E_out, mu_out). In diagonal (g' = g) blocks, T_bar is a kinematic triangle
(E_out < E_in for A = 1) while s_bar is flat, so the weights scale with the full
in-scatter norm instead of ||r_s||. At G = 100: E[w^2] = 1.36e-2 per history, against
2.8e-5 for an ideal sampler with weights of ±||r_s||_1.

**Fewer r_s particles does not help.** Weights carry the factor N / N_s, so r_s noise
grows like 1/sqrt(N_s). Scaling the weights down instead biases the answer.

## 2. Current scheme: two-phase schedule (implemented)

1. Iterate with the collision (+ edge) residual only. This converges exponentially to
   the fixed point of the binned in-scatter operator. That fixed point carries a small,
   G-dependent bias.
2. Optionally run `N_correction` final passes with the full residual (collision, edge,
   r_s, r_f) at the fixed point psi~*. Each pass estimates psi − psi~* without bias.
   The passes are averaged, not accumulated, so the correction's noise falls as
   1/sqrt(passes).

With the current r_s sampler, removing a bias of ~1e-3 needs many passes, because a
single pass has ~0.4x SMC noise at a 10% r_s budget.

### 2a. Bin-averaged Sigma_t in the collision-only phase (2026-10-02)

**Observation.** With the pointwise Sigma_t(E) in r_c = Q + S_bar - Sigma_t(E) psi~,
the collision-only phase still stalled for energy-dependent cross sections: O-16
(1-10 MeV, G = 50) had ||eps~|| flat after the first iteration and an error at the
noise level of one iteration (~8% rms against SMC). The A = 1, sigma ~ E^-1/2 case
stalled at a lower level (||eps~|| ~ 1e-4 of ||psi~||).

**Cause.** Same mechanism as r_s: for a piecewise-constant psi~, Sigma_t(E) psi~ keeps
the in-bin shape of Sigma_t(E). Its bin integral equals Sigma_bar_t psi~ dE exactly,
so that part of r_c has zero bin integral and never vanishes.

**Fix (implemented, `collision_xs="binned"`, the default).** The collision-only phase
uses the bin average Sigma_bar_t (the dissertation's assumption of constant flux and
cross sections within a bin, as in MC^2-3; [D] Secs. 3.4.3 and 4.4.1). The phase-1 residual then vanishes at the flat-flux-weighted
multigroup solution, reached exponentially. The in-bin part, (Sigma_bar_t -
Sigma_t(E)) psi~, is sampled with r_s in the correction passes, which always use the
pointwise Sigma_t(E). For constant cross sections nothing changes.

### 2b. 1D: face residuals stall the same way

In a slab, the edge residual -mu (psi~_R - psi~_L) delta(z - z_f) sits on the faces
while the collision residual is spread over the cell. The in-cell shape never vanishes,
so 1D iterations plateau at the noise of one iteration. Averaging the iterates after the plateau, or a
linear-in-z trial space, would reduce it. Not implemented.

### 2c. Tally energy tolerance and narrow source bins

MC/DC scores particles within 1e-5 eV (absolute) above a bin edge in the bin below. A
narrow source bin [E0 - delta, E0] has a large uncollided flux, so with delta = 0.01
about 1e-3 of it lands in the top collided bin (a ~6% error there for SMC). RMC is
unaffected in 0D: its fixed point is set by the binned equations, not by the tally.
The examples start the SMC source 3e-5 eV above the edge. For smooth collided flux
the net effect per bin is ~1e-5 / E (relative).

### 2d. Quadrature tolerance floor (2026-10-03)

The lowest-decade tolerance tol_low = 1e-10 applied to every row when the whole window
lies within one decade (Al-27 low, fuel rods). For tabulated data that target is below
the integrand's noise (inner angular integrals at TAU_TOLERANCE = 1e-10, table-lookup
round-off), so every piece refined to MAX_DEPTH = 30 and single rows ran for hours
(O-16, 1-10 eV scaled: 6 ms at 1e-8, 0.14 s at 1e-9, unfinished after hours at 1e-10).
At 1e-9 it is still ~1e4x slower than 1e-8 (O-16 row 90: 134 s vs 6 ms, same 7
digits): the angular transfer table is linear in theta between 8193 nodes, and each
node is an unlisted kink at that accuracy. Fixes: tol is clamped at TOLERANCE_FLOOR =
1e-8, tol_low defaults to 1e-8 (so the low-decade setting has no effect until the
angular table is made smoother, e.g. cubic in theta or kinks as breakpoints),
MAX_DEPTH is 20, and a piece is also accepted within its width share of tol times the
row magnitude. Remaining cost: continuum laws (U-235 MT-5, MT-91) take ~1e2 s per row
regardless of tolerance (many breakpoint panels); worth optimizing.

## 3. Deferred fix: low-variance r_s (deterministic E_in integration)

Evaluate r_s(E_out, mu_out) = sum_g',j' psi~_g'j' int_g' dE_in sum_r N sigma_r
tau_bar_r,j' − s_bar pointwise, with the E_in integral done deterministically. The
kinematic support limits it, e.g. E_in in [E_out, E_out / alpha] for elastic. Then sample
(E_out, mu_out) per bin with weights of about ±||r_s||_1. Expected variance reduction:
~485x at G = 100 for A = 1. Even then, the per-iteration r_s noise (about 1e-3 at 20k
histories, G = 100) is above the bias it removes (about 2e-4), so use it in the
final-correction phase rather than every iteration.

## 4. Other open items

- Higher-order (e.g. linear-discontinuous) trial space in energy: reduces both the bias
  and ||r_s||.
- Adaptive energy refinement near resonances and kinematic edges.
- Tally edge tolerance: MC/DC's absolute 1e-5 eV energy coincidence tolerance assigns
  particles near a bin edge to the lower bin. Narrow bins (e.g. a smeared monoenergetic
  source bin) must be much wider than 1e-5 eV.
- Upstream MC/DC bugs found along the way: Kalbach/Law 61 table bounds (fix branch
  `fix/inelastic-scattering-distribution-bounds`); energy-dependent inelastic yields
  stored as an integer 101 (fix branch `fix/energy-dependent-multiplicity`); delayed
  fission neutrons keep the parent energy (RMC runs use all-prompt fission).

## 5. Kernel coverage and verification (2026-10-03)

Every law MC/DC samples is now inverted (write-ups with the math and the
sampling-vs-inverted verification in `writeups/`, generated by
`writeups/verification/verify_laws.py`): elastic (target at rest, anisotropic COM
data), free gas (E <= 400 kT, isotropic COM only; the driver stops otherwise), level
scattering, tabulated/multi-table spectra and angles, fission (all prompt),
Kalbach-Mann, Law 61, evaporation, Maxwellian, N-body, and reactions with several
spectra and a tabulated yield. All 22 verification points pass (chi-square p from 0.16
to 0.96, yields to 4e-4 or better).

MC/DC bugs found while inverting, fixed on separate branches merged into feature/RMC:
- `fix/free-gas-temperature-nbody-scaling`: the free-gas target Maxwellian used a
  hard-coded 293.6 K, and its relative-speed rejection was inverted (it sampled an
  unweighted Maxwellian); N-body (Law 66) outgoing energies were the reduced variable
  times 1 MeV, independent of E (generator dropped NPSX and Ap). H-2 regenerated.
- `fix/evaporation-maxwellian-loading`: evaporation/Maxwellian temperature tables
  with no interpolation regions (ACE NR = 0) failed to load (C-12 could not be loaded
  at all); the Maxwellian interpolation dataset name differed between generator and
  loader.

Not inverted: anisotropic free gas (no closed form) and S(alpha, beta) (MC/DC does
not sample it yet).

## 6. External audit of feature/RMC at 22d1e6ae (2026-10-03)

Triage by how likely each finding is to affect real runs:

| # | Finding | Likelihood / impact | Status |
|---|---------|---------------------|--------|
| 1 | Integrated correction (`in_scatter_density`) splits E_in only at trial edges and support crossings, not at the cross-section grid; a narrow resonance missed by all 15 G-K nodes returns 0 (deterministic bias) | Likely in resonant materials; affects the correction passes only | Deferred (see below) |
| 2 | Pilot mass floor `0.1 abs(S_bar)` gives no or tiny support when signed psi~ makes S_bar ~ 0 while T != 0 | Exact zero is measure-zero; near-zero raises variance with negative/noisy psi~ | Deferred: build the floor from sum abs(psi~) M |
| 3 | Moment cache keyed without data contents, version, or effective quadrature; non-atomic writes | Real: data were regenerated three times this session | **Fixed**: key/provenance include the data SHA-256, `MOMENT_FORMAT_VERSION`, effective tolerances, N_theta, depth; validated on load; atomic rename |
| 4 | Rounded scatter/fission budget can give a small active term zero histories | Only when a term's mass < ~1/(2N); bias bounded by that term | Deferred: probabilistic term selection with 1/P weights, or one history minimum |
| 5 | `_neutron_inelastic_scattering_production_xs` reads the integer multiplicity (-1 for tabulated yields) | nu-production tally scores only, MT-5-type reactions | Deferred (MC/DC fix branch) |
| 6 | `m (gamma - 1)` speed-to-energy conversion cancels catastrophically (MC/DC and RMC kinematics) | Relative error ~1e-16 / (E / 470 MeV): 1e-6 at thermal, 1% at 1e-5 eV | Deferred (MC/DC fix branch; use beta^2 / (sqrt(1-beta^2)(1+sqrt(1-beta^2)))) |
| - | Phase-1 results shown as "RMC" without noting the binned bias | Communication | Deferred: record scheme/N_correction in output and labels |
| - | Dense G^2 J^2 storage, pilot repeated per rank, round-robin rows, unreported depth-limit acceptance, numba_only marker at import, CI data skips | Performance / diagnostics / test infrastructure | Deferred |

### Deferred fix for #1, and why it is not a convergence fix

Convergence (rate and floor) is limited by the piecewise-constant trial space, not by
quadrature: the moments are accurate to ~1e-8, the trial-space errors are 1e-3 to
1e-1. Finding 1 only affects whether the *correction passes* are unbiased in
resonance bins. It becomes essential with better trial spaces (a self-shielded basis
f_g ~ 1/(E Sigma_t) carries the full resonance structure; adaptive refinement makes
bins resonance-dominated), so do it as part of that work:

- Split the E_in integral at every non-smooth point: cross-section grid points in the
  kinematic window, reaction thresholds, the distributions' incident-energy grids,
  support crossings, trial-space edges, the 400 kT free-gas threshold; few Gauss points
  per smooth piece; drop the `K == 0` early acceptance at shallow depth.
- Cost is set by kinematic pruning: U-238 elastic needs E_in in [E', E'/alpha]
  (~1.7%), a few hundred of its ~2e5 points per evaluation.
- Wide-range reactions (fission: every E_in reaches every E'): product integration --
  per reaction, prefix sums of int sigma E^n dE (n = 0..3, exact for lin-lin data);
  approximate the smooth kernel by a low-order polynomial between its own breakpoints;
  each segment integral is then a few lookups, independent of the data density. For
  fission spectra the E_in dependence is only table interpolation, so the bin integral
  factorizes exactly: sum_k (int nu sigma_f w_k) chi_k(E').
- Precompute T at fixed nodes per outgoing bin in the cached moment library, so this
  cost is paid once per nuclide and grid.

## 7. Linear discontinuous energy basis (2026-10-03)

`energy_basis="linear"`: psi~(E) = psi~_0 + psi~_1 x_g(E) on each energy bin (Legendre
P_0, P_1 with x_g in [-1, 1]); z and mu stay piecewise constant for now.

Choice of continuity, from the literature: the Texas A&M ECMC work (Peterson, Morel,
Ragusa, M&C 2013 and thesis; Franke, Bruss, Morel 2013; Bolding, Morel 2017) used linear
discontinuous trial spaces in space and angle. Vermaak and Morel (JCP 2022) then showed
that spatial discontinuities are a pitfall: mu d/dz turns each jump into equal and
opposite face sources on the two sides of a face, which mostly cancel (piecewise-constant
RMC was 6.5x *less* efficient than SMC on a 1D slab, continuous linear 3x more). Only z
is differentiated by the transport operator (no d/dE without CSD, no d/dmu in slab
geometry), so: **continuous linear in z** (planned next), **discontinuous linear in E and
mu** (no residual cost, local projection, and real jumps at E0, alpha E0, mu = 0).

Pieces:
- Moments M[a', g', j', a, g, j] weight E_in by P_a' and E_out by P_a (same kernel
  evaluations; E_out is known at every inner node). Cache key includes `energy_order`
  (format version 3).
- Projected source S_bar_a = (2a+1)/(dE dmu) sum M psi~; collision-only iterations use
  the projected removal R[g, a, b] = (2a+1)/dE int Sigma_t P_a P_b (the Galerkin fixed
  point; for the constant basis this is the old bin-averaged Sigma_t).
- |r_c| masses are exact: c(E) - Sigma_t(E) psi~(E) is quadratic between xs grid points.
- Tally: new MC/DC track-length score `flux-energy-slope` (flux times x_g(E)); the
  coefficient is (2a+1) score / volume.

First results (A = 1 analytic, 20 bins, 500 histories/bin):
- Collision-only fixed point: bin averages within 2.5e-4 of exact (constant basis
  3e-3), P_1 coefficients within 0.09%; eps~ falls ~10x per iteration.
- Full-residual iterations (r_c + r_e + integrated r_s every iteration): constant
  basis converges 2 iterations then stalls at |eps~| ~1e-3, rms error 0.7-1%; linear
  converges 4 iterations (3.5, 0.19, 1.2e-2, 5.3e-4, 8.5e-5) and stalls at ~7e-5,
  rms error ~5e-4 (15-20x lower).

Linear basis, further findings (2026-10-03):
- RMC converges exactly to the deterministic Galerkin solution of the projected
  equations (A = 1 absorber: RMC vs direct solve of R psi = Q + S_bar[psi] with the same
  moments, 1e-16). O-16 1-10 MeV: that Galerkin solution is within 1.9% rms of SMC
  (constant-basis fixed point ~4.5%).
- Absorber at 100 P/B/I: the linear runs plateau at ~5e-10 for G = 50, 100, 200 alike.
  With exact (closed-form) moments the linear Galerkin error is 2.6e-10, 1.9e-11,
  3.6e-12 (about 4th order in h; constant: 8.1e-7, 2.1e-7, 5.2e-8, 2nd order, equal
  to the constant RMC plateaus). The common ~1e-7 relative floor is physics, not
  quadrature: MC/DC (and the inverted kernels) use relativistic elastic kinematics;
  the moment difference to the non-relativistic closed form is 1.4 E/mc^2 in every
  bin, and the analytic reference is non-relativistic. A relativistic reference is
  needed to see the linear basis' G-convergence below ~1e-7.
- Minimum histories: the P_1 coefficient is tallied as 3 x track length x x, so its
  noise per unit flux is ~sqrt(3) that of the average; the iteration's noise gain is
  ~sqrt(3) larger. Absorber: constant converges at 10 P/B/I (gain ~0.6), linear does
  not (~1.0) and needs ~3x the histories; O-16 1-10 MeV at 50 P/B/I diverges for the
  same reason. Antithetic mirrored-energy pairs with shared transport random numbers
  (now used for the linear basis' r_c + r_e) do not change this: most of a bin's flux
  in slowing down comes from in-scatter, not first flights.

## 8. 0D end-to-end kernel tests: C-12 and H-2 (2026-10-03)

`examples/rmc/c12`, `examples/rmc/h2` (phase 1 against SMC, 10^6 histories):
- H-2 1-20 MeV (N-body, A = 2 elastic): pass, rms z 0.98.
- C-12 1-30 MeV (elastic, levels, evaporation, KM with a tabulated yield): pass in
  flat-Sigma_t bins (rms z 1.25); the top bin 29-30 MeV is +4.4% (z = 6), unexplained.
- C-12 0.01-10 eV (free gas): not decisive at c = 0.9993 (phase 1 converges slowly
  and its bias is amplified by ~1/(1-c)). The free-gas moments match MC/DC's sampler
  directly (per-bin rms z ~1, upscatter fractions within 3e-4, conservation 2e-7).
- Fixes found by these runs: zero-width interpolation intervals at repeated grid
  energies (`grid_fraction` in evaluate.py; C-12 elastic angles repeat 20 MeV), and the
  free-gas isotropy guard now weights the next angular table by its interpolation
  fraction below the threshold (tolerance 1e-4; C-12's sampled anisotropy is 3e-6).
- z-scores of a scalar flux from SMC must treat the polar bins as correlated (in 0D
  every track scores both); assuming independence inflated z by up to sqrt(2).

## 9. Continuous linear spatial basis (2026-10-03)

`spatial_basis="linear"` (writeups/continuous_linear_space.tex): nodal values on
z_edges, boundary conditions imposed on the trial space (vacuum incoming nodal values
zero, reflective tied to the mirror bin), so there is no face residual. New MC/DC
track-length scores `flux-z-slope` and `flux-z-energy-slope` give the cell P_1 moments
in z; the projection (`mcdc/rmc/space.py`) assembles hat-function moments and solves
the constrained mass system. The residual uses the linear interpolation of the nodal
in-scatter (no projection in z) and the streaming term -mu (Phi_{k+1} - Phi_k) / h_k.
Phase 2 supports the integrated sampler only.

- RMC converges to Pi psi, the L2 projection onto continuous linears. That projection
  is global and does **not** preserve cell averages; it preserves the hat moments of
  unconstrained nodes. Comparing Pi psi's cell averages with SMC cell tallies on the
  two-material vacuum slab (cells 2 mfp, source in cell 0) gave rms z = 16.6 although
  nothing was wrong; comparing interior hat moments (SMC from its flux and z-slope cell
  tallies) passes (`test_slab_against_smc_linear_z`).
- The phase-1 plateau of that slab did not drop (eps after the first iteration ~0.05,
  constant basis 0.02-0.09): with two polar bins, the in-bin mu shape of the streaming
  term (mu - mu_j) dPhi/dz is as large as the face residual it replaces. Linear
  discontinuous angle (writeups/linear_discontinuous_angle.tex) or more polar bins
  should address this; the face-cancellation pitfall itself is gone.

## 10. Linear discontinuous angular basis (2026-10-04)

`angular_basis="linear"` (writeups/linear_discontinuous_angle.tex), combinable with the
energy and spatial bases: coefficients c = b P + a per bin (energy a < P, angle b < 2).
Closed-form outgoing moment Pi^1 (`kinematics.azimuthal_bin_moments`, checked against
brute force to 2e-6); angular transfer tables A^{b'b}; the correction kernel returns
both incident moments from one adaptive integration; reflective boundaries flip the
slope's sign; MC/DC scores flux-mu-slope and its products with the energy and z slopes.

Slab tests (`examples/rmc/slab_bases`): all four spatial x angular combinations are
unbiased against SMC. At 200 histories per bin none lowers the phase-1 plateau; linear
angle plateaus slightly higher (eps ~0.08-0.13 vs 0.02-0.09). The plateau there is the
single-iteration noise, which the slope coefficients (~sqrt(3) noisier tallies) raise;
the trial-space gain needs more histories per bin to show.
