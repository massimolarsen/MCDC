# CE-RMC investigation notes

Running log of findings and deferred fixes for the continuous-energy Residual Monte
Carlo implementation in `mcdc/rmc`.

## 1. Why the scattering residual r_s stalls convergence (2026-10-02)

**Observation.** If every iteration includes the scattering correction r_s
(T_bar − s_bar, with T_bar the true in-scatter of psi~ and s_bar its flat per-block
estimate), RMC stalls after 1–2 iterations near plain-SMC accuracy. If r_s is left out
(collision residual only), RMC converges exponentially, about 10–20x per iteration in
||eps~||. The error then settles on a flat floor that shrinks with G. This is the
behaviour in the NSE article's Fig. 3.

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
uses the bin average Sigma_bar_t (the dissertation's piecewise-constant cross-section
assumption, as in MC^2-3). The phase-1 residual then vanishes at the flat-flux-weighted
multigroup solution, reached exponentially. The in-bin part, (Sigma_bar_t -
Sigma_t(E)) psi~, is sampled with r_s in the correction passes, which always use the
pointwise Sigma_t(E). For constant cross sections nothing changes.

### 2b. 1D: face residuals stall the same way

In a slab, the edge residual -mu (psi~_R - psi~_L) delta(z - z_f) sits on the faces
while the collision residual is spread over the cell. The in-cell shape never vanishes,
so 1D iterations plateau at the noise of one iteration ([D] Fig. 4.9 shows the same
plateaus for the fuel rod). Averaging the iterates after the plateau, or a
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
