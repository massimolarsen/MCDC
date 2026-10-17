# RMC write-ups

LaTeX notes for `mcdc/rmc`, grouped by topic. Each `.tex` builds on its own
(`\input{../preamble}`); run `pdflatex` inside its folder.

- `method/`: the RMC method itself.
  - `rmc_math.tex`: residual, iterations, correction passes (slab).
  - `scattering_correction_samplers.tex`: pointwise vs integrated samplers of r_s.
  - `residual_allocation.tex`: allocating correction-pass histories between terms.
  - `basis_study.tex`: projection quadrature, residual-norm and reference-operator
    qualifications for the 3D trial-space screening experiments.
- `kernels/`: inverted emission kernels, one note per ENDF law, and the
  sampling-vs-inverted verification (`verification/verify_laws.py` writes
  `figures/<case>/`).
- `bases/`: trial spaces beyond piecewise constant (linear in space, energy and
  angle; the self-shielded energy basis).
- `cartesian_3d/`: the 3D Cartesian solver.
  - `cartesian_3d.tex`: angular patches, the patch transfer and the face residual.
  - `tabulated_correction.tex`: correction passes with the tabulated in-scatter kernel.
  - `spatial_bases.tex`: linear and quadratic spatial bases (discontinuous and
    continuous) and the `flux-xyz` tally moments.
  - `streaming_source.tex`: experimental characteristic source trials, unbiased
    randomized inner and baseline integration, error accounting and scope.
