# Recent changes on `feature/g4-hpc`

Inventory taken on 2026-09-25 at `b27fa054`. It covers the 14 commits after
`49bbecff` (the starting `feature/G4-coupling` commit), plus the resulting file
changes. The working tree was clean when this inventory was made. The local
`upstream/dev` reference was last updated on 2026-09-14; check a fresh `dev`
before preparing PRs. This branch also inherits the broader Geant4 coupling
work from its parent branch.

## Core fixes to separate for `dev`

| Change | Commits and files | Suggested PR |
| --- | --- | --- |
| Electron data: convert generated MeV energies for eV transport, restore total versus large-angle elastic cross sections, use the expected `CDF` dataset name, and correct the bremsstrahlung data meaning. | `6ab2d565`, `f5936251`; `mcdc/object_/electron_reaction.py`, `mcdc/object_/element.py`, `tools/data_library_generator/electron/{generate.py,util.py,README.md}` | Use the existing `fix/electron-data-energy-units` branch. Avoid opening a duplicate PR from this branch. |
| Inelastic distributions: correct Kalbach–Mann outgoing-table bounds and tabulated energy-angle table bounds; hold the edge neutron cross section for off-grid energies. | `afe7bccf`, `644bbd66`; `mcdc/transport/distribution.py`, `mcdc/transport/physics/neutron/native.py`, `mcdc/transport/physics/util.py` | Use the existing `fix/inelastic-scattering-distribution-bounds` branch, which has focused tests. Review whether off-grid cross-section handling belongs in that same PR. |
| Cell surface-mesh tallies: recognize geometrically coincident boundary surfaces so current scores reach the intended cell face. | `b27fa054`; `mcdc/main.py`, `mcdc/object_/surface.py`, `mcdc/transport/tally/score.py`, `test/unit/tally/test_surface_mesh_tally.py` | Make a separate surface-tally fix PR. A `fix/surface-mesh-cell-face-classification` branch exists, but inspect its actual patch before reusing it because its history has diverged. |
| Electron elastic-angle sampling: use one quantile in neighboring angular tables and interpolate the two samples. | `b27fa054`; `mcdc/transport/distribution.py`, `mcdc/transport/physics/electron/native.py`, `test/unit/distributions/test_elastic_angular_interpolation.py` | Keep separate from the data-library PR. Validate before proposing it as a production fix: this change does **not** resolve the measured transport-moment mismatch between 0.256 and 10 MeV. |
| NumPy 2 compatibility for annotated array types. | `25420af8`; `mcdc/code_factory/numba_objects_generator.py` | Independent compatibility fix PR. |
| Neutron ACE data generation: supply explicit interpolation metadata when evaporation or Maxwellian spectra omit it. | `25420af8`; `tools/data_library_generator/neutron/util.py` | Independent neutron data-generator fix PR. |

## Coupling, examples, and diagnostics

| Change | Commits and files | Suggested destination |
| --- | --- | --- |
| Save MC/DC output before launching Geant4, retain transport results if the handoff fails, and propagate handoff errors across MPI ranks. | `48f4d960`; `mcdc/main.py` | Coupling PR; this depends on the Geant4 handoff code. The output-order and surface-tally edits in `mcdc/main.py` are separate changes. |
| CubeSat/HPC workflow: submit inputs and scripts, local HPC setup notes, electron source and tally settings, and supporting coupling/example adjustments. | Mainly `25420af8`, `e8ae7676`, `47ea7e51`, `546a010d`, `0e067aff`; `examples/cubesat_test/`, `hpc/`, `examples/coupling_test/common.py`, `.gitignore` | Example/HPC work, separate from core transport fixes. |
| Plotting: CubeSat source/payload plots and logarithmic spatial-current colors. | `0bd0cc3e`, `f5936251`, `b27fa054`; `examples/cubesat_test/diag/` | Keep with CubeSat diagnostics or make a small plotting PR. |
| Copied CubeSat runs and the Al-slab MC/DC versus Geant4 benchmark, including HDF5 files, plots, logs, decks, and analysis scripts. | Run-data commits plus `b27fa054`; `examples/cubesat_test/runs/` | Treat as records. Choose any benchmark source or summary files deliberately; do not bring the full run directory into a core bug-fix PR by default. |

## PR-splitting notes

- `b27fa054` combines the surface-tally fix, electron sampler, plotting, and
  run artifacts. `f5936251` combines a library fix with example edits. Split
  these by file or hunk rather than cherry-picking either entire commit.
- `mcdc/transport/distribution.py` contains both the inelastic bounds changes
  and the later electron angular-sampling change; split its hunks by behavior.
- Refresh `dev` before creating fix branches, then compare each proposed patch
  with the existing fix branches to avoid duplicate changes.
