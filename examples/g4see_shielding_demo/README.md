# MCDC-G4SEE Shielding Demo

This example is intentionally simple and currently includes two standalone MC/DC
inputs:

- `input_case_A.py`
- `input_case_B.py`
- `input_case_C.py`
- `input_case_D.py`

Each file defines its own materials, geometry, source, tallies, settings, and
run call directly in the script, following the style of the other MC/DC
examples.

For coupling to G4SEE, the primary test problems should be:

- `input_case_C.py`
- `input_case_D.py`

Those two cases use volumetric `flux` tallies in the SRAM region, which is a
better handoff for this demo than the surface `net-current` tallies used in
cases A and B.

## Run

```bash
python input_case_A.py
python input_case_B.py
python input_case_C.py
python input_case_D.py
```

For a faster larger case C run, use MC/DC's Numba mode from the command line:

```bash
python input_case_C.py --mode=numba --caching --N_particle=100000
```

## Notes

- Geometry is defined in `cm`.
- Neutron source energy is `14 MeV`.
- The SRAM region is modeled as a vacuum placeholder volume for the handoff.
- Case C uses a box outer wall and six isotropic volumetric source slabs around the geometry.
- Case D adds an aluminum shield around the entire PCB.
- The FR4 material entries are nuclide atom densities in `atoms/barn-cm`, so
  they are not expected to sum to `1.0`.
- Numba mode is selected with `--mode=numba`; it is not a setting inside the input file.
- Use `--caching` with Numba mode, otherwise MC/DC clears the cache and recompiles from scratch each run.
