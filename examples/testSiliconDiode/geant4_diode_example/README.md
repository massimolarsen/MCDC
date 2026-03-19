# Geant4 Diode Example

This is a small standalone Geant4 test problem for the MCDC diode coupling.

It models:
- a silicon cylindrical diode bulk
- a slightly smaller silicon sensitive volume
- a thin aluminum layer above the diode

The source is driven through Geant4 GPS so it can consume the exported MCDC
histograms:
- `spectrum.mac`
- `angle_spectrum.mac`

The app is intentionally serial so the output stays clean and easy to compare
while we are validating the coupling.

The example currently uses the standard `FTFP_BERT` physics list to keep this
reference problem lightweight and stable. If you want a higher-fidelity neutron
physics setup later, we can make an `HP` variant as a second example.

## Build

```bash
cd /Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/geant4_diode_example
cmake -S . -B build
cmake --build build -j4
```

## Run a simple monoenergetic test

```bash
cd /Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/geant4_diode_example/build
./mcdc_diode_geant4 run_mono.mac
```

## Visualize the geometry and source placement

To inspect the diode geometry interactively:

```bash
cd /Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/geant4_diode_example/build
./mcdc_diode_geant4
```

To draw a few monoenergetic events on top of the geometry:

```bash
./mcdc_diode_geant4 inspect_mono.mac
```

## Run with MCDC-exported source histograms

1. Run the MCDC diode example and produce `output.h5`.
2. Prepare the Geant4 source files:

```bash
cd /Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/geant4_diode_example
/Users/massimolarsen/opt/anaconda3/envs/mcdc-env/bin/python prepare_mcdc_source.py
```

By default that reads:
- `/Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/output.h5`

and writes:
- `build/spectrum.mac`
- `build/angle_spectrum.mac`
- `build/mcdc_source/manifest.json`

You can point it at a different MCDC output with:

```bash
/Users/massimolarsen/opt/anaconda3/envs/mcdc-env/bin/python prepare_mcdc_source.py /path/to/output.h5
```

3. Run:

```bash
cd /Users/massimolarsen/Documents/Repos/MCDC/examples/testSiliconDiode/geant4_diode_example/build
./mcdc_diode_geant4 run_diode.mac
```

To visualize a few MCDC-driven events instead of running the full 1000-event test:

```bash
./mcdc_diode_geant4 inspect_diode.mac
```

## Output

The app prints:
- number of events
- total energy deposited in the sensitive volume
- mean sensitive-volume energy deposition per event

The visualization highlights:
- `DiodeBulk` in translucent blue
- `SensitiveVolume` in green
- `BEOL` in orange
- `SourcePlane` in red wireframe at the GPS injection plane

This is meant to be a lightweight coupling test problem, not a full production SEE workflow.
