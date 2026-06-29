from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from common import (
    box_region,
    bridge_build_dir,
    g4_world_size_from_handoff_zone,
    geant4_output_path,
    mcdc_output_name,
)

ENERGY_BINS_EV = np.logspace(3.0, np.log10(14.01e6), 21)
MU_BINS = np.array([-1.0e-3, 1.0e-3])
AZI_BINS = np.array([-1.0e-3, 1.0e-3])
SURFACE_MESH = (1, 1)

vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"Si28": 0.0})
silicon = mcdc.Material(name="Silicon", nuclide_composition={"Si28": 1.0})

world = box_region(
    mcdc,
    "world",
    (-6.0, 6.0),
    (-3.0, 3.0),
    (-3.0, 3.0),
    boundary="vacuum",
)
cpu = box_region(mcdc, "cpu", (-3.0, -2.0), (-1.0, 1.0), (-1.0, 1.0))
silicon_slab = box_region(mcdc, "silicon_slab", (-.5, .5), (-1.0, 1.0), (-1.0, 1.0))
battery = box_region(mcdc, "battery", (1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))

cpu_cell = mcdc.Cell(
    name="CPU Handoff",
    region=cpu["region"],
    fill=vacuum,
)
battery_cell = mcdc.Cell(
    name="Battery Handoff",
    region=battery["region"],
    fill=vacuum,
)

mcdc.Cell(
    name="Silicon Between Devices",
    region=silicon_slab["region"],
    fill=silicon,
)

mcdc.Cell(
    name="Vacuum Outside Devices",
    region=world["region"]
    & ~cpu["region"]
    & ~silicon_slab["region"]
    & ~battery["region"],
    fill=vacuum,
)

mcdc.Source(
    name="Two Device Validation Source",
    x=[-5.0, -4.9],
    y=[-1.0, 1.0],
    z=[-1.0, 1.0],
    direction=[1.0, 0.0, 0.0],
    energy=14.0e6,
)

mcdc.Tally(
    name="cpu_src",
    cell=cpu_cell,
    scores=["current-in"],
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_EV,
    surface_mesh=SURFACE_MESH,
)
mcdc.Tally(
    name="battery_src",
    cell=battery_cell,
    scores=["current-in"],
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_EV,
    surface_mesh=SURFACE_MESH,
)

mcdc.settings.N_particle = 500
mcdc.settings.output_name = mcdc_output_name("ct06_two_region_dist")

mcdc.enable_geant4_handoff(
    name="cpu",
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(cpu),
    detector_size_mm=g4_world_size_from_handoff_zone(cpu, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    source_mode="distribution",
    n_geant4_particles=500,
    source_tally_name="cpu_src",
    geant4_output_path=geant4_output_path("ct06_cpu_geant4.h5"),
)
mcdc.add_geant4_handoff(
    name="battery",
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(battery),
    detector_size_mm=g4_world_size_from_handoff_zone(battery, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    source_mode="distribution",
    n_geant4_particles=500,
    source_tally_name="battery_src",
    geant4_output_path=geant4_output_path("ct06_battery_geant4.h5"),
)

mcdc.run()
