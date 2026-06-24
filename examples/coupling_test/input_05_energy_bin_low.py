from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from common import (
    bridge_build_dir,
    build_vacuum_handoff_model,
    g4_world_size_from_handoff_zone,
    geant4_output_path,
    mcdc_output_name,
)

ENERGY_BINS_EV = np.array([0.0, 5.0e6, 20.0e6])
MU_BINS = np.linspace(-1.0, 1.0, 5)
AZI_BINS = np.linspace(-np.pi, np.pi, 5)
SURFACE_MESH = (1, 1)

target, target_cell = build_vacuum_handoff_model(
    mcdc,
    source_energy_ev=1.0e6,
)

mcdc.Tally(
    name="energy_low_src",
    cell=target_cell,
    scores=["current-in"],
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_EV,
    surface_mesh=SURFACE_MESH,
)

mcdc.settings.N_particle = 300
mcdc.settings.output_name = mcdc_output_name("ct05_energy_low")

mcdc.enable_geant4_handoff(
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(target),
    detector_size_mm=g4_world_size_from_handoff_zone(target, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    source_mode="distribution",
    n_geant4_particles=300,
    source_tally_name="energy_low_src",
    geant4_output_path=geant4_output_path("ct05_energy_low_geant4.h5"),
)

mcdc.run()
