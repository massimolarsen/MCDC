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

ENERGY_BINS_EV = np.array([13.99e6, 14.01e6])
MU_BINS = np.array([-1.0e-3, 1.0e-3])
AZI_BINS = np.array([-1.0e-3, 1.0e-3])
SURFACE_MESH = (4, 4)

target, target_cell = build_vacuum_handoff_model(
    mcdc,
    source_energy_ev=14.0e6,
)

mcdc.Tally(
    name="validation_dist_src",
    cell=target_cell,
    scores=["current-in"],
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_EV,
    surface_mesh=SURFACE_MESH,
)

mcdc.settings.N_particle = 2000
mcdc.settings.active_bank_buffer = 2 * mcdc.settings.N_particle
mcdc.settings.output_name = mcdc_output_name("ct03_validation_dist")

mcdc.enable_geant4_handoff(
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(target),
    detector_size_mm=g4_world_size_from_handoff_zone(target, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    source_mode="distribution",
    n_geant4_particles=2000,
    source_tally_name="validation_dist_src",
    distribution_box_cm=(
        target["x_span_cm"],
        target["y_span_cm"],
        target["z_span_cm"],
    ),
    geant4_output_path=geant4_output_path("ct03_validation_dist_geant4.h5"),
)

mcdc.run()
