from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from common import (
    bridge_build_dir,
    build_cube_in_cube_model,
    g4_world_size_from_handoff_zone,
    geant4_output_path,
    mcdc_output_name,
)

ENERGY_BINS_EV = np.concatenate(([0.0], np.logspace(0.0, np.log10(20.0e6), 12)))
MU_BINS = np.linspace(-1.0, 1.0, 5)
AZI_BINS = np.linspace(-np.pi, np.pi, 5)
SURFACE_MESH = (8, 8)
N_MCDC_PARTICLES = 2000
N_GEANT4_PARTICLES = 2000
ACTIVE_BANK_BUFFER = 10000

sram, sram_cell = build_cube_in_cube_model(mcdc)

mcdc.Tally(
    name="cube_spatial_src",
    cell=sram_cell,
    scores=["current-in"],
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_EV,
    surface_mesh=SURFACE_MESH,
)

mcdc.settings.N_particle = N_MCDC_PARTICLES
mcdc.settings.active_bank_buffer = ACTIVE_BANK_BUFFER
mcdc.settings.output_name = mcdc_output_name("ct04_cube_spatial")

mcdc.enable_geant4_handoff(
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(sram),
    detector_size_mm=g4_world_size_from_handoff_zone(sram, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    source_mode="distribution",
    n_geant4_particles=N_GEANT4_PARTICLES,
    source_tally_name="cube_spatial_src",
    distribution_box_cm=(
        sram["x_span_cm"],
        sram["y_span_cm"],
        sram["z_span_cm"],
    ),
    geant4_output_path=geant4_output_path("ct04_cube_spatial_geant4.h5"),
)

mcdc.run()
