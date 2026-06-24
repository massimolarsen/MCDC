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
N_MCDC_PARTICLES = 10000
ACTIVE_BANK_BUFFER = N_MCDC_PARTICLES * 2

target, target_cell = build_vacuum_handoff_model(
    mcdc,
    source_energy_ev=14.0e6,
    handoff_bank=True,
    source_y_span_cm=(-1.0, 1.0),
    source_z_span_cm=(-1.0, 1.0),
)

mcdc.Tally(
    name="validation_bank_current",
    cell=target_cell,
    scores=["current-in"],
    energy=ENERGY_BINS_EV,
)

mcdc.settings.N_particle = N_MCDC_PARTICLES
mcdc.settings.handoff_bank_buffer = 2 * mcdc.settings.N_particle
mcdc.settings.active_bank_buffer = ACTIVE_BANK_BUFFER
mcdc.settings.output_name = mcdc_output_name("ct03_validation_bank")

mcdc.enable_geant4_handoff(
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(target),
    detector_size_mm=g4_world_size_from_handoff_zone(target, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    geant4_output_path=geant4_output_path("ct03_validation_bank_geant4.h5"),
)

mcdc.run()
