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

sram, sram_cell = build_cube_in_cube_model(mcdc, handoff=True)

mcdc.Tally(
    name="cube_bank_current",
    cell=sram_cell,
    scores=["current-in"],
    energy=ENERGY_BINS_EV,
)

mcdc.settings.N_particle = 2000
mcdc.settings.handoff_bank_buffer = 2 * mcdc.settings.N_particle
mcdc.settings.active_bank_buffer = 10000
mcdc.settings.output_name = mcdc_output_name("ct03_cube_bank")

mcdc.enable_geant4_handoff(
    bridge_build_dir=bridge_build_dir(),
    world_size_mm=g4_world_size_from_handoff_zone(sram),
    detector_size_mm=g4_world_size_from_handoff_zone(sram, padding_scale=1.0),
    detector_material="G4_Si",
    physics_list="QGSP_BIC",
    geant4_output_path=geant4_output_path("ct03_cube_bank_geant4.h5"),
)

mcdc.run()
