from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc
from common import (
    bridge_build_dir,
    build_vacuum_handoff_model,
    g4_world_size_from_handoff_zone,
    mcdc_output_name,
)

ENERGY_BINS_EV = np.array([0.0, 10.0e6, 20.0e6])
MU_BINS = np.array([-1.0, -0.25, 0.25, 1.0])
AZI_BINS = np.array([-np.pi, -0.25 * np.pi, 0.25 * np.pi, np.pi])

target, target_cell = build_vacuum_handoff_model(
    mcdc,
    source_energy_ev=14.0e6,
)

mcdc.Tally(
    name="mono_dir_src",
    cell=target_cell,
    scores=["current-in"],
    energy=ENERGY_BINS_EV,
)
mcdc.Tally(
    name="mono_mu",
    cell=target_cell,
    scores=["current-in"],
    mu=MU_BINS,
)
mcdc.Tally(
    name="mono_azi",
    cell=target_cell,
    scores=["current-in"],
    azi=AZI_BINS,
)

mcdc.settings.N_particle = 10000
mcdc.settings.output_name = mcdc_output_name("ct02_mono_dir_dist")


mcdc.run()
