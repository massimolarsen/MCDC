from __future__ import annotations

from typing import Any

import numpy as np

from mcdc.coupling.geant4_bank import run_bank_handoff
from mcdc.coupling.geant4_config import (
    CONFIG,
    Geant4HandoffConfig,
    configure,
)
from mcdc.coupling.geant4_distribution import run_distribution_handoff


def run_handoff_from_simulation(
    simulation: np.ndarray,
    data: np.ndarray | None = None,
) -> dict[str, Any]:
    if simulation["mpi_size"] != 1:
        raise RuntimeError(
            "Geant4 handoff coupling currently supports mpi_size == 1 only."
        )

    # Bank mode is the exact-particle replay path and remains the reference mode.
    if CONFIG.source_mode == "bank":
        return run_bank_handoff(simulation)

    # Distribution mode needs finalized tally means, which live in the data array.
    if CONFIG.source_mode != "distribution":
        raise RuntimeError("Geant4 source_mode must be 'bank' or 'distribution'.")
    if data is None:
        raise RuntimeError("Distribution source mode requires tally data.")
    return run_distribution_handoff(simulation, data, CONFIG)
