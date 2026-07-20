from __future__ import annotations

from typing import Any

import numpy as np


def convert_handoff_bank_to_geant4(particles: np.ndarray) -> np.ndarray:
    # require neutron-only handoff particles for now
    if np.any(particles["particle_type"] != 0):
        raise RuntimeError(
            "Handoff bank contains non-neutron particles; current Geant4 bridge supports neutrons only."
        )

    bank = np.empty((len(particles), 10), dtype=np.float64)

    # particle_id, x_mm, y_mm, z_mm, ux, uy, uz, E_MeV, weight, time_ns
    bank[:, 0] = 2112.0  # neutron id
    bank[:, 1] = particles["x"] * 10.0  # cm to mm
    bank[:, 2] = particles["y"] * 10.0  # cm to mm
    bank[:, 3] = particles["z"] * 10.0  # cm to mm
    bank[:, 4] = particles["ux"]
    bank[:, 5] = particles["uy"]
    bank[:, 6] = particles["uz"]
    bank[:, 7] = particles["E"] * 1.0e-6  # eV to MeV
    bank[:, 8] = particles["w"]
    bank[:, 9] = particles["t"] * 1.0e9  # s to ns
    return bank


def build_bank_payload(simulation: np.ndarray) -> dict[str, Any] | None:
    # read exact handoff bank size
    N = int(simulation["bank_handoff"]["size"][0])
    if N <= 0:
        return None

    # convert handoff bank to bridge primary array
    handoff_particles = simulation["bank_handoff"]["particle_data"][:N]
    geant4_bank = convert_handoff_bank_to_geant4(handoff_particles)
    return {
        "source_mode": "bank",
        "source_size": N,
        "bank": geant4_bank,
    }


def empty_handoff_summary(source_mode: str = "bank") -> dict[str, Any]:
    return {
        "source_mode": source_mode,
        "source_size": 0,
        "loaded_primaries": 0,
        "events_run": 0,
        "total_edep_mev": 0.0,
        "dose_gy": 0.0,
        "edep_spectrum_edges_mev": np.asarray([], dtype=np.float64),
        "edep_spectrum_counts": np.asarray([], dtype=np.int64),
        "edep_spectrum_edep_mev": np.asarray([], dtype=np.float64),
        "edep_spectrum_underflow": 0,
        "edep_spectrum_overflow": 0,
        "status": "skipped_empty_handoff",
    }
