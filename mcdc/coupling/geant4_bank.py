from __future__ import annotations

from typing import Any

import numpy as np

from mcdc.coupling import geant4_config


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


def run_bank_handoff(simulation: np.ndarray) -> dict[str, Any]:
    # read exact handoff bank size
    N = int(simulation["bank_handoff"]["size"][0])
    if N <= 0:
        return {
            "source_mode": "bank",
            "source_size": 0,
            "handoff_bank_size": 0,
            "loaded_primaries": 0,
            "events_run": 0,
            "status": "skipped_empty_handoff",
        }

    # convert handoff bank to bridge primary array
    handoff_particles = simulation["bank_handoff"]["particle_data"][:N]
    geant4_bank = convert_handoff_bank_to_geant4(handoff_particles)

    # use the cached geant4 bridge session
    session = geant4_config.get_session()
    session.load_primaries(geant4_bank)
    session.beam_on()
    results = session.get_results()

    # collect exact-bank handoff summary
    summary = {
        "source_mode": "bank",
        "source_size": N,
        "handoff_bank_size": N,
        "loaded_primaries": int(results.loaded_primaries),
        "events_run": int(results.last_events_run),
        "status": str(results.status),
    }

    # verify geant4 loaded every handoff particle
    if summary["loaded_primaries"] != N:
        raise RuntimeError(
            "Geant4 coupling mismatch: loaded_primaries does not match handoff bank size."
        )

    return summary
