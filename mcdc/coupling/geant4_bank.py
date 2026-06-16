from __future__ import annotations

from typing import Any

import numpy as np

from mcdc.coupling import geant4_config


def convert_handoff_bank_to_geant4(particles: np.ndarray) -> np.ndarray:
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
    N = int(simulation["bank_handoff"]["size"][0])
    if N <= 0:
        return {
            "handoff_bank_size": 0,
            "loaded_primaries": 0,
            "events_run": 0,
            "status": "skipped_empty_handoff",
        }

    handoff_particles = simulation["bank_handoff"]["particle_data"][:N]
    geant4_bank = convert_handoff_bank_to_geant4(handoff_particles)

    # Each handoff owns its Geant4 session so later targets do not share source state.
    session = geant4_config.create_session()
    try:
        session.load_primaries(geant4_bank)
        session.beam_on()
        results = session.get_results()
        summary = {
            "handoff_bank_size": N,
            "loaded_primaries": int(results.loaded_primaries),
            "events_run": int(results.last_events_run),
            "status": str(results.status),
            "primary_summary": geant4_config.primary_summary(results),
        }
    finally:
        session.close()

    if summary["loaded_primaries"] != N:
        raise RuntimeError(
            "Geant4 coupling mismatch: loaded_primaries does not match handoff bank size."
        )

    return summary
