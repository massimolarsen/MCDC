from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import pathlib
from typing import Any

import numpy as np


@dataclass
class Geant4HandoffConfig:
    enabled: bool = False
    bridge_build_dir: str = ""
    world_size_mm: tuple[float, float, float] = (100.0, 100.0, 100.0)
    detector_size_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    detector_material: str = "G4_Si"
    physics_list: str = "QGSP_BIC"
    local_origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass
class _SessionState:
    config: Geant4HandoffConfig = field(default_factory=Geant4HandoffConfig)
    session: Any = None


_STATE = _SessionState()


def configure(**kwargs) -> None:
    for key, value in kwargs.items():
        if not hasattr(_STATE.config, key):
            raise ValueError(f"Unknown Geant4 coupling option: {key}")
        setattr(_STATE.config, key, value)


def reset() -> None:
    if _STATE.session is not None:
        _STATE.session.close()
    _STATE.session = None


def is_enabled() -> bool:
    return bool(_STATE.config.enabled)


def _load_local_geant4_bridge(build_dir: pathlib.Path):
    extension_candidates = sorted(build_dir.glob("geant4_bridge*.so"))
    if not extension_candidates:
        raise RuntimeError(
            f"Could not find built geant4_bridge extension under: {build_dir}"
        )

    module_path = extension_candidates[0]
    spec = importlib.util.spec_from_file_location("geant4_bridge", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _convert_handoff_bank_to_geant4(particles: np.ndarray) -> np.ndarray:
    if np.any(particles["particle_type"] != 0):
        raise RuntimeError(
            "Handoff bank contains non-neutron particles; current Geant4 bridge supports neutrons only."
        )

    bank = np.empty((len(particles), 10), dtype=np.float64)

    # particle_id, x_mm, y_mm, z_mm, ux, uy, uz, E_MeV, weight, time_ns
    bank[:, 0] = 2112.0 # neutron id
    bank[:, 1] = particles["x"] * 10.0 # convert to cm
    bank[:, 2] = particles["y"] * 10.0 # convert to cm
    bank[:, 3] = particles["z"] * 10.0 # convert to cm
    bank[:, 4] = particles["ux"]
    bank[:, 5] = particles["uy"]
    bank[:, 6] = particles["uz"]
    bank[:, 7] = particles["E"] * 1.0e-6 # convert to eV
    bank[:, 8] = particles["w"]
    bank[:, 9] = particles["t"] * 1.0e9 # convert to s
    return bank


def _ensure_session() -> Any:
    if _STATE.session is not None:
        return _STATE.session

    cfg = _STATE.config
    if not cfg.bridge_build_dir:
        raise RuntimeError(
            "Geant4 coupling is enabled but bridge_build_dir is empty. "
            "Set it via mcdc.enable_geant4_handoff(bridge_build_dir=...)."
        )

    build_dir = pathlib.Path(cfg.bridge_build_dir)
    if not build_dir.exists():
        raise RuntimeError(f"Geant4 bridge build directory does not exist: {build_dir}")

    g4 = _load_local_geant4_bridge(build_dir)

    session_cfg = g4.SessionConfig()
    session_cfg.world_size_mm = list(cfg.world_size_mm)
    session_cfg.detector_size_mm = list(cfg.detector_size_mm)
    session_cfg.detector_material = cfg.detector_material
    session_cfg.physics_list = cfg.physics_list

    local_frame = g4.LocalFrame()
    local_frame.origin_mm = list(cfg.local_origin_mm)
    local_frame.rotation = [list(row) for row in cfg.local_rotation]

    _STATE.session = g4.Session(session_cfg, local_frame)
    _STATE.session.initialize()
    return _STATE.session


def run_handoff_from_simulation(simulation: np.ndarray) -> dict[str, Any]:
    if simulation["mpi_size"] != 1:
        raise RuntimeError("Geant4 handoff coupling currently supports mpi_size == 1 only.")

    N = int(simulation["bank_handoff"]["size"][0])
    if N <= 0:
        return {
            "handoff_bank_size": 0,
            "loaded_primaries": 0,
            "events_run": 0,
            "status": "skipped_empty_handoff",
        }

    handoff_particles = simulation["bank_handoff"]["particle_data"][:N]
    geant4_bank = _convert_handoff_bank_to_geant4(handoff_particles)

    session = _ensure_session()
    session.load_primaries(geant4_bank)
    session.beam_on()
    results = session.get_results()

    summary = {
        "handoff_bank_size": N,
        "loaded_primaries": int(results.loaded_primaries),
        "events_run": int(results.last_events_run),
        "status": str(results.status),
        "primary_summary": {
            "first_primary": list(results.first_primary),
            "last_primary": list(results.last_primary),
            "min_position_mm": list(results.min_position_mm),
            "max_position_mm": list(results.max_position_mm),
            "min_direction": list(results.min_direction),
            "max_direction": list(results.max_direction),
            "min_energy_mev": float(results.min_energy_mev),
            "max_energy_mev": float(results.max_energy_mev),
        },
    }
    if summary["loaded_primaries"] != N:
        raise RuntimeError(
            "Geant4 coupling mismatch: loaded_primaries does not match handoff bank size."
        )

    return summary
