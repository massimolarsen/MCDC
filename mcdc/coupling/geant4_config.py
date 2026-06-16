from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import pathlib
from typing import Any


@dataclass
class Geant4HandoffConfig:
    enabled: bool = False
    bridge_build_dir: str = ""
    world_size_mm: tuple[float, float, float] = (100.0, 100.0, 100.0)
    detector_size_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    detector_material: str = "G4_Si"
    physics_list: str = "QGSP_BIC"
    source_mode: str = "bank"
    n_geant4_particles: int = 0
    source_tally_name: str = ""
    distribution_box_cm: tuple[tuple[float, float], ...] | None = None


CONFIG = Geant4HandoffConfig()


def configure(**kwargs) -> None:
    for key, value in kwargs.items():
        if not hasattr(CONFIG, key):
            raise ValueError(f"Unknown Geant4 coupling option: {key}")
        setattr(CONFIG, key, value)


def create_session() -> Any:
    cfg = CONFIG
    if not cfg.bridge_build_dir:
        raise RuntimeError(
            "Geant4 coupling is enabled but bridge_build_dir is empty. "
            "Set it via mcdc.enable_geant4_handoff(bridge_build_dir=...)."
        )

    build_dir = pathlib.Path(cfg.bridge_build_dir)
    if not build_dir.exists():
        raise RuntimeError(f"Geant4 bridge build directory does not exist: {build_dir}")

    module_path = None
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = build_dir / f"geant4_bridge{suffix}"
        if candidate.exists():
            module_path = candidate
            break
    if module_path is None:
        raise RuntimeError(f"Could not find built geant4_bridge module under: {build_dir}")

    spec = importlib.util.spec_from_file_location("geant4_bridge", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")

    g4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g4)

    session_cfg = g4.SessionConfig()
    session_cfg.world_size_mm = list(cfg.world_size_mm)
    session_cfg.detector_size_mm = list(cfg.detector_size_mm)
    session_cfg.detector_material = cfg.detector_material
    session_cfg.physics_list = cfg.physics_list

    # Keep session lifetime per handoff target. Future multi-volume handoffs should
    # create one bridge session per target/process instead of sharing global state.
    session = g4.Session(session_cfg)
    session.initialize()
    return session


def primary_summary(results) -> dict[str, Any]:
    return {
        "first_primary": list(results.first_primary),
        "last_primary": list(results.last_primary),
        "min_position_mm": list(results.min_position_mm),
        "max_position_mm": list(results.max_position_mm),
        "min_direction": list(results.min_direction),
        "max_direction": list(results.max_direction),
        "min_energy_mev": float(results.min_energy_mev),
        "max_energy_mev": float(results.max_energy_mev),
    }
