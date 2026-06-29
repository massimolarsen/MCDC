from __future__ import annotations

from dataclasses import dataclass
import h5py
from typing import Any


@dataclass
class Geant4HandoffConfig:
    name: str = "geant4"
    bridge_build_dir: str = ""
    world_size_mm: tuple[float, float, float] = (100.0, 100.0, 100.0)
    detector_size_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    detector_material: str = "G4_Si"
    physics_list: str = "QGSP_BIC"
    source_mode: str = "bank"
    n_geant4_particles: int = 0
    source_tally_name: str = ""
    geant4_output_path: str = ""


CONFIGS: list[Geant4HandoffConfig] = []


def configure(**kwargs) -> None:
    # keep legacy single-region setup as a wrapper
    CONFIGS.clear()
    add_config(**kwargs)


def add_config(**kwargs) -> None:
    cfg = Geant4HandoffConfig()
    for key, value in kwargs.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown Geant4 coupling option: {key}")
        setattr(cfg, key, value)
    CONFIGS.append(cfg)


def clear_configs() -> None:
    CONFIGS.clear()


def read_summary_hdf5(output_path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with h5py.File(output_path, "r") as file:
        for key, dataset in file.items():
            value = dataset[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            summary[key] = value
    return summary
