from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import h5py
import numpy as np
from typing import Any


@dataclass
class Geant4HandoffConfig:
    name: str = "geant4"
    bridge_build_dir: str = ""
    world_size_mm: tuple[float, float, float] = (100.0, 100.0, 100.0)
    detector_size_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    detector_material: str = "G4_Si"
    envelope_material: str = "G4_Galactic"
    device_components: list[dict[str, Any]] = field(default_factory=list)
    physics_list: str = "QGSP_BIC"
    source_mode: str = "bank"
    n_geant4_particles: int = 0
    source_tally_name: str = ""
    geant4_output_path: str = ""
    random_seed: int | None = None


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


def has_configs() -> bool:
    return len(CONFIGS) > 0


def normalized_device_components(cfg: Geant4HandoffConfig) -> list[dict[str, Any]]:
    components = list(cfg.device_components)
    if not components:
        components = [
            {
                "name": "detector",
                "material": cfg.detector_material,
                "center_mm": [0.0, 0.0, 0.0],
                "size_mm": list(cfg.detector_size_mm),
                "parent": "",
                "score": True,
            }
        ]

    normalized = []
    names = set()
    for component in components:
        name = str(component.get("name", ""))
        material = str(component.get("material", ""))
        center = np.asarray(component.get("center_mm", []), dtype=np.float64)
        size = np.asarray(component.get("size_mm", []), dtype=np.float64)
        parent = str(component.get("parent", ""))
        score = bool(component.get("score", False))

        if not name:
            raise RuntimeError("Geant4 device component name cannot be empty.")
        if name in names:
            raise RuntimeError(f"Duplicate Geant4 device component name: {name}.")
        names.add(name)
        if not material:
            raise RuntimeError(
                f"Geant4 device component '{name}' material cannot be empty."
            )
        if center.shape != (3,):
            raise RuntimeError(
                f"Geant4 device component '{name}' center_mm must have length 3."
            )
        if size.shape != (3,):
            raise RuntimeError(
                f"Geant4 device component '{name}' size_mm must have length 3."
            )
        if not np.all(np.isfinite(center)):
            raise RuntimeError(
                f"Geant4 device component '{name}' center_mm must be finite."
            )
        if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise RuntimeError(
                f"Geant4 device component '{name}' size_mm entries must be positive."
            )

        normalized.append(
            {
                "name": name,
                "material": material,
                "center_mm": center,
                "size_mm": size,
                "parent": parent,
                "score": score,
            }
        )
    _validate_component_parents(normalized)
    return normalized


def _validate_component_parents(components: list[dict[str, Any]]) -> None:
    names = {component["name"] for component in components}
    parent_by_name = {
        component["name"]: component["parent"] for component in components
    }

    for component in components:
        name = component["name"]
        parent = component["parent"]
        if parent == name:
            raise RuntimeError(
                f"Geant4 device component '{name}' cannot parent itself."
            )
        if parent and parent not in names:
            raise RuntimeError(
                f"Geant4 device component '{name}' references unknown parent '{parent}'."
            )

    for component in components:
        seen = set()
        name = component["name"]
        parent = parent_by_name[name]
        while parent:
            if parent in seen:
                raise RuntimeError(
                    f"Geant4 device component parent cycle includes '{parent}'."
                )
            seen.add(parent)
            parent = parent_by_name[parent]


def _decode_hdf5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        decode = np.vectorize(
            lambda item: item.decode("utf-8") if isinstance(item, bytes) else item
        )
        return decode(value)
    return value


def read_summary_hdf5(output_path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with h5py.File(output_path, "r") as file:
        for key, dataset in file.items():
            summary[key] = _decode_hdf5_value(dataset[()])
    return summary
