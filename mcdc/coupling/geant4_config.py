from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import h5py
import numpy as np
from typing import Any

from mcdc.constant import PARTICLE_ELECTRON, PARTICLE_NEUTRON, PARTICLE_PROTON

# MC/DC particle types handed to Geant4, and their PDG codes
PARTICLE_PDG = {
    PARTICLE_NEUTRON: 2112,
    PARTICLE_ELECTRON: 11,
    PARTICLE_PROTON: 2212,
}
PARTICLE_TYPE_BY_NAME = {
    "neutron": PARTICLE_NEUTRON,
    "electron": PARTICLE_ELECTRON,
    "proton": PARTICLE_PROTON,
}


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
    em_production_cut_mm: float = 0.0
    record_seu_events: bool = False
    diagnostic_min_Eion_mev: float = 0.001
    source_mode: str = "bank"
    # Distribution mode: one current-in source tally and event count per species,
    # as dicts {"particle": "neutron"|"electron"|"proton", "tally": name,
    # "n_events": count}. source_tally_name/n_geant4_particles give one entry.
    species_sources: list[dict[str, Any]] = field(default_factory=list)
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


def distribution_sources(cfg: Geant4HandoffConfig) -> list[dict[str, Any]]:
    """Return the distribution source entries as (particle, tally, n_events).

    ``particle`` is None for the source_tally_name shorthand, which takes the
    species from the tally's particle filter.
    """
    if not cfg.species_sources:
        return [
            {
                "particle": None,
                "tally": cfg.source_tally_name,
                "n_events": int(cfg.n_geant4_particles),
            }
        ]

    if cfg.source_tally_name or cfg.n_geant4_particles:
        raise RuntimeError(
            "Use either species_sources or source_tally_name/n_geant4_particles, "
            "not both."
        )

    sources = []
    for entry in cfg.species_sources:
        unknown = set(entry) - {"particle", "tally", "n_events"}
        if unknown:
            raise RuntimeError(f"Unknown species_sources keys: {sorted(unknown)}")
        particle = str(entry.get("particle", ""))
        if particle not in PARTICLE_TYPE_BY_NAME:
            raise RuntimeError(
                f"species_sources particle must be one of "
                f"{sorted(PARTICLE_TYPE_BY_NAME)}, got '{particle}'."
            )
        sources.append(
            {
                "particle": particle,
                "tally": str(entry.get("tally", "")),
                "n_events": int(entry.get("n_events", 0)),
            }
        )

    particles = [source["particle"] for source in sources]
    if len(particles) != len(set(particles)):
        raise RuntimeError("species_sources must list each particle at most once.")
    return sources


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

    return [
        {
            "name": str(component["name"]),
            "material": str(component["material"]),
            "center_mm": np.asarray(component["center_mm"], dtype=np.float64),
            "size_mm": np.asarray(component["size_mm"], dtype=np.float64),
            "parent": str(component.get("parent", "")),
            "score": bool(component.get("score", False)),
        }
        for component in components
    ]


def _decode_hdf5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        decode = np.vectorize(
            lambda item: item.decode("utf-8") if isinstance(item, bytes) else item,
            otypes=[object],
        )
        return decode(value)
    return value


def read_summary_hdf5(output_path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with h5py.File(output_path, "r") as file:
        for key, dataset in file.items():
            if isinstance(dataset, h5py.Group):
                continue
            summary[key] = _decode_hdf5_value(dataset[()])
    return summary
