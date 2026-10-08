from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Any

import h5py
import numpy as np

try:
    from .geant4_config import _decode_hdf5_value
    from .seu_diagnostics import write_diagnostic_tables
except ImportError:
    from geant4_config import _decode_hdf5_value
    from seu_diagnostics import write_diagnostic_tables


ARRAY_FIELDS = {
    "bank",
    "box_bounds_mm",
    "component_centers_mm",
    "component_materials",
    "component_names",
    "component_parents",
    "component_score",
    "component_sizes_mm",
    "mu_edges",
    "azi_edges",
    "energy_edges_mev",
    "weights",
    "handoff_species_pdg",
    "handoff_species_count",
    "handoff_species_weight",
}


# per-species distribution source blocks, stored as groups sources/<i>
SOURCE_ARRAY_FIELDS = {"mu_edges", "azi_edges", "energy_edges_mev", "weights"}


def write_payload(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    with h5py.File(path, "w") as file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in payload.items():
            if key == "sources":
                for i, block in enumerate(value):
                    group = file.create_group(f"sources/{i}")
                    for block_key, block_value in block.items():
                        if block_key in SOURCE_ARRAY_FIELDS:
                            group.create_dataset(block_key, data=block_value)
                        else:
                            group.attrs[block_key] = block_value
            elif key in ARRAY_FIELDS:
                if key in {
                    "component_names",
                    "component_materials",
                    "component_parents",
                }:
                    file.create_dataset(
                        key, data=np.asarray(value, dtype=object), dtype=string_dtype
                    )
                else:
                    file.create_dataset(key, data=value)
            else:
                file.attrs[key] = value


def read_payload(path: str | pathlib.Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    with h5py.File(path, "r") as file:
        for key, value in file.attrs.items():
            payload[key] = _decode_hdf5_value(value)
        for key, item in file.items():
            if key == "sources":
                payload[key] = [
                    _read_source_block(item[name])
                    for name in sorted(item.keys(), key=int)
                ]
            else:
                payload[key] = _decode_hdf5_value(item[()])
    return payload


def _read_source_block(group) -> dict[str, Any]:
    block = {key: _decode_hdf5_value(value) for key, value in group.attrs.items()}
    for key, dataset in group.items():
        block[key] = dataset[()]
    return block


def source_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-species source blocks, reading older single-neutron payloads."""
    if "sources" in payload:
        return list(payload["sources"])
    return [
        {
            "pdg": 2112,
            "tally_name": payload.get("source_tally_name", ""),
            "mu_edges": payload["mu_edges"],
            "azi_edges": payload["azi_edges"],
            "energy_edges_mev": payload["energy_edges_mev"],
            "weights": payload["weights"],
            "Nu": payload["Nu"],
            "Nv": payload["Nv"],
            "n_events": payload["n_events"],
            "total_weight": payload.get("source_total_weight", 0.0),
        }
    ]


def load_bridge(build_dir: str):
    build_path = pathlib.Path(build_dir)
    # allow tests to provide a lightweight stub bridge beside the compiled module
    module_path = build_path / "geant4_bridge.py"
    if not module_path.exists():
        module_path = None
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            candidate = build_path / f"geant4_bridge{suffix}"
            if candidate.exists():
                module_path = candidate
                break
    if module_path is None:
        raise RuntimeError(
            f"Could not find built geant4_bridge module under: {build_path}"
        )

    spec = importlib.util.spec_from_file_location("geant4_bridge", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")

    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def write_summary_hdf5(summary: dict[str, Any], output_path: str) -> None:
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in summary.items():
            if isinstance(value, str):
                file.create_dataset(key, data=value, dtype=string_dtype)
            elif isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
                file.create_dataset(key, data=value.astype(object), dtype=string_dtype)
            else:
                file.create_dataset(key, data=value)


def result_summary(
    results,
    payload: dict[str, Any],
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    source_mode = str(payload["source_mode"])
    source_size = int(payload["source_size"])
    summary = {
        "name": str(payload["name"]),
        "source_mode": source_mode,
        "source_size": source_size,
        "loaded_primaries": int(results.loaded_primaries),
        "events_run": int(results.last_events_run),
        "total_edep_mev": float(results.last_total_edep_mev),
        "dose_gy": float(results.last_dose_gy),
        "edep_spectrum_edges_mev": np.asarray(
            results.edep_spectrum_edges_mev, dtype=np.float64
        ),
        "edep_spectrum_counts": np.asarray(
            results.edep_spectrum_counts, dtype=np.int64
        ),
        "edep_spectrum_edep_mev": np.asarray(
            results.edep_spectrum_edep_mev, dtype=np.float64
        ),
        "edep_spectrum_underflow": int(results.edep_spectrum_underflow),
        "edep_spectrum_overflow": int(results.edep_spectrum_overflow),
        "status": "ok",
        "random_seed": int(payload["random_seed"]),
        "n_geant4_threads": int(payload["n_geant4_threads"]),
        "component_names": np.asarray(results.component_names, dtype=str),
        "component_edep_mev": np.asarray(
            results.component_edep_mev, dtype=np.float64
        ),
        "component_mass_kg": np.asarray(
            results.component_mass_kg, dtype=np.float64
        ),
        "component_dose_gy": np.asarray(
            results.component_dose_gy, dtype=np.float64
        ),
        "geant4_version": str(results.geant4_version),
        "physics_list": str(results.physics_list),
        "em_production_cut_mm": float(results.em_production_cut_mm),
        "proton_production_cut_mm": float(results.proton_production_cut_mm),
        "electronics_cut_materials": np.asarray(
            results.electronics_cut_materials, dtype=str
        ),
        "electronics_cut_energy_mev": np.asarray(
            results.electronics_cut_energy_mev, dtype=np.float64
        ).reshape(-1, 4),
        "electronics_cut_particles": np.asarray(["gamma", "e-", "e+", "proton"]),
        "component_niel_mev": np.asarray(results.component_niel_mev, dtype=np.float64),
        "component_ionizing_mev": np.asarray(
            results.component_ionizing_mev, dtype=np.float64
        ),
        "component_ionizing_sum_sq_mev2": np.asarray(
            results.component_ionizing_sum_sq_mev2, dtype=np.float64
        ),
        "seu_species_names": np.asarray(results.seu_species_names, dtype=str),
        "component_species_ionizing_mev": np.asarray(
            results.component_species_ionizing_mev, dtype=np.float64
        ).reshape(len(results.component_names), len(results.seu_species_names)),
        "component_species_ionizing_sum_sq_mev2": np.asarray(
            results.component_species_ionizing_sum_sq_mev2, dtype=np.float64
        ).reshape(len(results.component_names), len(results.seu_species_names)),
        "component_species_positive_events": np.asarray(
            results.component_species_positive_events, dtype=np.int64
        ).reshape(len(results.component_names), len(results.seu_species_names)),
        "component_primary_ionizing_mev": np.asarray(
            results.component_primary_ionizing_mev, dtype=np.float64
        ),
        "component_secondary_ionizing_mev": np.asarray(
            results.component_secondary_ionizing_mev, dtype=np.float64
        ),
        "component_event_ionizing_edges_mev": np.asarray(
            results.component_event_ionizing_edges_mev, dtype=np.float64
        ),
        "component_event_ionizing_count": np.asarray(
            results.component_event_ionizing_count, dtype=np.float64
        ).reshape(len(results.component_names), -1),
        "component_event_ionizing_sumw": np.asarray(
            results.component_event_ionizing_sumw, dtype=np.float64
        ).reshape(len(results.component_names), -1),
        "component_event_ionizing_sumw2": np.asarray(
            results.component_event_ionizing_sumw2, dtype=np.float64
        ).reshape(len(results.component_names), -1),
        **_primary_species_results(results),
        "record_seu_events": bool(payload.get("record_seu_events", False)),
        "diagnostic_min_Eion_mev": float(
            payload.get("diagnostic_min_Eion_mev", 0.001)
        ),
        "rng_state_min_Eion_mev": float(payload.get("rng_state_min_Eion_mev", 0.0)),
    }
    if timings is not None:
        summary.update(timings)
    if source_mode == "bank":
        summary["handoff_bank_size"] = source_size
        for key in (
            "handoff_species_pdg",
            "handoff_species_count",
            "handoff_species_weight",
        ):
            if key in payload:
                summary[key] = np.asarray(payload[key])
    else:
        summary["source_tally_name"] = str(payload["source_tally_name"])
        summary["source_total_weight"] = float(payload["source_total_weight"])
        blocks = source_blocks(payload)
        summary["source_species_pdg"] = np.asarray(
            [int(block["pdg"]) for block in blocks], dtype=np.int64
        )
        summary["source_species_n_events"] = np.asarray(
            [int(block["n_events"]) for block in blocks], dtype=np.int64
        )
        summary["source_species_weight"] = np.asarray(
            [float(block["total_weight"]) for block in blocks], dtype=np.float64
        )
        summary["source_species_tally_name"] = np.asarray(
            [str(block["tally_name"]) for block in blocks], dtype=object
        )
    return summary


def _primary_species_results(results) -> dict[str, np.ndarray]:
    # scores split by the species of each event's primary, in seu_species_names order
    n_components = len(results.component_names)
    n_species = len(results.seu_species_names)
    summary = {}
    for name in (
        "edep_mev",
        "ionizing_mev",
        "ionizing_sum_sq_mev2",
    ):
        values = getattr(results, f"component_primary_species_{name}")
        summary[f"component_primary_species_{name}"] = np.asarray(
            values, dtype=np.float64
        ).reshape(n_components, n_species)
    for name in ("count", "sumw", "sumw2"):
        values = getattr(results, f"component_primary_species_event_ionizing_{name}")
        summary[f"component_primary_species_event_ionizing_{name}"] = np.asarray(
            values, dtype=np.float64
        ).reshape(n_components, n_species, -1)
    return summary


def make_session_config(bridge, payload: dict[str, Any], diagnostic_dir=None):
    """Bridge SessionConfig for a payload; diagnostics go to diagnostic_dir."""
    session_cfg = bridge.SessionConfig()
    session_cfg.world_size_mm = list(payload["world_size_mm"])
    session_cfg.detector_size_mm = list(payload["detector_size_mm"])
    session_cfg.detector_material = str(payload["detector_material"])
    session_cfg.envelope_material = str(payload["envelope_material"])
    session_cfg.physics_list = str(payload["physics_list"])
    session_cfg.random_seed = int(payload["random_seed"])
    session_cfg.n_threads = int(payload["n_geant4_threads"])
    session_cfg.em_production_cut_mm = float(payload.get("em_production_cut_mm", 0.0))
    session_cfg.record_seu_events = diagnostic_dir is not None
    session_cfg.diagnostic_min_Eion_mev = float(
        payload.get("diagnostic_min_Eion_mev", 0.001)
    )
    if diagnostic_dir is not None:
        session_cfg.diagnostic_dir = str(diagnostic_dir)
        # only bridge builds with event replay capture RNG states; payloads
        # written before the capture existed have no threshold
        if hasattr(session_cfg, "rng_state_min_Eion_mev"):
            session_cfg.rng_state_min_Eion_mev = float(
                payload.get("rng_state_min_Eion_mev", 0.0)
            )
    session_cfg.device_components = _bridge_components(bridge, payload)
    return session_cfg


def load_source(session, payload: dict[str, Any]) -> None:
    """Load the payload's bank or per-species distributions into a session."""
    if str(payload["source_mode"]) == "bank":
        session.load_primaries(np.ascontiguousarray(payload["bank"], dtype=np.float64))
        return
    # species run in load order, each for its own event count
    for block in source_blocks(payload):
        session.load_source_distribution(
            np.ascontiguousarray(payload["box_bounds_mm"], dtype=np.float64),
            np.ascontiguousarray(block["mu_edges"], dtype=np.float64),
            np.ascontiguousarray(block["azi_edges"], dtype=np.float64),
            np.ascontiguousarray(block["energy_edges_mev"], dtype=np.float64),
            np.ascontiguousarray(block["weights"], dtype=np.float64),
            int(block["Nu"]),
            int(block["Nv"]),
            int(block["n_events"]),
            int(block["pdg"]),
        )


def run_payload(path: str | pathlib.Path) -> dict[str, Any]:
    payload = read_payload(path)
    bridge = load_bridge(str(payload["bridge_build_dir"]))

    diagnostic_dir = None
    if bool(payload.get("record_seu_events", False)):
        output_dir = pathlib.Path(payload["geant4_output_path"]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_dir = pathlib.Path(
            tempfile.mkdtemp(prefix="mcdc_g4_seu_", dir=output_dir)
        )
    session_cfg = make_session_config(bridge, payload, diagnostic_dir)

    timings: dict[str, float] = {}

    init_wall_start = time.perf_counter()
    init_cpu_start = time.process_time()
    session = bridge.Session(session_cfg)
    session.initialize()
    timings["geant4_init_wall_s"] = time.perf_counter() - init_wall_start
    timings["geant4_init_cpu_s"] = time.process_time() - init_cpu_start

    load_wall_start = time.perf_counter()
    load_cpu_start = time.process_time()
    load_source(session, payload)
    timings["geant4_source_load_wall_s"] = time.perf_counter() - load_wall_start
    timings["geant4_source_load_cpu_s"] = time.process_time() - load_cpu_start

    beam_wall_start = time.perf_counter()
    beam_cpu_start = time.process_time()
    session.beam_on()
    timings["geant4_beam_wall_s"] = time.perf_counter() - beam_wall_start
    timings["geant4_beam_cpu_s"] = time.process_time() - beam_cpu_start
    if timings["geant4_beam_wall_s"] > 0.0:
        timings["geant4_beam_cpu_per_wall"] = (
            timings["geant4_beam_cpu_s"] / timings["geant4_beam_wall_s"]
        )
    else:
        timings["geant4_beam_cpu_per_wall"] = 0.0

    results = session.get_results()
    session.close()
    summary = result_summary(results, payload, timings)
    if summary["events_run"] != summary["source_size"]:
        raise RuntimeError(
            "Geant4 events_run does not match source_size: "
            f"{summary['events_run']} != {summary['source_size']}."
        )

    write_summary_hdf5(summary, str(payload["geant4_output_path"]))
    if diagnostic_dir is not None:
        write_diagnostic_tables(payload["geant4_output_path"], diagnostic_dir)
        shutil.rmtree(diagnostic_dir)
    return summary


def _bridge_components(bridge, payload: dict[str, Any]):
    names = np.asarray(payload["component_names"]).astype(str)
    materials = np.asarray(payload["component_materials"]).astype(str)
    parents = np.asarray(payload["component_parents"]).astype(str)
    centers = np.asarray(payload["component_centers_mm"], dtype=np.float64)
    sizes = np.asarray(payload["component_sizes_mm"], dtype=np.float64)
    scores = np.asarray(payload["component_score"], dtype=np.bool_)

    components = []
    for i, name in enumerate(names):
        component = bridge.DeviceComponent()
        component.name = str(name)
        component.material = str(materials[i])
        component.parent = str(parents[i])
        component.center_mm = [float(x) for x in centers[i]]
        component.size_mm = [float(x) for x in sizes[i]]
        component.score = bool(scores[i])
        components.append(component)
    return components


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    args = parser.parse_args(argv)
    run_payload(args.payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
