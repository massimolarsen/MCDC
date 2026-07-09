from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import pathlib
import sys
from typing import Any

import h5py
import numpy as np


ARRAY_FIELDS = {
    "bank",
    "box_bounds_mm",
    "mu_edges",
    "azi_edges",
    "energy_edges_mev",
    "weights",
}


def write_payload(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    with h5py.File(path, "w") as file:
        for key, value in payload.items():
            if key in ARRAY_FIELDS:
                file.create_dataset(key, data=value)
            else:
                file.attrs[key] = value


def read_payload(path: str | pathlib.Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    with h5py.File(path, "r") as file:
        for key, value in file.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            payload[key] = value
        for key, dataset in file.items():
            payload[key] = dataset[()]
    return payload


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
        raise RuntimeError(f"Could not find built geant4_bridge module under: {build_path}")

    spec = importlib.util.spec_from_file_location("geant4_bridge", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {module_path}")

    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    return bridge


def write_summary_hdf5(summary: dict[str, Any], output_path: str) -> None:
    with h5py.File(output_path, "w") as file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in summary.items():
            if isinstance(value, str):
                file.create_dataset(key, data=value, dtype=string_dtype)
            else:
                file.create_dataset(key, data=value)


def result_summary(results, payload: dict[str, Any]) -> dict[str, Any]:
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
        "status": str(results.status),
        "random_seed": int(payload["random_seed"]),
    }
    if source_mode == "bank":
        summary["handoff_bank_size"] = source_size
    else:
        summary["source_tally_name"] = str(payload["source_tally_name"])
        summary["source_total_weight"] = float(payload["source_total_weight"])
    return summary


def run_payload(path: str | pathlib.Path) -> dict[str, Any]:
    payload = read_payload(path)
    bridge = load_bridge(str(payload["bridge_build_dir"]))

    session_cfg = bridge.SessionConfig()
    session_cfg.world_size_mm = list(payload["world_size_mm"])
    session_cfg.detector_size_mm = list(payload["detector_size_mm"])
    session_cfg.detector_material = str(payload["detector_material"])
    session_cfg.physics_list = str(payload["physics_list"])
    session_cfg.random_seed = int(payload["random_seed"])

    session = bridge.Session(session_cfg)
    session.initialize()
    if str(payload["source_mode"]) == "bank":
        session.load_primaries(np.ascontiguousarray(payload["bank"], dtype=np.float64))
    else:
        session.load_source_distribution(
            np.ascontiguousarray(payload["box_bounds_mm"], dtype=np.float64),
            np.ascontiguousarray(payload["mu_edges"], dtype=np.float64),
            np.ascontiguousarray(payload["azi_edges"], dtype=np.float64),
            np.ascontiguousarray(payload["energy_edges_mev"], dtype=np.float64),
            np.ascontiguousarray(payload["weights"], dtype=np.float64),
            int(payload["Nu"]),
            int(payload["Nv"]),
            int(payload["n_events"]),
        )

    session.beam_on()
    summary = result_summary(session.get_results(), payload)
    if summary["loaded_primaries"] != summary["source_size"]:
        raise RuntimeError(
            "Geant4 loaded_primaries does not match source_size: "
            f"{summary['loaded_primaries']} != {summary['source_size']}."
        )
    if summary["events_run"] != summary["source_size"]:
        raise RuntimeError(
            "Geant4 events_run does not match source_size: "
            f"{summary['events_run']} != {summary['source_size']}."
        )
    if summary["status"] != "ok":
        raise RuntimeError(f"Geant4 worker failed with status '{summary['status']}'.")

    write_summary_hdf5(summary, str(payload["geant4_output_path"]))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload")
    args = parser.parse_args(argv)
    run_payload(args.payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
