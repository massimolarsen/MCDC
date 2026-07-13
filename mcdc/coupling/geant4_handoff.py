from __future__ import annotations

import pathlib
import secrets
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from mcdc.coupling.geant4_bank import (
    build_bank_payload,
    empty_bank_summary,
    empty_handoff_summary,
)
from mcdc.coupling.geant4_config import (
    CONFIGS,
    Geant4HandoffConfig,
    add_config,
    clear_configs,
    configure,
    read_summary_hdf5,
)
from mcdc.coupling.geant4_distribution import build_source_distribution_payload
from mcdc.coupling import geant4_worker


def has_configs() -> bool:
    return len(CONFIGS) > 0


def add_handoff(**kwargs) -> None:
    add_config(**kwargs)


def disable() -> None:
    clear_configs()


def run_handoff_from_simulation(
    simulation: np.ndarray,
    data: np.ndarray | None = None,
) -> dict[str, Any]:
    # require serial mcdc state for geant4 handoff
    if simulation["mpi_size"] != 1:
        raise RuntimeError(
            "Geant4 handoff coupling currently supports mpi_size == 1 only."
        )

    configs = list(CONFIGS)
    _validate_configs(configs)

    if configs[0].source_mode == "distribution" and data is None:
        raise RuntimeError("Distribution source mode requires tally data.")

    region_results = []
    failures = []
    for index, cfg in enumerate(configs):
        try:
            region_results.append(_run_one_region(simulation, data, cfg, index))
        except Exception as exc:
            message = str(exc)
            failures.append(
                message if message.startswith(f"{cfg.name}:") else f"{cfg.name}: {message}"
            )

    summary = _aggregate(region_results)
    if failures:
        raise RuntimeError("Geant4 handoff worker failure:\n" + "\n".join(failures))
    return summary


def _validate_configs(configs: list[Geant4HandoffConfig]) -> None:
    if not configs:
        raise RuntimeError("Geant4 handoff is enabled but no regions are configured.")

    modes = {cfg.source_mode for cfg in configs}
    if not modes <= {"bank", "distribution"}:
        raise RuntimeError("Geant4 source_mode must be 'bank' or 'distribution'.")
    if len(modes) != 1:
        raise RuntimeError("Geant4 handoff cannot mix bank and distribution regions.")
    if "bank" in modes and len(configs) != 1:
        raise RuntimeError("Multiple Geant4 bank handoff regions are not supported.")

    names = [cfg.name for cfg in configs]
    if len(names) != len(set(names)):
        raise RuntimeError("Geant4 handoff region names must be unique.")

    output_paths = [cfg.geant4_output_path for cfg in configs if cfg.geant4_output_path]
    if len(output_paths) != len(set(output_paths)):
        raise RuntimeError("Geant4 output paths must be unique.")

    if "distribution" in modes:
        tally_names = [cfg.source_tally_name for cfg in configs]
        if len(tally_names) != len(set(tally_names)):
            raise RuntimeError("Geant4 source_tally_name values must be unique.")


def _run_one_region(
    simulation: np.ndarray,
    data: np.ndarray | None,
    cfg: Geant4HandoffConfig,
    index: int,
) -> dict[str, Any]:
    payload = _build_region_payload(simulation, data, cfg)
    if payload is None:
        summary = empty_bank_summary()
        summary["name"] = cfg.name
        return summary
    if payload.get("status") == "skipped_empty_handoff":
        if cfg.geant4_output_path:
            geant4_worker.write_summary_hdf5(payload, cfg.geant4_output_path)
        return payload

    payload_path = _temporary_path(f"mcdc_g4_{cfg.name}_", ".h5")
    user_output = bool(cfg.geant4_output_path)
    output_path = pathlib.Path(
        cfg.geant4_output_path or _temporary_path(f"mcdc_g4_{cfg.name}_out_", ".h5")
    )
    payload["geant4_output_path"] = str(output_path)
    geant4_worker.write_payload(payload_path, payload)

    worker_path = pathlib.Path(geant4_worker.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(worker_path), str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        if not user_output:
            output_path.unlink(missing_ok=True)
        raise RuntimeError(_worker_failure(cfg.name, result, payload_path, output_path))
    if not output_path.exists():
        raise RuntimeError(
            f"{cfg.name}: Geant4 worker succeeded but did not write {output_path}; "
            f"payload retained at {payload_path}"
        )

    try:
        summary = read_summary_hdf5(str(output_path))
    except Exception as exc:
        raise RuntimeError(
            f"{cfg.name}: could not read Geant4 output {output_path}: {exc}; "
            f"payload retained at {payload_path}"
        ) from exc

    pathlib.Path(payload_path).unlink(missing_ok=True)
    if not user_output:
        output_path.unlink(missing_ok=True)
    return summary


def _build_region_payload(
    simulation: np.ndarray,
    data: np.ndarray | None,
    cfg: Geant4HandoffConfig,
) -> dict[str, Any] | None:
    if cfg.source_mode == "bank":
        source = build_bank_payload(simulation)
        if source is None:
            return None
    else:
        try:
            source = build_source_distribution_payload(simulation, data, cfg)
        except RuntimeError as exc:
            if "zero total current-in weight" in str(exc):
                summary = empty_handoff_summary("distribution")
                summary["name"] = cfg.name
                summary["source_tally_name"] = cfg.source_tally_name
                summary["source_total_weight"] = 0.0
                return summary
            raise

    if source.get("status") == "skipped_empty_handoff":
        return source

    payload = {
        "name": cfg.name,
        "source_mode": cfg.source_mode,
        "bridge_build_dir": cfg.bridge_build_dir,
        "world_size_mm": cfg.world_size_mm,
        "detector_size_mm": cfg.detector_size_mm,
        "detector_material": cfg.detector_material,
        "physics_list": cfg.physics_list,
        "random_seed": _random_seed(cfg),
    }
    payload.update(source)
    return payload


def _random_seed(cfg: Geant4HandoffConfig) -> int:
    if cfg.random_seed is None:
        return secrets.randbelow(2_147_483_646) + 1

    seed = int(cfg.random_seed)
    if seed <= 0:
        raise RuntimeError("Geant4 random_seed must be a positive integer.")
    return seed


def _temporary_path(prefix: str, suffix: str) -> pathlib.Path:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    handle.close()
    return pathlib.Path(handle.name)


def _worker_failure(
    name: str,
    result: subprocess.CompletedProcess[str],
    payload_path: pathlib.Path,
    output_path: pathlib.Path,
) -> str:
    stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
    return (
        f"{name}: worker return code {result.returncode}; "
        f"output={output_path}; payload retained at {payload_path}; "
        f"stderr tail:\n{stderr_tail}"
    )


def _aggregate(regions: list[dict[str, Any]]) -> dict[str, Any]:
    status = (
        "ok"
        if all(region["status"] in {"ok", "skipped_empty_handoff"} for region in regions)
        else "error"
    )
    return {
        "status": status,
        "n_regions": len(regions),
        "source_size": sum(int(region["source_size"]) for region in regions),
        "loaded_primaries": sum(int(region["loaded_primaries"]) for region in regions),
        "events_run": sum(int(region["events_run"]) for region in regions),
        "regions": regions,
    }
