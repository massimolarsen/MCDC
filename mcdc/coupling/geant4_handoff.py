from __future__ import annotations

import concurrent.futures
import pathlib
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from mcdc.coupling.geant4_bank import (
    build_bank_payload,
    empty_handoff_summary,
)
from mcdc.coupling.geant4_config import (
    CONFIGS,
    Geant4HandoffConfig,
    normalized_device_components,
    read_summary_hdf5,
)
from mcdc.coupling.geant4_distribution import build_source_distribution_payload
from mcdc.coupling import geant4_worker


def run_handoff_from_simulation(
    simulation: np.ndarray,
    data: np.ndarray | None = None,
) -> dict[str, Any]:
    configs = list(CONFIGS)
    validate_mpi_compatibility(simulation, configs)

    if configs[0].source_mode == "distribution" and data is None:
        raise RuntimeError("Distribution source mode requires tally data.")

    max_workers = _geant4_max_workers(simulation)
    n_threads = _geant4_n_threads(simulation)
    worker_count = 1 if max_workers == 1 else min(max_workers, len(configs))
    _print_handoff_progress(
        " Geant4 handoff: "
        f"regions={len(configs)} workers={worker_count} "
        f"threads_per_worker={n_threads} mode={configs[0].source_mode}"
    )
    if max_workers == 1 or len(configs) == 1:
        region_results, failures = _run_regions_serial(simulation, data, configs)
    else:
        region_results, failures = _run_regions_parallel(
            simulation, data, configs, max_workers
        )

    summary = _aggregate(region_results)
    if failures:
        raise RuntimeError("Geant4 handoff worker failure:\n" + "\n".join(failures))
    _print_timing_summary(summary)
    return summary


def validate_mpi_compatibility(
    simulation: np.ndarray,
    configs: list[Geant4HandoffConfig] | None = None,
) -> None:
    configs = list(CONFIGS) if configs is None else configs
    _validate_configs(configs)

    max_workers = _geant4_max_workers(simulation)
    if max_workers < 1:
        raise RuntimeError("Geant4 geant4_max_workers must be at least 1.")
    n_threads = _geant4_n_threads(simulation)
    if n_threads < 1:
        raise RuntimeError("Geant4 geant4_n_threads must be at least 1.")

    if int(simulation["mpi_size"]) <= 1:
        return

    if configs[0].source_mode == "bank":
        raise RuntimeError(
            "Geant4 bank source_mode does not support MPI MCDC runs; "
            "MPI bank gather/streaming is future work. Use "
            "source_mode='distribution' or run with one MPI process."
        )


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


def _run_regions_serial(
    simulation: np.ndarray,
    data: np.ndarray | None,
    configs: list[Geant4HandoffConfig],
) -> tuple[list[dict[str, Any]], list[str]]:
    region_results = []
    failures = []
    for cfg in configs:
        try:
            region_results.append(_run_one_region(simulation, data, cfg))
        except Exception as exc:
            failures.append(_format_region_failure(cfg, exc))
    return region_results, failures


def _run_regions_parallel(
    simulation: np.ndarray,
    data: np.ndarray | None,
    configs: list[Geant4HandoffConfig],
    max_workers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    worker_count = min(max_workers, len(configs))
    region_results: list[dict[str, Any] | None] = [None] * len(configs)
    failures: list[str | None] = [None] * len(configs)

    # Keep MPI on the main rank-0 thread. These worker threads only run
    # subprocesses and HDF5 payload/summary I/O; h5py serializes HDF5 access
    # internally, so scaling comes from the Geant4 subprocess runtime.
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_one_region, simulation, data, cfg): (idx, cfg)
            for idx, cfg in enumerate(configs)
        }
        for future in concurrent.futures.as_completed(futures):
            idx, cfg = futures[future]
            try:
                region_results[idx] = future.result()
            except Exception as exc:
                failures[idx] = _format_region_failure(cfg, exc)

    return (
        [result for result in region_results if result is not None],
        [failure for failure in failures if failure is not None],
    )


def _format_region_failure(cfg: Geant4HandoffConfig, exc: Exception) -> str:
    message = str(exc)
    return message if message.startswith(f"{cfg.name}:") else f"{cfg.name}: {message}"


def _geant4_max_workers(simulation: np.ndarray) -> int:
    settings = simulation["settings"]
    try:
        return int(settings["geant4_max_workers"])
    except (KeyError, TypeError, ValueError):
        return 1


def _geant4_n_threads(simulation: np.ndarray) -> int:
    settings = simulation["settings"]
    try:
        return int(settings["geant4_n_threads"])
    except (KeyError, TypeError, ValueError):
        return 1


def _print_handoff_progress(message: str) -> None:
    print(message)
    sys.stdout.flush()


def _run_one_region(
    simulation: np.ndarray,
    data: np.ndarray | None,
    cfg: Geant4HandoffConfig,
) -> dict[str, Any]:
    _print_handoff_progress(f" Geant4 region '{cfg.name}': preparing source")
    payload = _build_region_payload(simulation, data, cfg)
    if payload is None:
        summary = empty_handoff_summary("bank")
        summary["handoff_bank_size"] = 0
        summary["name"] = cfg.name
        _print_handoff_progress(f" Geant4 region '{cfg.name}': skipped empty bank")
        return summary
    if payload.get("status") == "skipped_empty_handoff":
        if cfg.geant4_output_path:
            geant4_worker.write_summary_hdf5(payload, cfg.geant4_output_path)
        _print_handoff_progress(
            f" Geant4 region '{cfg.name}': skipped empty distribution"
        )
        return payload

    payload_path = _payload_path(simulation, cfg)
    retain_payload = _geant4_payload_dir(simulation) != ""
    user_output = bool(cfg.geant4_output_path)
    output_path = pathlib.Path(
        cfg.geant4_output_path or _temporary_path(f"mcdc_g4_{cfg.name}_out_", ".h5")
    )
    payload["geant4_output_path"] = str(output_path)
    geant4_worker.write_payload(payload_path, payload)
    if retain_payload:
        _print_handoff_progress(
            f" Geant4 region '{cfg.name}': payload saved {payload_path}"
        )

    worker_path = pathlib.Path(geant4_worker.__file__).resolve()
    _print_handoff_progress(
        f" Geant4 region '{cfg.name}': worker started "
        f"source_size={payload['source_size']} output={output_path}"
    )
    worker_wall_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(worker_path), str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    worker_wall_s = time.perf_counter() - worker_wall_start

    if result.returncode != 0:
        if not user_output:
            output_path.unlink(missing_ok=True)
        _print_handoff_progress(
            f" Geant4 region '{cfg.name}': worker failed "
            f"return_code={result.returncode}"
        )
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

    summary["worker_wall_s"] = worker_wall_s
    if user_output:
        geant4_worker.write_summary_hdf5(summary, str(output_path))

    if not retain_payload:
        pathlib.Path(payload_path).unlink(missing_ok=True)
    if not user_output:
        output_path.unlink(missing_ok=True)
    _print_handoff_progress(
        f" Geant4 region '{cfg.name}': worker finished "
        f"events_run={summary['events_run']} "
        f"worker_wall={worker_wall_s:.2f}s "
        f"beam_wall={float(summary.get('geant4_beam_wall_s', 0.0)):.2f}s "
        f"beam_cpu/wall={float(summary.get('geant4_beam_cpu_per_wall', 0.0)):.2f} "
        f"status={summary['status']}"
    )
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
        "envelope_material": cfg.envelope_material,
        "physics_list": cfg.physics_list,
        "random_seed": _random_seed(cfg),
        "n_geant4_threads": _geant4_n_threads(simulation),
    }
    components = normalized_device_components(cfg)
    payload["component_names"] = np.asarray(
        [component["name"] for component in components], dtype=object
    )
    payload["component_materials"] = np.asarray(
        [component["material"] for component in components], dtype=object
    )
    payload["component_parents"] = np.asarray(
        [component["parent"] for component in components], dtype=object
    )
    payload["component_centers_mm"] = np.asarray(
        [component["center_mm"] for component in components], dtype=np.float64
    )
    payload["component_sizes_mm"] = np.asarray(
        [component["size_mm"] for component in components], dtype=np.float64
    )
    payload["component_score"] = np.asarray(
        [component["score"] for component in components], dtype=np.bool_
    )
    payload.update(source)
    return payload


def _random_seed(cfg: Geant4HandoffConfig) -> int:
    if cfg.random_seed is None:
        return secrets.randbelow(2_147_483_646) + 1

    seed = int(cfg.random_seed)
    if seed <= 0:
        raise RuntimeError("Geant4 random_seed must be a positive integer.")
    return seed


def _payload_path(simulation: np.ndarray, cfg: Geant4HandoffConfig) -> pathlib.Path:
    payload_dir = _geant4_payload_dir(simulation)
    if not payload_dir:
        return _temporary_path(f"mcdc_g4_{cfg.name}_", ".h5")

    directory = pathlib.Path(payload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cfg.name}_payload.h5"


def _geant4_payload_dir(simulation: np.ndarray) -> str:
    settings = simulation["settings"]
    try:
        return str(settings["geant4_payload_dir"])
    except (KeyError, TypeError, ValueError):
        return ""


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
        if all(
            region["status"] in {"ok", "skipped_empty_handoff"} for region in regions
        )
        else "error"
    )
    summary = {
        "status": status,
        "n_regions": len(regions),
        "source_size": sum(int(region["source_size"]) for region in regions),
        "loaded_primaries": sum(int(region["loaded_primaries"]) for region in regions),
        "events_run": sum(int(region["events_run"]) for region in regions),
        "regions": regions,
    }
    summary["worker_wall_s_sum"] = sum(
        float(region.get("worker_wall_s", 0.0)) for region in regions
    )
    summary["geant4_beam_wall_s_sum"] = sum(
        float(region.get("geant4_beam_wall_s", 0.0)) for region in regions
    )
    summary["geant4_beam_cpu_s_sum"] = sum(
        float(region.get("geant4_beam_cpu_s", 0.0)) for region in regions
    )
    return summary


def _print_timing_summary(summary: dict[str, Any]) -> None:
    regions = summary.get("regions", [])
    if not regions:
        return

    _print_handoff_progress(" Geant4 timing:")
    _print_handoff_progress(
        "   region        events   worker_wall    init_wall    load_wall"
        "    beam_wall     beam_cpu  cpu/wall"
    )
    for region in regions:
        _print_handoff_progress(
            f"   {str(region['name']):<8} "
            f"{int(region['events_run']):>9} "
            f"{float(region.get('worker_wall_s', 0.0)):>11.2f}s "
            f"{float(region.get('geant4_init_wall_s', 0.0)):>10.2f}s "
            f"{float(region.get('geant4_source_load_wall_s', 0.0)):>10.2f}s "
            f"{float(region.get('geant4_beam_wall_s', 0.0)):>10.2f}s "
            f"{float(region.get('geant4_beam_cpu_s', 0.0)):>10.2f}s "
            f"{float(region.get('geant4_beam_cpu_per_wall', 0.0)):>8.2f}"
        )
    _print_handoff_progress(
        " Geant4 timing totals: "
        f"sum_worker_wall={float(summary.get('worker_wall_s_sum', 0.0)):.2f}s "
        f"sum_beam_wall={float(summary.get('geant4_beam_wall_s_sum', 0.0)):.2f}s "
        f"sum_beam_cpu={float(summary.get('geant4_beam_cpu_s_sum', 0.0)):.2f}s"
    )
