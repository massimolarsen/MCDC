from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import subprocess
import sys
import tempfile
import time


EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_PAYLOAD_DIR = EXAMPLE_DIR / "geant4_payloads"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rerun saved CubeSat Geant4 payloads without rerunning MCDC."
    )
    parser.add_argument("--payload-dir", default=str(DEFAULT_PAYLOAD_DIR))
    parser.add_argument("--pattern", default="*_payload.h5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bridge-build-dir", default=None)
    args = parser.parse_args()

    payload_paths = sorted(pathlib.Path(args.payload_dir).glob(args.pattern))
    if not payload_paths:
        raise FileNotFoundError(
            f"No payload files matching {args.pattern!r} in {args.payload_dir}"
        )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.n_events is not None and args.n_events <= 0:
        raise ValueError("--n-events must be positive")

    geant4_worker, read_summary_hdf5 = _load_mcdc_helpers()
    worker_count = min(args.workers, len(payload_paths))
    print(
        " Geant4 payload replay: "
        f"payloads={len(payload_paths)} workers={worker_count} "
        f"threads_per_worker={args.threads if args.threads is not None else 'payload'}"
    )
    sys.stdout.flush()

    summaries = [None] * len(payload_paths)
    failures = [None] * len(payload_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_one_payload,
                path,
                args,
                geant4_worker,
                read_summary_hdf5,
            ): (idx, path)
            for idx, path in enumerate(payload_paths)
        }
        for future in concurrent.futures.as_completed(futures):
            idx, path = futures[future]
            try:
                summaries[idx] = future.result()
            except Exception as exc:
                failures[idx] = f"{path.name}: {exc}"

    for failure in failures:
        if failure is not None:
            print(f"ERROR: {failure}")
    if any(failure is not None for failure in failures):
        return 1

    total_events = sum(int(summary["events_run"]) for summary in summaries)
    print(f" Geant4 payload replay complete: events_run={total_events}")
    _print_timing_summary(summaries)
    return 0


def _load_mcdc_helpers():
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from mcdc.coupling import geant4_worker
        from mcdc.coupling.geant4_config import read_summary_hdf5
    finally:
        sys.argv = original_argv
    return geant4_worker, read_summary_hdf5


def _run_one_payload(
    payload_path: pathlib.Path,
    args,
    geant4_worker,
    read_summary_hdf5,
) -> dict:
    payload = geant4_worker.read_payload(payload_path)
    name = str(payload["name"])
    run_payload_path = payload_path
    temp_payload_path = None

    if _needs_payload_rewrite(args):
        payload = dict(payload)
        if args.n_events is not None:
            if str(payload["source_mode"]) != "distribution":
                raise RuntimeError("--n-events override only supports distribution mode")
            payload["n_events"] = int(args.n_events)
            payload["source_size"] = int(args.n_events)
        if args.output_dir is not None:
            output_dir = pathlib.Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            payload["geant4_output_path"] = str(
                output_dir / pathlib.Path(str(payload["geant4_output_path"])).name
            )
        if args.bridge_build_dir is not None:
            payload["bridge_build_dir"] = str(pathlib.Path(args.bridge_build_dir))
        if args.threads is not None:
            payload["n_geant4_threads"] = int(args.threads)

        handle = tempfile.NamedTemporaryFile(
            prefix=f"mcdc_g4_replay_{name}_", suffix=".h5", delete=False
        )
        handle.close()
        temp_payload_path = pathlib.Path(handle.name)
        geant4_worker.write_payload(temp_payload_path, payload)
        run_payload_path = temp_payload_path

    output_path = pathlib.Path(str(payload["geant4_output_path"]))
    worker_path = pathlib.Path(geant4_worker.__file__).resolve()
    print(
        f" Geant4 payload '{name}': worker started "
        f"source_size={payload['source_size']} "
        f"threads={payload.get('n_geant4_threads', 1)} output={output_path}"
    )
    sys.stdout.flush()
    worker_wall_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(worker_path), str(run_payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    worker_wall_s = time.perf_counter() - worker_wall_start
    if temp_payload_path is not None:
        temp_payload_path.unlink(missing_ok=True)
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(
            f"worker return code {result.returncode}; stderr tail:\n{stderr_tail}"
        )

    summary = read_summary_hdf5(str(output_path))
    summary["worker_wall_s"] = worker_wall_s
    geant4_worker.write_summary_hdf5(summary, str(output_path))
    print(
        f" Geant4 payload '{name}': worker finished "
        f"events_run={summary['events_run']} "
        f"worker_wall={worker_wall_s:.2f}s "
        f"beam_wall={float(summary.get('geant4_beam_wall_s', 0.0)):.2f}s "
        f"beam_cpu/wall={float(summary.get('geant4_beam_cpu_per_wall', 0.0)):.2f} "
        f"status={summary['status']}"
    )
    sys.stdout.flush()
    return summary


def _needs_payload_rewrite(args) -> bool:
    return (
        args.n_events is not None
        or args.output_dir is not None
        or args.bridge_build_dir is not None
        or args.threads is not None
    )


def _print_timing_summary(summaries: list[dict]) -> None:
    if not summaries:
        return

    worker_wall_sum = sum(float(s.get("worker_wall_s", 0.0)) for s in summaries)
    beam_wall_sum = sum(float(s.get("geant4_beam_wall_s", 0.0)) for s in summaries)
    beam_cpu_sum = sum(float(s.get("geant4_beam_cpu_s", 0.0)) for s in summaries)

    print(" Geant4 timing:")
    print(
        "   region        events   worker_wall    init_wall    load_wall"
        "    beam_wall     beam_cpu  cpu/wall"
    )
    for summary in summaries:
        print(
            f"   {str(summary['name']):<8} "
            f"{int(summary['events_run']):>9} "
            f"{float(summary.get('worker_wall_s', 0.0)):>11.2f}s "
            f"{float(summary.get('geant4_init_wall_s', 0.0)):>10.2f}s "
            f"{float(summary.get('geant4_source_load_wall_s', 0.0)):>10.2f}s "
            f"{float(summary.get('geant4_beam_wall_s', 0.0)):>10.2f}s "
            f"{float(summary.get('geant4_beam_cpu_s', 0.0)):>10.2f}s "
            f"{float(summary.get('geant4_beam_cpu_per_wall', 0.0)):>8.2f}"
        )
    print(
        " Geant4 timing totals: "
        f"sum_worker_wall={worker_wall_sum:.2f}s "
        f"sum_beam_wall={beam_wall_sum:.2f}s "
        f"sum_beam_cpu={beam_cpu_sum:.2f}s"
    )


if __name__ == "__main__":
    sys.exit(main())
