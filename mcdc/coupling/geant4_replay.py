"""Replay one Geant4 event of a finished coupled run and save all its tracks.

Run on the machine (and Geant4 build) that made the run: the event is rebuilt
from the run's random seed, or from its saved RNG state when the run has one,
and checked against the run's SEU diagnostic rows before the file is written.
The output holds the geometry and every track and step point, for
``mcdc.coupling.geant4_view``.

    python -m mcdc.coupling.geant4_replay --run-dir RUN --region comms \\
        --event 26612987 --bridge-build BUILD [--out FILE]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
from typing import Any

import h5py
import numpy as np

try:
    from .geant4_worker import (
        load_bridge,
        load_source,
        make_session_config,
        read_payload,
        source_blocks,
    )
    from .seu_diagnostics import write_diagnostic_tables
except ImportError:
    from geant4_worker import (
        load_bridge,
        load_source,
        make_session_config,
        read_payload,
        source_blocks,
    )
    from seu_diagnostics import write_diagnostic_tables


GEOMETRY_FIELDS = (
    "world_size_mm",
    "detector_size_mm",
    "envelope_material",
    "em_production_cut_mm",
    "component_names",
    "component_materials",
    "component_parents",
    "component_centers_mm",
    "component_sizes_mm",
    "component_score",
)
TRACK_FIELDS = (
    "track_id",
    "parent_id",
    "pdg",
    "particle",
    "charge",
    "creator_process",
    "track_point_start",
)
POINT_FIELDS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "t_ns",
    "ke_mev",
    "edep_mev",
    "process",
    "volume",
)
STRING_FIELDS = {
    "envelope_material",
    "component_names",
    "component_materials",
    "component_parents",
    "particle",
    "creator_process",
    "process",
    "volume",
}


def original_rows(path: pathlib.Path, event_id: int) -> dict[str, np.ndarray]:
    """The run's diagnostic rows of one event (run 0: one BeamOn per region)."""
    rows = {}
    with h5py.File(path, "r") as file:
        if "seu_diagnostics" not in file:
            raise RuntimeError(f"{path} has no SEU diagnostics to check a replay against.")
        for name, dataset in file["seu_diagnostics"].items():
            table = dataset[()]
            rows[name] = table[(table["run_id"] == 0) & (table["event_id"] == event_id)]
    return rows


def _decoded(values) -> list:
    return [v.decode() if isinstance(v, bytes) else v for v in values]


def compare(original: dict[str, np.ndarray], replayed: dict[str, np.ndarray]) -> list[str]:
    """Differences between original and replayed rows; empty when they match."""
    problems = []
    for name in ("event_sv", "em_secondary_summary", "nuclear_birth", "sv_entry"):
        a = original.get(name)
        b = replayed.get(name)
        if a is None or len(a) == 0:
            continue
        if b is None or len(a) != len(b):
            problems.append(
                f"{name}: {len(a)} original rows, {0 if b is None else len(b)} replayed"
            )
            continue
        # one worker writes an event's rows in a fixed order; sort to be safe
        key = "sv_id" if name == "event_sv" else "track_id"
        if key in a.dtype.names:
            a = a[np.argsort(a[key], kind="stable")]
            b = b[np.argsort(b[key], kind="stable")]
        for field in a.dtype.names:
            if a.dtype[field].kind in "iuf":
                if not np.array_equal(a[field], b[field]):
                    diff = np.max(np.abs(a[field].astype(float) - b[field].astype(float)))
                    problems.append(f"{name}.{field}: max difference {diff:.6g}")
            elif _decoded(a[field]) != _decoded(b[field]):
                problems.append(f"{name}.{field}: values differ")
    return problems


def replay_event(
    run_dir: pathlib.Path,
    region: str,
    event_id: int,
    bridge_build: str,
    mode: str = "auto",
    original_path: pathlib.Path | None = None,
) -> tuple[dict[str, Any], Any, pathlib.Path]:
    payload = read_payload(run_dir / "geant4_payloads" / f"{region}_payload.h5")
    if original_path is None:
        name = pathlib.Path(str(payload["geant4_output_path"])).name
        original_path = run_dir / "geant4_h5" / name
    original = original_rows(original_path, event_id)

    states = original.get("event_rng_state")
    if mode == "auto":
        mode = "state" if states is not None and len(states) else "seeds"
    if mode == "state" and (states is None or len(states) == 0):
        raise RuntimeError(f"The run saved no RNG state for event {event_id}.")
    if str(payload["source_mode"]) == "bank":
        n_events = len(payload["bank"])
    else:
        n_events = sum(int(block["n_events"]) for block in source_blocks(payload))
    if not 0 <= event_id < n_events:
        raise RuntimeError(f"Event {event_id} is outside the run's {n_events} events.")

    bridge = load_bridge(bridge_build)
    diagnostic_dir = pathlib.Path(tempfile.mkdtemp(prefix="mcdc_g4_replay_"))
    config = make_session_config(bridge, payload, diagnostic_dir)
    # write the event's rows whatever it deposits, and keep every track
    config.diagnostic_min_Eion_mev = 0.0
    config.rng_state_min_Eion_mev = 0.0
    config.n_threads = 1
    config.replay_mode = mode
    config.event_id_offset = event_id
    config.record_tracks = True
    if mode == "state":
        config.replay_state = [int(x) for x in _decoded(states["state"])[0].split()]

    session = bridge.Session(config)
    session.initialize()
    load_source(session, payload)
    session.beam_on(1)
    results = session.get_results()
    session.close()

    info = {
        "run_dir": str(run_dir),
        "region": region,
        "run_id": 0,
        "event_id": event_id,
        "replay_mode": mode,
        "random_seed": int(payload["random_seed"]),
        "original_n_geant4_threads": int(payload["n_geant4_threads"]),
        "geant4_version": str(results.geant4_version),
        "physics_list": str(results.physics_list),
        "original_path": str(original_path),
    }
    return {"info": info, "payload": payload, "original": original}, results, diagnostic_dir


def write_replay(
    path: pathlib.Path,
    replay: dict[str, Any],
    results,
    diagnostic_dir: pathlib.Path,
) -> dict[str, np.ndarray]:
    """Write geometry, tracks, and replayed rows; return the replayed rows."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    payload = replay["payload"]
    tracks = results.tracks
    with h5py.File(path, "w") as file:
        for key, value in replay["info"].items():
            file.attrs[key] = value
        for group_name, fields, source in (
            ("geometry", GEOMETRY_FIELDS, None),
            ("tracks", TRACK_FIELDS, tracks),
            ("points", POINT_FIELDS, tracks),
        ):
            group = file.create_group(group_name)
            for field in fields:
                value = payload[field] if source is None else getattr(source, field)
                if field in STRING_FIELDS:
                    group.create_dataset(
                        field, data=np.asarray(value, dtype=object), dtype=string_dtype
                    )
                else:
                    group.create_dataset(field, data=np.asarray(value))
        original = file.create_group("original")
        for name, rows in replay["original"].items():
            original.create_dataset(name, data=rows)
    write_diagnostic_tables(path, diagnostic_dir)
    for table in diagnostic_dir.glob("worker_*.tsv"):
        table.unlink()
    diagnostic_dir.rmdir()
    with h5py.File(path, "r") as file:
        return {name: data[()] for name, data in file["seu_diagnostics"].items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--event", required=True, type=int)
    parser.add_argument("--bridge-build", required=True, help="bridge build with replay support")
    parser.add_argument("--out", type=pathlib.Path, help="default: <region>_<event>_tracks.h5")
    parser.add_argument("--mode", choices=("auto", "seeds", "state"), default="auto",
                        help="auto uses a saved RNG state when the run has one")
    parser.add_argument("--original", type=pathlib.Path,
                        help="region Geant4 output (default: from the payload name)")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="keep the file when the replay differs from the run")
    args = parser.parse_args(argv)

    out = args.out or pathlib.Path(f"{args.region}_{args.event}_tracks.h5")
    replay, results, diagnostic_dir = replay_event(
        args.run_dir, args.region, args.event, args.bridge_build, args.mode, args.original
    )
    partial = out.with_name(out.name + ".partial")
    replayed = write_replay(partial, replay, results, diagnostic_dir)

    original = replay["original"]
    names = [str(n) for n, s in zip(replay["payload"]["component_names"],
                                    replay["payload"]["component_score"]) if s]
    info = replay["info"]
    print(f"Replayed {info['region']} event {info['event_id']} by {info['replay_mode']} "
          f"(seed {info['random_seed']}, Geant4 {info['geant4_version']}): "
          f"{len(results.tracks.track_id)} tracks, {len(results.tracks.x_mm)} points")
    for label, rows in (("original", original.get("event_sv")), ("replayed", replayed.get("event_sv"))):
        if rows is None or len(rows) == 0:
            print(f"  {label}: no SV rows")
            continue
        hit = rows[rows["Edep_ionizing_mev"] > 0]
        print(f"  {label}: " + (", ".join(
            f"{names[int(r['sv_id'])]} {r['Edep_ionizing_mev']:.6g} MeV" for r in hit
        ) or "no SV deposit"))

    if original.get("event_sv") is None or len(original["event_sv"]) == 0:
        problems = ["the run recorded no diagnostic rows for this event, so it cannot be checked"]
    else:
        problems = compare(original, replayed)
    with h5py.File(partial, "a") as file:
        file.attrs["verified"] = not problems
    if problems:
        print("Replay does NOT match the run:\n  " + "\n  ".join(problems))
        if not args.allow_mismatch:
            partial.unlink()
            return 1
    else:
        print("Replay matches the run's recorded rows exactly.")
    os.replace(partial, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
