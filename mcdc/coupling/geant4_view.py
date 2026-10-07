"""Open Geant4's Qt viewer on a replayed event from ``geant4_replay``.

    python -m mcdc.coupling.geant4_view comms_26612987_tracks.h5 \\
        --bridge-build BUILD [--sv NAME] [--min-ke-kev 10]

Needs a bridge built against a Geant4 with Qt (``geant4_bridge.has_viewer``).
Tracks are colored by particle; the SV with the largest replayed deposit (or
--sv) is drawn solid red and the view is centered on it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np

try:
    from .geant4_replay import POINT_FIELDS, TRACK_FIELDS
    from .geant4_worker import _bridge_components, load_bridge
except ImportError:
    from geant4_replay import POINT_FIELDS, TRACK_FIELDS
    from geant4_worker import _bridge_components, load_bridge


PARTICLE_COLORS = {
    "e-": "red",
    "e+": "blue",
    "gamma": "green",
    "neutron": "yellow",
    "proton": "cyan",
    "alpha": "magenta",
}


def _read(group) -> dict:
    data = {}
    for key, dataset in group.items():
        value = dataset[()]
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
            value = [v.decode() if isinstance(v, bytes) else v for v in value]
        data[key] = value
    return data


def vis_commands(geometry, tracks, sv_name, sv_index, min_ke_kev) -> list[str]:
    names = list(geometry["component_names"])
    scored = [n for n, s in zip(names, geometry["component_score"]) if s]
    center = np.asarray(geometry["component_centers_mm"])[sv_index]
    size = np.asarray(geometry["component_sizes_mm"])[sv_index]
    zoom = max(float(np.max(geometry["world_size_mm"])) / float(np.max(size)), 1.0)

    commands = [
        "/vis/open",
        "/vis/viewer/set/autoRefresh false",
        "/vis/verbose errors",
        "/vis/drawVolume",
        "/vis/geometry/set/visibility World 0 false",
        "/vis/geometry/set/visibility DetectorEnvelope 0 false",
        "/vis/viewer/set/style wireframe",
        "/vis/geometry/set/colour all 0 0.6 0.6 0.6 0.25",
    ]
    for name in scored:
        commands.append(f"/vis/geometry/set/colour {name} 0 1 0.6 0 0.6")
    commands += [
        f"/vis/geometry/set/colour {sv_name} 0 1 0 0 0.35",
        f"/vis/geometry/set/forceSolid {sv_name} 0 true",
        "/vis/scene/add/trajectories",
        # the viewer's own geantino is not drawn; the recorded tracks are
        "/tracking/storeTrajectory 0",
        "/vis/modeling/trajectories/create/drawByParticleID",
        "/vis/modeling/trajectories/drawByParticleID-0/setDefaultRGBA 1 0.5 0 1",
        "/vis/modeling/trajectories/drawByParticleID-0/default/setDrawStepPts true",
        "/vis/modeling/trajectories/drawByParticleID-0/default/setStepPtsSize 2",
    ]
    for particle in sorted(set(tracks["particle"])):
        if particle in PARTICLE_COLORS:
            commands.append(
                f"/vis/modeling/trajectories/drawByParticleID-0/set {particle} "
                f"{PARTICLE_COLORS[particle]}"
            )
    if min_ke_kev > 0.0:
        commands += [
            "/vis/filtering/trajectories/create/attributeFilter keCut",
            "/vis/filtering/trajectories/keCut/setAttribute IKE",
            f"/vis/filtering/trajectories/keCut/addInterval {min_ke_kev} keV 1000 TeV",
        ]
    commands += [
        "/vis/scene/endOfEventAction accumulate",
        "/vis/viewer/set/viewpointThetaPhi 70 20 deg",
        f"/vis/viewer/set/targetPoint {center[0]} {center[1]} {center[2]} mm",
        f"/vis/viewer/zoomTo {zoom:.6g}",
        "/vis/viewer/set/autoRefresh true",
        "/run/beamOn 1",
    ]
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay_file", type=pathlib.Path)
    parser.add_argument("--bridge-build", required=True, help="bridge built with Qt vis")
    parser.add_argument("--sv", help="SV to highlight (default: largest replayed deposit)")
    parser.add_argument("--min-ke-kev", type=float, default=0.0,
                        help="hide tracks born below this kinetic energy")
    args = parser.parse_args(argv)

    with h5py.File(args.replay_file, "r") as file:
        info = dict(file.attrs)
        geometry = _read(file["geometry"])
        tracks = _read(file["tracks"])
        points = _read(file["points"])
        event_sv = file["seu_diagnostics/event_sv"][()]

    names = list(geometry["component_names"])
    scored = [n for n, s in zip(names, geometry["component_score"]) if s]
    sv_name = args.sv or scored[int(event_sv["sv_id"][np.argmax(event_sv["Edep_ionizing_mev"])])]
    if sv_name not in names:
        raise SystemExit(f"Unknown SV '{sv_name}'; components are {names}")

    bridge = load_bridge(args.bridge_build)
    if not getattr(bridge, "has_viewer", False):
        raise SystemExit("This bridge build has no viewer (Geant4 without Qt).")

    config = bridge.SessionConfig()
    config.world_size_mm = [float(x) for x in geometry["world_size_mm"]]
    config.detector_size_mm = [float(x) for x in geometry["detector_size_mm"]]
    config.envelope_material = str(geometry["envelope_material"])
    config.em_production_cut_mm = float(geometry["em_production_cut_mm"])
    config.device_components = _bridge_components(bridge, geometry)

    recorded = bridge.RecordedTracks()
    for field in TRACK_FIELDS + POINT_FIELDS:
        source = tracks if field in TRACK_FIELDS else points
        value = source[field]
        setattr(recorded, field, value if isinstance(value, list) else value.tolist())

    print(f"{info['region']} event {info['event_id']} ({info['replay_mode']} replay, "
          f"verified={bool(info.get('verified', False))}): {len(tracks['track_id'])} tracks; "
          f"highlighting {sv_name}")
    print("Pick a track or point in the viewer for its attributes; "
          "/vis/filtering/trajectories/list shows active filters.")
    bridge.view_event(
        config, recorded, vis_commands(geometry, tracks, sv_name, names.index(sv_name),
                                       args.min_ke_kev)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
