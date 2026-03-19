import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np


def _load_geant4_export_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "mcdc" / "geant4_export.py"
    )
    spec = importlib.util.spec_from_file_location("mcdc_geant4_export", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


geant4_export = _load_geant4_export_module()
classify_candidates = geant4_export.classify_candidates
inspect_and_export = geant4_export.inspect_and_export
inspect_hdf5_tallies = geant4_export.inspect_hdf5_tallies


def _write_dataset(group, path, data):
    group.create_dataset(path, data=np.asarray(data))


def _make_fixture(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        tallies = h5.create_group("tallies")

        surface_energy = tallies.create_group("surface_energy_candidate")
        _write_dataset(surface_energy, "grid/mu", [-1.0, 1.0])
        _write_dataset(surface_energy, "grid/azi", [-np.pi, np.pi])
        _write_dataset(surface_energy, "grid/energy", [0.0, 1.0, 2.0, 3.0])
        _write_dataset(surface_energy, "grid/time", [0.0, 1.0])
        _write_dataset(surface_energy, "net-current/mean", [1.0, 2.0, 3.0])
        _write_dataset(surface_energy, "net-current/sdev", [0.1, 0.2, 0.3])

        surface_angle = tallies.create_group("surface_angle_candidate")
        _write_dataset(surface_angle, "grid/mu", [-1.0, 0.0, 1.0])
        _write_dataset(surface_angle, "grid/azi", [-np.pi, np.pi])
        _write_dataset(surface_angle, "grid/energy", [0.0, 10.0])
        _write_dataset(surface_angle, "grid/time", [0.0, 1.0])
        _write_dataset(surface_angle, "flux/mean", [4.0, 5.0])
        _write_dataset(surface_angle, "flux/sdev", [0.4, 0.5])

        mesh = tallies.create_group("mesh_candidate")
        _write_dataset(mesh, "grid/mu", [-1.0, 1.0])
        _write_dataset(mesh, "grid/azi", [-np.pi, np.pi])
        _write_dataset(mesh, "grid/energy", [0.0, 10.0])
        _write_dataset(mesh, "grid/time", [0.0, 1.0, 2.0])
        _write_dataset(mesh, "grid/x", [0.0, 1.0, 2.0, 3.0])
        _write_dataset(mesh, "grid/y", [0.0, 5.0])
        _write_dataset(mesh, "grid/z", [0.0, 1.0, 2.0])
        _write_dataset(
            mesh,
            "flux/mean",
            np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2) + 1.0,
        )
        _write_dataset(mesh, "flux/sdev", np.full((2, 3, 2), 0.25))

        global_tally = tallies.create_group("global_candidate")
        _write_dataset(global_tally, "grid/mu", [-1.0, 1.0])
        _write_dataset(global_tally, "grid/azi", [-np.pi, np.pi])
        _write_dataset(global_tally, "grid/energy", [0.0, 5.0, 10.0])
        _write_dataset(global_tally, "grid/time", [0.0, 1.0, 2.0])
        _write_dataset(global_tally, "density/mean", [[1.0, 10.0], [2.0, 20.0]])
        _write_dataset(global_tally, "density/sdev", np.full((2, 2), 0.1))

        unsupported = tallies.create_group("unsupported_candidate")
        _write_dataset(unsupported, "fission/mean", [1.0])


def test_inspect_hdf5_tallies_discovers_schema(tmp_path):
    hdf_path = tmp_path / "output.h5"
    _make_fixture(hdf_path)

    tallies = inspect_hdf5_tallies(hdf_path)
    names = {tally.name for tally in tallies}

    assert names == {
        "surface_energy_candidate",
        "surface_angle_candidate",
        "mesh_candidate",
        "global_candidate",
        "unsupported_candidate",
    }

    surface = next(tally for tally in tallies if tally.name == "surface_energy_candidate")
    assert surface.tally_class == "surface-like"
    assert set(surface.grids.keys()) == {"mu", "azi", "energy", "time"}
    assert set(surface.scores.keys()) == {"net-current"}

    mesh = next(tally for tally in tallies if tally.name == "mesh_candidate")
    assert mesh.tally_class == "mesh"
    assert set(mesh.grids.keys()) == {"mu", "azi", "energy", "time", "x", "y", "z"}


def test_classify_candidates_ranks_and_marks_unsupported(tmp_path):
    hdf_path = tmp_path / "output.h5"
    _make_fixture(hdf_path)

    tallies = inspect_hdf5_tallies(hdf_path)
    candidates, unsupported = classify_candidates(tallies)

    candidate_ids = {candidate.id for candidate in candidates}
    assert "surface_energy_candidate__net-current__energy" in candidate_ids
    assert "surface_angle_candidate__flux__angle" in candidate_ids
    assert "global_candidate__density__energy" in candidate_ids

    assert candidates[0].id == "surface_energy_candidate__net-current__energy"
    assert unsupported == [
        {
            "tally_name": "mesh_candidate",
            "tally_class": "mesh",
            "reasons": [
                "Tally dimensions are present, but no supported source interpretation was considered safe."
            ],
        },
        {
            "tally_name": "unsupported_candidate",
            "tally_class": "unknown",
            "reasons": [
                "No exportable source dimensions found: expected energy or mu grids."
            ],
        }
    ]


def test_inspect_and_export_writes_manifest_and_candidate_outputs(tmp_path):
    hdf_path = tmp_path / "output.h5"
    _make_fixture(hdf_path)
    outdir = tmp_path / "exports"

    manifest = inspect_and_export(hdf_path=hdf_path, outdir=outdir)

    manifest_path = outdir / "manifest.json"
    assert manifest_path.exists()
    assert manifest["exports"]

    energy_dir = outdir / "surface_energy_candidate__net-current__energy"
    assert (energy_dir / "spectrum.mac").exists()
    assert (energy_dir / "energy_histogram.npz").exists()
    assert (energy_dir / "source_metadata.json").exists()
    assert "/gps/hist/type energy" in (energy_dir / "spectrum.mac").read_text()

    angle_dir = outdir / "surface_angle_candidate__flux__angle"
    assert (angle_dir / "angle_spectrum.mac").exists()
    assert (angle_dir / "angle_histogram.npz").exists()
    assert "/gps/hist/type theta" in (angle_dir / "angle_spectrum.mac").read_text()

    global_energy_dir = outdir / "global_candidate__density__energy"
    assert (global_energy_dir / "spectrum.mac").exists()
    assert (global_energy_dir / "energy_histogram.npz").exists()
