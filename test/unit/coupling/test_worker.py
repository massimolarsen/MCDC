import subprocess
import sys

import h5py
import numpy as np

from mcdc.coupling import geant4_config, geant4_worker
from mcdc.coupling.seu_diagnostics import write_diagnostic_tables


STUB_BRIDGE = """
class SessionConfig:
    def __init__(self):
        self.world_size_mm = []
        self.detector_size_mm = []
        self.detector_material = ""
        self.envelope_material = ""
        self.device_components = []
        self.physics_list = ""
        self.random_seed = 0
        self.n_threads = 1
        self.em_production_cut_mm = 0.0
        self.record_seu_events = False
        self.diagnostic_min_Eion_mev = 0.001

class DeviceComponent:
    pass

class Results:
    loaded_primaries = 3
    last_events_run = 3
    last_total_edep_mev = 1.5
    last_dose_gy = 2.0
    component_names = ["detector"]
    component_edep_mev = [1.5]
    component_mass_kg = [0.25]
    component_dose_gy = [2.0]
    edep_spectrum_edges_mev = [0.0, 1.0]
    edep_spectrum_counts = [3]
    edep_spectrum_edep_mev = [1.5]
    edep_spectrum_underflow = 0
    edep_spectrum_overflow = 0
    geant4_version = "11.4.0"
    physics_list = "QGSP_BIC_HP"
    em_production_cut_mm = 0.0
    proton_production_cut_mm = 0.0
    electronics_cut_materials = []
    electronics_cut_energy_mev = []
    component_niel_mev = [0.1]
    component_ionizing_mev = [1.4]
    component_ionizing_sum_sq_mev2 = [1.1]
    seu_species_names = ["electron_positron", "proton", "neutron", "gamma", "alpha", "ion_recoil", "other"]
    component_species_ionizing_mev = [1.4, 0, 0, 0, 0, 0, 0]
    component_species_ionizing_sum_sq_mev2 = [1.1, 0, 0, 0, 0, 0, 0]
    component_species_positive_events = [2, 0, 0, 0, 0, 0, 0]
    component_primary_ionizing_mev = [0.4]
    component_secondary_ionizing_mev = [1.0]
    component_event_ionizing_edges_mev = [0.001, 0.002]
    component_event_ionizing_count = [0, 3, 0, 0]
    component_event_ionizing_sumw = [0, 2.5, 0, 0]
    component_event_ionizing_sumw2 = [0, 2.1, 0, 0]
    component_primary_species_edep_mev = [0, 0, 1.5, 0, 0, 0, 0]
    component_primary_species_ionizing_mev = [0, 0, 1.4, 0, 0, 0, 0]
    component_primary_species_ionizing_sum_sq_mev2 = [0, 0, 1.1, 0, 0, 0, 0]
    component_primary_species_event_ionizing_count = [0] * 28
    component_primary_species_event_ionizing_sumw = [0] * 28
    component_primary_species_event_ionizing_sumw2 = [0] * 28

class Session:
    def __init__(self, config):
        if config.random_seed != 13579:
            raise RuntimeError("random seed was not passed to bridge")
        if config.envelope_material != "G4_Galactic":
            raise RuntimeError("envelope material was not passed to bridge")
        if config.detector_material != "G4_Si":
            raise RuntimeError("detector material was not passed to bridge")
        if config.n_threads != 2:
            raise RuntimeError("thread count was not passed to bridge")
        if config.device_components[0].name != "detector":
            raise RuntimeError("device components were not passed to bridge")
        if config.device_components[0].parent != "":
            raise RuntimeError("component parent was not passed to bridge")
        self.config = config
    def initialize(self):
        pass
    def load_primaries(self, bank):
        pass
    def load_source_distribution(self, *args):
        # record (particle_id, n_events) for each loaded species
        import pathlib
        log = pathlib.Path(__file__).with_name("loaded_sources.txt")
        with open(log, "a") as file:
            file.write(f"{args[-1]} {args[-2]}\\n")
    def beam_on(self):
        pass
    def get_results(self):
        return Results()
    def close(self):
        pass
"""


def test_payload_round_trip_distribution(tmp_path):
    payload = {
        "name": "cpu",
        "source_mode": "distribution",
        "bridge_build_dir": "bridge",
        "geant4_output_path": str(tmp_path / "out.h5"),
        "world_size_mm": (100.0, 100.0, 100.0),
        "detector_size_mm": (10.0, 20.0, 30.0),
        "detector_material": "G4_Si",
        "envelope_material": "G4_Galactic",
        "component_names": np.asarray(["die"]),
        "component_materials": np.asarray(["G4_Si"]),
        "component_parents": np.asarray([""]),
        "component_centers_mm": np.asarray([[0.0, 0.0, 0.0]]),
        "component_sizes_mm": np.asarray([[10.0, 20.0, 30.0]]),
        "component_score": np.asarray([True]),
        "physics_list": "QGSP_BIC",
        "random_seed": 777,
        "n_geant4_threads": 3,
        "source_size": 4,
        "source_tally_name": "cpu_src",
        "source_total_weight": 2.5,
        "box_bounds_mm": np.asarray([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]),
        "mu_edges": np.asarray([-1.0, 1.0]),
        "azi_edges": np.asarray([-np.pi, np.pi]),
        "energy_edges_mev": np.asarray([0.0, 14.0]),
        "weights": np.ones(6),
        "Nu": 1,
        "Nv": 1,
        "n_events": 4,
    }
    path = tmp_path / "payload.h5"

    geant4_worker.write_payload(path, payload)
    result = geant4_worker.read_payload(path)

    assert result["name"] == "cpu"
    assert result["source_mode"] == "distribution"
    assert result["detector_material"] == "G4_Si"
    assert int(result["random_seed"]) == 777
    assert int(result["n_geant4_threads"]) == 3
    np.testing.assert_allclose(result["box_bounds_mm"], payload["box_bounds_mm"])
    np.testing.assert_allclose(result["weights"], payload["weights"])
    np.testing.assert_array_equal(result["component_names"], ["die"])
    np.testing.assert_array_equal(result["component_materials"], ["G4_Si"])
    np.testing.assert_array_equal(result["component_parents"], [""])
    np.testing.assert_allclose(
        result["component_centers_mm"], payload["component_centers_mm"]
    )
    np.testing.assert_array_equal(result["component_score"], [True])
    assert int(result["n_events"]) == 4


def test_worker_distribution_subprocess_with_stub_bridge(tmp_path):
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "geant4_bridge.py").write_text(STUB_BRIDGE, encoding="utf-8")

    payload_path = tmp_path / "payload.h5"
    output_path = tmp_path / "out.h5"
    geant4_worker.write_payload(
        payload_path,
        {
            "name": "distribution",
            "source_mode": "distribution",
            "bridge_build_dir": str(bridge_dir),
            "geant4_output_path": str(output_path),
            "world_size_mm": (100.0, 100.0, 100.0),
            "detector_size_mm": (10.0, 10.0, 10.0),
            "detector_material": "G4_Si",
            "envelope_material": "G4_Galactic",
            "component_names": np.asarray(["detector"]),
            "component_materials": np.asarray(["G4_Si"]),
            "component_parents": np.asarray([""]),
            "component_centers_mm": np.asarray([[0.0, 0.0, 0.0]]),
            "component_sizes_mm": np.asarray([[10.0, 10.0, 10.0]]),
            "component_score": np.asarray([True]),
            "physics_list": "QGSP_BIC",
            "random_seed": 13579,
            "n_geant4_threads": 2,
            "source_size": 3,
            "source_tally_name": "test_source",
            "source_total_weight": 2.5,
            "box_bounds_mm": np.asarray([[-1, 1], [-1, 1], [-1, 1]]),
            "mu_edges": np.asarray([0.0, 1.0]),
            "azi_edges": np.asarray([-np.pi, np.pi]),
            "energy_edges_mev": np.asarray([1.0, 2.0]),
            "weights": np.asarray([0, 0, 0, 0, 2.5, 0]),
            "Nu": 1,
            "Nv": 1,
            "n_events": 3,
        },
    )

    result = subprocess.run(
        [sys.executable, str(geant4_worker.__file__), str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    with h5py.File(output_path, "r") as file:
        assert file["source_mode"][()].decode("utf-8") == "distribution"
        assert int(file["random_seed"][()]) == 13579
        assert int(file["n_geant4_threads"][()]) == 2
        assert int(file["events_run"][()]) == 3
        assert float(file["total_edep_mev"][()]) == 1.5
        assert file["component_names"][0].decode("utf-8") == "detector"
        assert float(file["component_edep_mev"][0]) == 1.5
        assert float(file["component_ionizing_mev"][0]) == 1.4
        assert float(file["component_ionizing_sum_sq_mev2"][0]) == 1.1
        assert file["component_species_ionizing_mev"].shape == (1, 7)
        assert file["component_species_ionizing_sum_sq_mev2"].shape == (1, 7)
        assert int(file["component_species_positive_events"][0, 0]) == 2
        assert file["component_event_ionizing_count"].shape == (1, 4)
        assert float(file["component_primary_species_ionizing_mev"][0, 2]) == 1.4
        assert file["component_primary_species_event_ionizing_count"].shape == (1, 7, 4)
        assert float(file["em_production_cut_mm"][()]) == 0.0
        assert not bool(file["record_seu_events"][()])

    stream_dir = tmp_path / "streams"
    stream_dir.mkdir()
    (stream_dir / "worker_0.tsv").write_text(
        "E\t0\t2\t0\t2112\t0.5\t1.0\t0.1\t0.9\t0.2\t0.7\t0.7\t0.2\t0\t0\t0\t0\t0\n"
        "M\t0\t2\t3\t0.4\t0\t0\t1\t0.2\n"
        "I\t0\t2\t0\t4\t1\t2212\t3\t0\t0\t0\t0\t0\t1\thadElastic\t0\t0\t-1\t4\tdie\n"
        "N\t0\t2\t4\t1\t2212\t4\t0\t0\t-1\tdie\thadElastic\n"
    )
    write_diagnostic_tables(output_path, stream_dir)
    with h5py.File(output_path) as file:
        details = file["seu_diagnostics"]
        assert details["event_sv"][0]["event_id"] == 2
        assert details["event_sv"][0]["primary_pdg"] == 2112
        assert details["sv_entry"][0]["track_id"] == 4
        assert details["nuclear_birth"][0]["track_id"] == 4
        assert details["em_secondary_summary"][0]["electron_count"] == 3
    assert geant4_config.read_summary_hdf5(str(output_path))["events_run"] == 3


def _species_payload(tmp_path, bridge_dir):
    def block(pdg, tally_name, n_events, total_weight):
        return {
            "pdg": pdg,
            "tally_name": tally_name,
            "mu_edges": np.asarray([0.0, 1.0]),
            "azi_edges": np.asarray([-np.pi, np.pi]),
            "energy_edges_mev": np.asarray([1.0, 2.0]),
            "weights": np.asarray([0, 0, 0, 0, total_weight, 0]),
            "Nu": 1,
            "Nv": 1,
            "n_events": n_events,
            "total_weight": total_weight,
        }

    return {
        "name": "species",
        "source_mode": "distribution",
        "bridge_build_dir": str(bridge_dir),
        "geant4_output_path": str(tmp_path / "out.h5"),
        "world_size_mm": (100.0, 100.0, 100.0),
        "detector_size_mm": (10.0, 10.0, 10.0),
        "detector_material": "G4_Si",
        "envelope_material": "G4_Galactic",
        "component_names": np.asarray(["detector"]),
        "component_materials": np.asarray(["G4_Si"]),
        "component_parents": np.asarray([""]),
        "component_centers_mm": np.asarray([[0.0, 0.0, 0.0]]),
        "component_sizes_mm": np.asarray([[10.0, 10.0, 10.0]]),
        "component_score": np.asarray([True]),
        "physics_list": "QGSP_BIC",
        "random_seed": 13579,
        "n_geant4_threads": 2,
        "source_size": 3,
        "source_tally_name": "p_src,n_src",
        "source_total_weight": 3.5,
        "box_bounds_mm": np.asarray([[-1, 1], [-1, 1], [-1, 1]]),
        "sources": [block(2212, "p_src", 1, 2.5), block(2112, "n_src", 2, 1.0)],
    }


def test_payload_round_trip_species_sources(tmp_path):
    payload = _species_payload(tmp_path, tmp_path / "bridge")
    path = tmp_path / "payload.h5"

    geant4_worker.write_payload(path, payload)
    result = geant4_worker.read_payload(path)

    assert [int(block["pdg"]) for block in result["sources"]] == [2212, 2112]
    assert [int(block["n_events"]) for block in result["sources"]] == [1, 2]
    assert result["sources"][0]["tally_name"] == "p_src"
    np.testing.assert_allclose(
        result["sources"][1]["weights"], payload["sources"][1]["weights"]
    )
    np.testing.assert_allclose(result["box_bounds_mm"], payload["box_bounds_mm"])


def test_worker_loads_each_species_with_stub_bridge(tmp_path):
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "geant4_bridge.py").write_text(STUB_BRIDGE, encoding="utf-8")
    payload_path = tmp_path / "payload.h5"
    geant4_worker.write_payload(payload_path, _species_payload(tmp_path, bridge_dir))

    result = subprocess.run(
        [sys.executable, str(geant4_worker.__file__), str(payload_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    loaded = (bridge_dir / "loaded_sources.txt").read_text().split()
    assert loaded == ["2212", "1", "2112", "2"]
    summary = geant4_config.read_summary_hdf5(str(tmp_path / "out.h5"))
    np.testing.assert_array_equal(summary["source_species_pdg"], [2212, 2112])
    np.testing.assert_array_equal(summary["source_species_n_events"], [1, 2])
    np.testing.assert_allclose(summary["source_species_weight"], [2.5, 1.0])
    assert list(summary["source_species_tally_name"]) == ["p_src", "n_src"]
