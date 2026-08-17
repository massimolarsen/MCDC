import subprocess
import sys

import h5py
import numpy as np

from mcdc.coupling import geant4_worker


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


def test_worker_real_subprocess_with_stub_bridge(tmp_path):
    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "geant4_bridge.py").write_text(
        """
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
    status = "ok"

class Session:
    def __init__(self, config):
        if config.random_seed != 13579:
            raise RuntimeError("random seed was not passed to bridge")
        if config.envelope_material != "G4_Galactic":
            raise RuntimeError("envelope material was not passed to bridge")
        if config.n_threads != 2:
            raise RuntimeError("thread count was not passed to bridge")
        if config.device_components[0]["name"] != "detector":
            raise RuntimeError("device components were not passed to bridge")
        if config.device_components[0]["parent"] != "":
            raise RuntimeError("component parent was not passed to bridge")
        self.config = config
    def initialize(self):
        pass
    def load_primaries(self, bank):
        pass
    def load_source_distribution(self, *args):
        pass
    def beam_on(self):
        pass
    def get_results(self):
        return Results()
""",
        encoding="utf-8",
    )

    payload_path = tmp_path / "payload.h5"
    output_path = tmp_path / "out.h5"
    geant4_worker.write_payload(
        payload_path,
        {
            "name": "bank",
            "source_mode": "bank",
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
            "bank": np.ones((3, 10)),
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
        assert file["source_mode"][()].decode("utf-8") == "bank"
        assert int(file["random_seed"][()]) == 13579
        assert int(file["n_geant4_threads"][()]) == 2
        assert int(file["events_run"][()]) == 3
        assert float(file["total_edep_mev"][()]) == 1.5
        assert file["component_names"][0].decode("utf-8") == "detector"
        assert float(file["component_edep_mev"][0]) == 1.5
