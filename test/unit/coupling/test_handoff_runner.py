import subprocess

import numpy as np
import pytest

from mcdc.coupling import geant4_config, geant4_handoff, geant4_worker
from mcdc.coupling.geant4_config import Geant4HandoffConfig

from ._helpers import distribution_config, distribution_simulation_and_data


def _write_fake_distribution_output(payload, total_edep_mev=0.0, dose_gy=0.0):
    geant4_worker.write_summary_hdf5(
        {
            "name": payload["name"],
            "source_mode": "distribution",
            "source_size": int(payload["source_size"]),
            "loaded_primaries": int(payload["source_size"]),
            "events_run": int(payload["source_size"]),
            "total_edep_mev": total_edep_mev,
            "dose_gy": dose_gy,
            "edep_spectrum_edges_mev": np.asarray([0.0, 1.0, 2.0]),
            "edep_spectrum_counts": np.asarray([10, 2]),
            "edep_spectrum_edep_mev": np.asarray([0.25, 1.0]),
            "edep_spectrum_underflow": 0,
            "edep_spectrum_overflow": 0,
            "status": "ok",
            "source_tally_name": payload["source_tally_name"],
            "source_total_weight": float(payload["source_total_weight"]),
            "random_seed": int(payload["random_seed"]),
        },
        str(payload["geant4_output_path"]),
    )


def test_run_handoff_empty_bank_skips_worker(monkeypatch):
    simulation, data, _ = distribution_simulation_and_data()
    geant4_config.configure(**Geant4HandoffConfig(source_mode="bank").__dict__)

    def fail_run(*args, **kwargs):
        raise AssertionError("empty bank should not spawn a worker")

    monkeypatch.setattr(subprocess, "run", fail_run)
    summary = geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert summary["status"] == "ok"
    assert summary["source_size"] == 0
    assert summary["regions"][0]["status"] == "skipped_empty_handoff"


def test_run_handoff_distribution_runs_worker_and_reads_hdf5(monkeypatch, tmp_path):
    simulation, data, weights = distribution_simulation_and_data()
    output_path = tmp_path / "g4.h5"
    geant4_config.configure(
        **distribution_config(geant4_output_path=str(output_path)).__dict__
    )

    def fake_run(cmd, capture_output, text, check):
        payload = geant4_worker.read_payload(cmd[-1])
        np.testing.assert_array_equal(payload["component_names"], ["detector"])
        np.testing.assert_array_equal(payload["component_materials"], ["G4_Si"])
        np.testing.assert_allclose(payload["component_sizes_mm"], [[10.0, 10.0, 10.0]])
        np.testing.assert_array_equal(payload["component_score"], [True])
        assert payload["envelope_material"] == "G4_Galactic"
        _write_fake_distribution_output(payload, total_edep_mev=1.25, dose_gy=2.5e-9)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summary = geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert summary["source_size"] == 12
    assert summary["n_regions"] == 1
    assert summary["events_run"] == 12
    region = summary["regions"][0]
    assert region["source_mode"] == "distribution"
    assert "handoff_bank_size" not in region
    assert region["random_seed"] > 0
    assert region["total_edep_mev"] == 1.25
    assert region["dose_gy"] == 2.5e-9
    np.testing.assert_allclose(region["edep_spectrum_edges_mev"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(region["edep_spectrum_counts"], [10, 2])
    np.testing.assert_allclose(region["edep_spectrum_edep_mev"], [0.25, 1.0])
    assert output_path.exists()
    assert weights.shape == (2, 2, 2, 6, 2, 3)


def test_run_handoff_zero_distribution_skips_worker(monkeypatch, tmp_path):
    simulation, data, _ = distribution_simulation_and_data(
        weights=np.zeros((2, 2, 2, 6, 2, 3))
    )
    output_path = tmp_path / "skipped.h5"
    geant4_config.configure(
        **distribution_config(geant4_output_path=str(output_path)).__dict__
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("zero-current distribution should not spawn a worker")

    monkeypatch.setattr(subprocess, "run", fail_run)

    summary = geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert summary["status"] == "ok"
    assert summary["source_size"] == 0
    region = summary["regions"][0]
    assert region["name"] == "source_region"
    assert region["source_mode"] == "distribution"
    assert region["source_tally_name"] == "source_tally"
    assert region["source_total_weight"] == 0.0
    assert region["status"] == "skipped_empty_handoff"
    assert output_path.exists()
    written = geant4_config.read_summary_hdf5(str(output_path))
    assert written["status"] == "skipped_empty_handoff"
    assert written["source_total_weight"] == 0.0


def test_run_handoff_uses_configured_random_seed(monkeypatch, tmp_path):
    simulation, data, _ = distribution_simulation_and_data()
    output_path = tmp_path / "g4.h5"
    geant4_config.configure(
        **distribution_config(
            geant4_output_path=str(output_path),
            random_seed=12345,
        ).__dict__
    )

    def fake_run(cmd, capture_output, text, check):
        payload = geant4_worker.read_payload(cmd[-1])
        assert int(payload["random_seed"]) == 12345
        _write_fake_distribution_output(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    summary = geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert summary["regions"][0]["random_seed"] == 12345


def test_run_handoff_rejects_invalid_random_seed(monkeypatch):
    simulation, data, _ = distribution_simulation_and_data()
    geant4_config.configure(**distribution_config(random_seed=0).__dict__)

    def fail_run(*args, **kwargs):
        raise AssertionError("invalid seed should not spawn a worker")

    monkeypatch.setattr(subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="random_seed"):
        geant4_handoff.run_handoff_from_simulation(simulation, data)


def test_run_handoff_generates_different_default_random_seeds(monkeypatch, tmp_path):
    simulation, data, _ = distribution_simulation_and_data()
    tallies = np.zeros(2, dtype=simulation["tallies"].dtype)
    tallies[0] = simulation["tallies"][0]
    tallies[1] = simulation["tallies"][0]
    tallies[1]["name"] = "second_tally"
    simulation["tallies"] = tallies
    geant4_config.add_config(
        **distribution_config(
            name="first",
            geant4_output_path=str(tmp_path / "a.h5"),
        ).__dict__
    )
    geant4_config.add_config(
        **distribution_config(
            name="second",
            source_tally_name="second_tally",
            geant4_output_path=str(tmp_path / "b.h5"),
        ).__dict__
    )

    seeds = []

    def fake_run(cmd, capture_output, text, check):
        payload = geant4_worker.read_payload(cmd[-1])
        seeds.append(int(payload["random_seed"]))
        _write_fake_distribution_output(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert len(seeds) == 2
    assert seeds[0] > 0
    assert seeds[1] > 0
    assert seeds[0] != seeds[1]


def test_run_handoff_distribution_runs_all_workers_before_failure(monkeypatch, tmp_path):
    simulation, data, _ = distribution_simulation_and_data()
    tallies = np.zeros(2, dtype=simulation["tallies"].dtype)
    tallies[0] = simulation["tallies"][0]
    tallies[1] = simulation["tallies"][0]
    tallies[1]["name"] = "second_tally"
    simulation["tallies"] = tallies
    geant4_config.add_config(
        **distribution_config(
            name="first", geant4_output_path=str(tmp_path / "a.h5")
        ).__dict__
    )
    geant4_config.add_config(
        **distribution_config(name="second", source_tally_name="second_tally").__dict__
    )

    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        return subprocess.CompletedProcess(cmd, 1, "", "also boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc:
        geant4_handoff.run_handoff_from_simulation(simulation, data)
    assert "first" in str(exc.value)
    assert "second" in str(exc.value)
    assert len(calls) == 2


def test_run_handoff_removes_temp_output_on_worker_failure(monkeypatch, tmp_path):
    simulation, data, _ = distribution_simulation_and_data()
    geant4_config.configure(**distribution_config().__dict__)
    temp_outputs = []

    def fake_temporary_path(prefix, suffix):
        path = tmp_path / f"{prefix}{len(temp_outputs)}{suffix}"
        if "_out_" in prefix:
            path.touch()
            temp_outputs.append(path)
        return path

    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(cmd, 1, "", "worker failed")

    monkeypatch.setattr(geant4_handoff, "_temporary_path", fake_temporary_path)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="payload retained"):
        geant4_handoff.run_handoff_from_simulation(simulation, data)

    assert temp_outputs
    assert not temp_outputs[0].exists()
