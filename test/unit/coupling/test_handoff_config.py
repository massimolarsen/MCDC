import pytest

from mcdc.coupling import geant4_config, geant4_handoff
from mcdc.coupling.geant4_config import Geant4HandoffConfig
from mcdc.coupling.geant4_config import normalized_device_components

from ._helpers import distribution_config, distribution_simulation_and_data


def test_enable_add_disable_geant4_handoff_config_list():
    geant4_handoff.configure(**distribution_config(name="one").__dict__)
    assert geant4_handoff.has_configs()
    assert len(geant4_config.CONFIGS) == 1

    geant4_handoff.add_handoff(**distribution_config(name="two").__dict__)
    assert len(geant4_config.CONFIGS) == 2

    geant4_handoff.disable()
    assert not geant4_handoff.has_configs()


def test_run_handoff_rejects_mixed_modes():
    simulation, data, _ = distribution_simulation_and_data()
    geant4_config.add_config(**Geant4HandoffConfig(source_mode="bank").__dict__)
    geant4_config.add_config(**distribution_config(name="dist").__dict__)

    with pytest.raises(RuntimeError, match="cannot mix"):
        geant4_handoff.run_handoff_from_simulation(simulation, data)


def test_run_handoff_rejects_multiple_bank_regions():
    simulation, data, _ = distribution_simulation_and_data()
    geant4_config.add_config(
        **Geant4HandoffConfig(name="a", source_mode="bank").__dict__
    )
    geant4_config.add_config(
        **Geant4HandoffConfig(name="b", source_mode="bank").__dict__
    )

    with pytest.raises(RuntimeError, match="Multiple Geant4 bank"):
        geant4_handoff.run_handoff_from_simulation(simulation, data)


def test_normalized_device_components_synthesizes_legacy_detector():
    cfg = Geant4HandoffConfig(
        detector_size_mm=(1.0, 2.0, 3.0),
        detector_material="G4_Si",
    )

    components = normalized_device_components(cfg)

    assert components[0]["name"] == "detector"
    assert components[0]["material"] == "G4_Si"
    assert components[0]["score"] is True
    assert components[0]["center_mm"].tolist() == [0.0, 0.0, 0.0]
    assert components[0]["size_mm"].tolist() == [1.0, 2.0, 3.0]


def test_normalized_device_components_rejects_bad_schema():
    cfg = Geant4HandoffConfig(
        device_components=[
            {
                "name": "die",
                "material": "G4_Si",
                "center_mm": [0.0, 0.0],
                "size_mm": [1.0, 1.0, 1.0],
                "score": True,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="center_mm"):
        normalized_device_components(cfg)
