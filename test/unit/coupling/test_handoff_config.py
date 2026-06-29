import pytest

from mcdc.coupling import geant4_config, geant4_handoff
from mcdc.coupling.geant4_config import Geant4HandoffConfig

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
