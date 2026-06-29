import pytest

from mcdc.coupling import geant4_config


@pytest.fixture(autouse=True)
def clear_geant4_configs():
    geant4_config.clear_configs()
    yield
    geant4_config.clear_configs()
