import numpy as np
import pytest

import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_ELECTRON, PARTICLE_NEUTRON
from mcdc.transport.simulation import kill_outside_neutron_energy_window

E_MIN = 1.0e3
E_MAX = 1.0e6


def configure_window(simulation):
    simulation.settings.use_neutron_energy_window = True
    simulation.settings.neutron_energy_min = E_MIN
    simulation.settings.neutron_energy_max = E_MAX


def run_kill(simulation, E, particle_type=PARTICLE_NEUTRON):
    particles = np.zeros(1, type_.particle)
    particles[0]["E"] = E
    particles[0]["alive"] = True
    particles[0]["particle_type"] = particle_type
    kill_outside_neutron_energy_window(particles, simulation)
    return particles[0]["alive"]


@pytest.mark.parametrize(
    "E, alive",
    [
        (0.5 * E_MIN, False),
        (E_MIN, True),
        (1.0e4, True),
        (E_MAX, True),
        (2.0 * E_MAX, False),
    ],
)
def test_neutron_energy_window(prepare_simulation, E, alive):
    simulation, _ = prepare_simulation(configure=configure_window)
    assert run_kill(simulation[0], E) == alive


def test_energy_window_ignores_other_particles(prepare_simulation):
    simulation, _ = prepare_simulation(configure=configure_window)
    assert run_kill(simulation[0], 0.5 * E_MIN, PARTICLE_ELECTRON)


def test_energy_window_default_off(prepare_simulation):
    simulation, _ = prepare_simulation()
    settings = simulation[0]["settings"]
    assert not settings["use_neutron_energy_window"]
    assert not settings["neutron_fission_all_prompt"]
