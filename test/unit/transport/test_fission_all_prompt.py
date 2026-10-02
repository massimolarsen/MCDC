import os

import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.transport.physics.neutron.native import sample_fission

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../regression/mcdc-regression_test_data"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "U235-293.6K.h5")),
    reason="regression nuclear data not available",
)


def count_delayed(prepare_simulation, monkeypatch, all_prompt, N_fission=2000):
    """Number of fission neutrons born later than their parent (i.e., delayed)."""
    monkeypatch.setenv("MCDC_LIB", os.path.abspath(DATA_DIR))
    fuel = mcdc.Material(nuclide_composition={"U235": 0.05})

    def configure(simulation):
        simulation.settings.neutron_fission_all_prompt = all_prompt
        simulation.settings.active_bank_buffer = 100

    container, data = prepare_simulation(
        cells=(mcdc.Cell(fill=fuel),), configure=configure
    )
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    reaction_ID = mcdc_get.nuclide.neutron_fission_reaction_IDs(0, nuclide, data)
    reaction = simulation["neutron_reactions"][reaction_ID]

    N_delayed = 0
    for seed in range(1, N_fission + 1):
        particle_bank_module.set_bank_size(simulation["bank_active"], 0)
        particles = np.zeros(1, type_.particle)
        particles[0]["E"] = 1.0e6
        particles[0]["w"] = 1.0
        particles[0]["alive"] = True
        particles[0]["uz"] = 1.0
        particles[0]["particle_type"] = PARTICLE_NEUTRON
        particles[0]["rng_seed"] = seed
        collision_data = np.zeros(1, type_.collision_data)

        sample_fission(reaction, particles, collision_data, nuclide, simulation, data)

        size = particle_bank_module.get_bank_size(simulation["bank_active"])
        times = simulation["bank_active"]["particle_data"][:size]["t"]
        N_delayed += np.count_nonzero(times > 0.0)
        if particles[0]["alive"] and particles[0]["t"] > 0.0:
            N_delayed += 1
    return N_delayed


def test_fission_all_prompt(prepare_simulation, monkeypatch):
    # Control: delayed neutrons do occur by default (nu_d/nu ~ 0.6% for U-235)
    assert count_delayed(prepare_simulation, monkeypatch, all_prompt=False) > 0
    assert count_delayed(prepare_simulation, monkeypatch, all_prompt=True) == 0
