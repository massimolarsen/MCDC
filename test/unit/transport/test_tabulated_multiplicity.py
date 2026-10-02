import os
import shutil

import h5py
import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle_bank as particle_bank_module
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.transport.physics.neutron.native import sample_inelastic_scattering

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../regression/mcdc-regression_test_data"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "U235-293.6K.h5")),
    reason="regression nuclear data not available",
)

YIELD = 0.5


@pytest.fixture
def tabulated_yield_library(tmp_path, monkeypatch):
    """U-235 whose MT-5 has an energy-dependent (here constant) tabulated yield."""
    path = tmp_path / "U235-293.6K.h5"
    shutil.copy(os.path.join(DATA_DIR, "U235-293.6K.h5"), path)
    with h5py.File(path, "r+") as f:
        group = f["neutron_reactions/inelastic_scattering/MT-005"]
        del group["multiplicity"]
        group.create_dataset("multiplicity", data=-1)
        table = group.create_group("multiplicity_table")
        table.attrs["type"] = "tabulated"
        table.create_dataset("energy", data=np.array([1.0e-11, 30.0]))  # MeV
        table.create_dataset("value", data=np.array([YIELD, YIELD]))
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))


def find_MT5(simulation, data, nuclide):
    for i in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
        ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
            i, nuclide, data
        )
        reaction = simulation["neutron_reactions"][ID]
        if reaction["MT"] == 5:
            return reaction
    raise AssertionError("MT-5 not found")


def test_tabulated_multiplicity(prepare_simulation, tabulated_yield_library):
    fuel = mcdc.Material(nuclide_composition={"U235": 0.05})
    container, data = prepare_simulation(cells=(mcdc.Cell(fill=fuel),))
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    reaction = find_MT5(simulation, data, nuclide)
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    assert inelastic["multiplicity_tabulated"]

    N_reaction = 4000
    N_emitted = 0
    for seed in range(1, N_reaction + 1):
        particle_bank_module.set_bank_size(simulation["bank_active"], 0)
        particles = np.zeros(1, type_.particle)
        particles[0]["E"] = 14.0e6
        particles[0]["w"] = 1.0
        particles[0]["alive"] = True
        particles[0]["uz"] = 1.0
        particles[0]["particle_type"] = PARTICLE_NEUTRON
        particles[0]["rng_seed"] = seed
        collision_data = np.zeros(1, type_.collision_data)
        sample_inelastic_scattering(
            reaction, particles, collision_data, nuclide, simulation, data
        )
        N_emitted += particle_bank_module.get_bank_size(simulation["bank_active"])
        N_emitted += int(particles[0]["alive"])

    # Mean emitted neutrons per reaction = tabulated yield (binomial: 0 or 1 here)
    sigma = np.sqrt(YIELD * (1 - YIELD) / N_reaction)
    assert abs(N_emitted / N_reaction - YIELD) < 5 * sigma


def test_integer_multiplicity_unchanged(prepare_simulation, monkeypatch):
    monkeypatch.setenv("MCDC_LIB", os.path.abspath(DATA_DIR))
    fuel = mcdc.Material(nuclide_composition={"U235": 0.05})
    container, data = prepare_simulation(cells=(mcdc.Cell(fill=fuel),))
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    for i in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
        ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
            i, nuclide, data
        )
        reaction = simulation["neutron_reactions"][ID]
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            reaction["sub_ID"]
        ]
        if reaction["MT"] != 5:
            assert not inelastic["multiplicity_tabulated"]
