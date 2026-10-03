import os
import shutil

import h5py
import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.transport.distribution import sample_distribution

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../regression/mcdc-regression_test_data"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "H1-293.6K.h5")),
    reason="regression nuclear data not available",
)

THETA, U = 0.3, 7.0  # nuclear temperature and restriction energy [MeV]


def add_reaction(f, MT, spectrum_type, interpolation_key):
    """An inelastic reaction with a constant-temperature spectrum, as generated from
    ACE data with no interpolation regions (NR = 0)."""
    N = len(f["neutron_reactions/xs_energy_grid"])
    group = f.create_group(f"neutron_reactions/inelastic_scattering/MT-{MT:03}")
    group.attrs["MT"] = MT
    xs = group.create_dataset("xs", data=np.ones(N))
    xs.attrs["offset"] = 0
    group.create_dataset("Q-value", data=-U)
    group.create_dataset("multiplicity", data=1)
    group.create_dataset("reference_frame", data="LAB")
    group.create_group("angular_cosine_distribution").attrs["type"] = "isotropic"
    group.create_dataset("spectrum_probability", data=np.ones((1, 1)))
    group.create_dataset("spectrum_probability_grid", data=[0.0, 30.0])
    spectrum = group.create_group("energy_spectrum-1")
    spectrum.attrs["type"] = spectrum_type
    spectrum.create_dataset(interpolation_key, data=np.array([], dtype="S10"))
    spectrum.create_dataset("interpolation_boundaries", data=np.array([], dtype=int))
    spectrum.create_dataset("temperature_energy_grid", data=[U, 20.0, 150.0])
    spectrum.create_dataset("temperature", data=[THETA, THETA, THETA])
    spectrum.create_dataset("restriction_energy", data=U)


@pytest.fixture
def library(tmp_path, monkeypatch):
    path = tmp_path / "H1-293.6K.h5"
    shutil.copy(os.path.join(DATA_DIR, "H1-293.6K.h5"), path)
    with h5py.File(path, "r+") as f:
        add_reaction(f, 91, "evaporation", "temperature_interpolations")
        # Older generators wrote the singular key for Maxwellian spectra
        add_reaction(f, 92, "maxwellian", "temperature_interpolation")
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))


def test_evaporation_and_maxwellian_load_and_sample(prepare_simulation, library):
    material = mcdc.Material(nuclide_composition={"H1": 0.05})
    container, data = prepare_simulation(cells=(mcdc.Cell(fill=material),))
    simulation = container[0]
    nuclide = simulation["nuclides"][0]

    state = np.zeros(1, type_.particle)
    state[0]["particle_type"] = PARTICLE_NEUTRON
    state[0]["rng_seed"] = 11
    E = 14.0e6
    limit = E - U * 1e6
    x = np.linspace(0.0, limit, 20001)
    N = 20000
    for i, power in enumerate(
        (1.0, 0.5)
    ):  # evaporation x e^-x/T, Maxwellian sqrt(x) e^-x/T
        ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
            i, nuclide, data
        )
        inelastic = simulation["neutron_inelastic_scattering_reactions"][
            simulation["neutron_reactions"][ID]["sub_ID"]
        ]
        spectrum = simulation["distributions"][
            mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
                0, inelastic, data
            )
        ]
        samples = np.array(
            [
                sample_distribution(E, spectrum, state, simulation, data)
                for _ in range(N)
            ]
        )
        pdf = x**power * np.exp(-x / (THETA * 1e6))
        mean = np.trapezoid(x * pdf, x) / np.trapezoid(pdf, x)
        std = np.sqrt(np.trapezoid((x - mean) ** 2 * pdf, x) / np.trapezoid(pdf, x))
        assert samples.min() >= 0.0 and samples.max() <= limit
        assert abs(samples.mean() - mean) < 4.0 * std / np.sqrt(N)
