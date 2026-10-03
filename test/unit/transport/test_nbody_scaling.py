import os
import shutil

import h5py
import numpy as np
import pytest

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.transport.distribution import sample_correlated_distribution

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "../../regression/mcdc-regression_test_data"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "H1-293.6K.h5")),
    reason="regression nuclear data not available",
)

AP, Q = 2.0, -2.0  # total mass ratio, Q-value [MeV]


@pytest.fixture
def nbody_library(tmp_path, monkeypatch):
    """H-1 with a synthetic N-body (Law 66) inelastic reaction, as the generator writes it."""
    path = tmp_path / "H1-293.6K.h5"
    shutil.copy(os.path.join(DATA_DIR, "H1-293.6K.h5"), path)
    with h5py.File(path, "r+") as f:
        N = len(f["neutron_reactions/xs_energy_grid"])
        group = f.create_group("neutron_reactions/inelastic_scattering/MT-016")
        group.attrs["MT"] = 16
        xs = group.create_dataset("xs", data=np.ones(N))
        xs.attrs["offset"] = 0
        group.create_dataset("Q-value", data=Q)
        group.create_dataset("multiplicity", data=2)
        group.create_dataset("reference_frame", data="COM")
        group.create_group("angular_cosine_distribution").attrs[
            "type"
        ] = "energy-correlated"
        group.create_dataset("spectrum_probability", data=np.ones((1, 1)))
        group.create_dataset("spectrum_probability_grid", data=[0.0, 30.0])
        spectrum = group.create_group("energy_spectrum-1")
        spectrum.attrs["type"] = "N-body"
        T = np.linspace(0.0, 1.0, 201)
        spectrum.create_dataset("value", data=T)
        spectrum.create_dataset("pdf", data=np.sqrt(T) * (1.0 - T) ** 0.5)  # n = 3
        spectrum.create_dataset("number_of_particles", data=3)
        spectrum.create_dataset("total_mass_ratio", data=AP)
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))


def test_nbody_energy_scales_with_incident_energy(prepare_simulation, nbody_library):
    material = mcdc.Material(nuclide_composition={"H1": 0.05})
    container, data = prepare_simulation(cells=(mcdc.Cell(fill=material),))
    simulation = container[0]
    nuclide = simulation["nuclides"][0]
    ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(0, nuclide, data)
    reaction = simulation["neutron_reactions"][ID]
    inelastic = simulation["neutron_inelastic_scattering_reactions"][reaction["sub_ID"]]
    spectrum = simulation["distributions"][
        mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
            0, inelastic, data
        )
    ]
    A = nuclide["atomic_weight_ratio"]

    T = np.linspace(0.0, 1.0, 20001)
    p = np.sqrt(T) * (1.0 - T) ** 0.5
    mean_T = np.trapezoid(T * p, T) / np.trapezoid(p, T)

    state = np.zeros(1, type_.particle)
    state[0]["particle_type"] = PARTICLE_NEUTRON
    state[0]["rng_seed"] = 7
    N = 20000
    for E in (10.0e6, 20.0e6):
        E_max = (AP - 1.0) / AP * (A / (A + 1.0) * E + Q * 1e6)
        samples = np.array(
            [
                sample_correlated_distribution(E, spectrum, state, simulation, data)[0]
                for _ in range(N)
            ]
        )
        assert samples.min() >= 0.0 and samples.max() <= E_max * (1.0 + 1e-12)
        # Var[T] < 1/4, so the standard error of the mean is below E_max / (2 sqrt(N))
        assert abs(samples.mean() - mean_T * E_max) < 4.0 * E_max / (2.0 * np.sqrt(N))
