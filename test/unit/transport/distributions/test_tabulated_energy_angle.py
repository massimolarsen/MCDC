import math

import mcdc.transport.distribution as dist

from .test_data import make_test_tabulated_energy_angle_data


def test_tabulated_energy_angle_sample(rng_sequence, rng_state):
    table, data = make_test_tabulated_energy_angle_data()

    xi1, xi2, xi3 = 0.3, 0.1, 0.25
    rng_sequence([xi1, xi2, xi3])

    E, mu = dist.sample_tabulated_energy_angle(
        2.0, rng_state, table, data
    )

    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    E_new = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)
    mu_expected = -1.0 + (xi3 - 0.0) / 0.5

    assert math.isclose(E, E_new, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, mu_expected, rel_tol=0.0, abs_tol=1e-12)
