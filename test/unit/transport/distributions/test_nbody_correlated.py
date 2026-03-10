import math

import mcdc.transport.distribution as dist
from mcdc.constant import DISTRIBUTION_N_BODY

from .test_data import make_test_tabulated_data


def test_nbody_sample_correlated(rng_sequence, rng_state):
    table, data = make_test_tabulated_data([2.0, 4.0, 6.0], [0.0, 0.4, 1.0])
    mcdc = {"nbody_distributions": [table]}
    distribution = {"child_type": DISTRIBUTION_N_BODY, "child_ID": 0}

    # First value samples energy, second value samples isotropic cosine.
    rng_sequence([0.2, 0.75])

    E, mu = dist.sample_correlated_distribution(
        2.0,
        distribution,
        rng_state,
        mcdc,
        data,
    )

    expected_E = 2.0 + (0.2 - 0.0) * (4.0 - 2.0) / (0.4 - 0.0)
    expected_mu = 2.0 * 0.75 - 1.0

    assert math.isclose(E, expected_E, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, expected_mu, rel_tol=0.0, abs_tol=1e-12)
