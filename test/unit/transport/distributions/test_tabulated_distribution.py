import math

import mcdc.transport.distribution as dist

from .test_data import make_test_tabulated_data


def test_tabulated_distribution_sample(rng_sequence, rng_state):
    table, data = make_test_tabulated_data([1.0, 3.0, 7.0], [0.0, 0.4, 1.0])
    rng_sequence([0.2])

    sample = dist.sample_tabulated(table, rng_state, data)
    expected = 1.0 + (0.2 - 0.0) * (3.0 - 1.0) / (0.4 - 0.0)

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)
