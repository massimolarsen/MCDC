import math

import mcdc.transport.distribution as dist

from .test_data import make_test_table_data_constant


def test_evaporation_sample(rng_sequence, rng_state):
    table, data = make_test_table_data_constant(1.0)
    mcdc = {"table_data": [table]}
    evaporation = {"nuclear_temperature_ID": 0, "restriction_energy": 0.0}

    xi1, xi2 = 0.1, 0.2
    rng_sequence([xi1, xi2])

    sample = dist.sample_evaporation(
        2.0, rng_state, evaporation, mcdc, data
    )
    w = 2.0
    g = 1.0 - math.exp(-w)
    expected = -math.log((1.0 - g * xi1) * (1.0 - g * xi2))

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)
