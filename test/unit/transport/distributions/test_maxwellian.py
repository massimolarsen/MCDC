import math

import mcdc.transport.distribution as dist
from mcdc.constant import PI

from .test_data import make_test_table_data_constant


def test_maxwellian_sample(rng_sequence, rng_state):
    table, data = make_test_table_data_constant(1.0)
    mcdc = {"table_data": [table]}
    maxwellian = {"nuclear_temperature_ID": 0, "restriction_energy": 0.0}

    xi1, xi2, xi3 = 0.9, 0.9, 0.0
    rng_sequence([xi1, xi2, xi3])

    sample = dist.sample_maxwellian(
        2.0, rng_state, maxwellian, mcdc, data
    )
    expected = -(math.log(xi1) + math.log(xi2) * math.cos(0.5 * PI * xi3) ** 2)

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)
