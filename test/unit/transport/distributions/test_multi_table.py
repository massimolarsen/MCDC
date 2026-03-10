import math

import mcdc.transport.distribution as dist

from .test_data import make_test_multi_table_data


def test_multi_table_distribution_sample(rng_sequence, rng_state):
    multi_table, data = make_test_multi_table_data()
    # xi0 chooses table (xi0 < 0.5 -> table 1), xi1 samples within that table.
    rng_sequence([0.3, 0.2])

    sample = dist.sample_multi_table(2.0, rng_state, multi_table, data)

    # Expected from the second table (values 100..300), first bin.
    expected = 100.0 + (0.2 - 0.0) / 0.01

    assert math.isclose(sample, expected, rel_tol=0.0, abs_tol=1e-12)
