import math

import mcdc.transport.distribution as dist

from .test_data import make_test_kalbach_mann_data


def test_kalbach_mann_sample(rng_sequence, rng_state):
    kalbach, data = make_test_kalbach_mann_data()

    # xi1 chooses table 1, xi2 samples within it, xi3/xi4 set mu deterministically.
    xi1, xi2, xi3, xi4 = 0.3, 0.1, 0.7, 0.5
    rng_sequence([xi1, xi2, xi3, xi4])

    E, mu = dist.sample_kalbach_mann(2.0, rng_state, kalbach, data)

    # Expected values from analytic reconstruction of the configured tables.
    E_min, E_max = 1.5, 4.5
    E_hat = 2.0 + (xi2 - 0.0) / 0.2
    E_new = E_min + (E_hat - 2.0) / (6.0 - 2.0) * (E_max - E_min)

    assert math.isclose(E, E_new, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mu, 0.0, rel_tol=0.0, abs_tol=1e-12)
