import numpy as np

import mcdc.transport.rng as rng
from mcdc.main import preparation
from mcdc.object_.distribution import DistributionMultiTable
from mcdc.object_.simulation import simulation
from mcdc.transport.distribution import sample_correlated_multi_table


def test_elastic_angular_tables_use_one_correlated_quantile():
    simulation.__init__()
    try:
        angles = DistributionMultiTable(
            np.array([1.0, 3.0]),
            np.array([0, 2]),
            np.array([0.0, 0.5, 0.5, 1.0]),
            cdf=np.array([0.0, 1.0, 0.0, 1.0]),
        )
        container, data = preparation()
        simulation_data = container[0]
        table = simulation_data["multi_table_distributions"][angles.child_ID]

        for energy, lower_bound in (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.25),
            (3.0, 0.5),
            (4.0, 0.5),
        ):
            state = np.zeros(1, dtype=[("rng_seed", np.uint64)])
            state[0]["rng_seed"] = 1
            expected_state = state.copy()
            quantile = rng.lcg(expected_state)

            sampled = sample_correlated_multi_table(
                energy, state, table, simulation_data, data
            )

            np.testing.assert_allclose(sampled, lower_bound + 0.5 * quantile)
            assert state[0]["rng_seed"] == expected_state[0]["rng_seed"]
    finally:
        simulation.__init__()
