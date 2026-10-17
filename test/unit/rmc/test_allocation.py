"""History allocation between residual terms (mcdc.rmc.driver.allocate_histories)."""

import numpy as np

from mcdc.rmc.driver import allocate_histories


def test_proportional_and_exact_total():
    N = allocate_histories(1000, [3.0, 1.0, 0.0])
    assert N.sum() == 1000
    assert list(N) == [750, 250, 0]


def test_small_terms_get_one_history():
    N = allocate_histories(100, [1.0, 1e-9, 0.0])
    assert N.sum() == 100
    assert N[1] == 1 and N[2] == 0


def test_rounding_keeps_the_total():
    rng = np.random.default_rng(0)
    for _ in range(200):
        masses = rng.random(3) * (rng.random(3) > 0.2)
        N = int(rng.integers(1, 5000))
        out = allocate_histories(N, masses)
        if masses.sum() > 0:
            assert out.sum() == N
            assert np.all(out[masses == 0] == 0)
            # within one history of the exact share (or the one-history minimum)
            share = N * masses / masses.sum()
            assert np.all(np.abs(out - share) <= 1.0 + 1e-9)


def test_no_mass_or_no_histories():
    assert allocate_histories(10, [0.0, 0.0]).sum() == 0
    assert allocate_histories(0, [1.0, 2.0]).sum() == 0
