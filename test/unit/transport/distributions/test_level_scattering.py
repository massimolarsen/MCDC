import math

import mcdc.transport.distribution as dist


def test_level_scattering_sample():
    level = {"C1": 1.0, "C2": 0.5}
    sample = dist.sample_level_scattering(5.0, level)
    assert math.isclose(sample, 2.0, rel_tol=0.0, abs_tol=1e-12)
