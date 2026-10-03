import math

import numpy as np

import mcdc.numba_types as type_
from mcdc.constant import BOLTZMANN_K, LIGHT_SPEED, NEUTRON_MASS, PARTICLE_NEUTRON
from mcdc.transport.physics.neutron.native import sample_nucleus_velocity


def test_free_gas_target_speed_uses_nuclide_temperature():
    """
    For a nearly stationary neutron the constant-cross-section free-gas sampler draws
    target speeds V with density ~ V^3 exp(-beta^2 V^2) (Maxwellian weighted by the
    relative speed, here ~ V), so <V^2> = 2 / beta^2 with beta = sqrt(M / (2 k T)).
    Catches both a fixed temperature (293.6 K) and an inverted relative-speed
    rejection (which gives the unweighted Maxwellian, <V^2> = 3 / (2 beta^2)).
    """
    A, temperature = 1.0, 600.0
    beta = math.sqrt(A * NEUTRON_MASS / (2.0 * BOLTZMANN_K * temperature)) / LIGHT_SPEED

    N = 20000
    V2 = np.empty(N)
    particles = np.zeros(1, type_.particle)
    particles[0]["E"] = 1.0e-8  # eV: speed << thermal target speed
    particles[0]["uz"] = 1.0
    particles[0]["particle_type"] = PARTICLE_NEUTRON
    particles[0]["rng_seed"] = 1
    for i in range(N):
        Vx, Vy, Vz = sample_nucleus_velocity(A, temperature, particles)
        V2[i] = Vx * Vx + Vy * Vy + Vz * Vz

    expected = 2.0 / beta**2
    # <V^2> beta^2 ~ Gamma(2, 1): relative standard error 1 / sqrt(2 N)
    assert abs(V2.mean() / expected - 1.0) < 4.0 / math.sqrt(2.0 * N)
