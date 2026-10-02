import numpy as np

import mcdc.numba_types as type_
from mcdc.constant import TINY
from mcdc.transport.technique import (
    weight_roulette,
    split_from_weight_window,
    particle_bank_module,
)

# =========================================================================== #
# Negative weights (Residual Monte Carlo) must mirror positive weights exactly:
# same random decisions, opposite sign.
# =========================================================================== #


def roulette(w, seed, threshold, target):
    particles = np.zeros(1, type_.particle)
    particles[0]["w"] = w
    particles[0]["alive"] = True
    particles[0]["rng_seed"] = seed
    weight_roulette(particles, threshold, target)
    return particles[0]["alive"], particles[0]["w"]


def test_roulette_negative_weight_mirrors_positive():
    threshold = 0.1 + TINY
    target = 0.2
    N_survived = 0
    for seed in range(1, 201):
        alive_pos, w_pos = roulette(0.1, seed, threshold, target)
        alive_neg, w_neg = roulette(-0.1, seed, threshold, target)
        assert alive_pos == alive_neg
        if alive_pos:
            N_survived += 1
            assert w_pos == target
            assert w_neg == -target
    # Both branches were exercised
    assert 0 < N_survived < 200


def test_roulette_negative_weight_above_threshold_untouched():
    alive, w = roulette(-0.5, 7, 0.1, 0.2)
    assert alive
    assert w == -0.5


def test_split_negative_weight_mirrors_positive(prepare_simulation):
    program, _ = prepare_simulation()
    program = program[0]

    def run_split(initial_weight, w_lower):
        particles = np.zeros(1, type_.particle)
        particles[0]["w"] = initial_weight
        particles[0]["alive"] = True
        particles[0]["rng_seed"] = 3
        start = particle_bank_module.get_bank_size(program["bank_active"])
        split_from_weight_window(
            particles, w_upper=1.0, w_target=0.5, w_lower=w_lower, program=program
        )
        end = particle_bank_module.get_bank_size(program["bank_active"])
        return (
            particles[0]["w"],
            program["bank_active"]["particle_data"][start:end]["w"].copy(),
        )

    for w_lower in (0.0, 0.2):
        w_pos, banked_pos = run_split(2.1, w_lower)
        w_neg, banked_neg = run_split(-2.1, w_lower)
        assert w_neg == -w_pos
        assert len(banked_neg) == len(banked_pos)
        np.testing.assert_array_equal(banked_neg, -banked_pos)
