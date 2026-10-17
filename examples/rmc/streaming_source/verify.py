"""Independent checks of the experimental characteristic estimator."""

import numpy as np
import solver as s


def check_characteristic_matches_exact_heterogeneous_ray():
    edges, _ = s.geometry(([0, 1, 2], [0, 1], [0, 1]))
    value = s.characteristic(
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 0.0, 0.0]),
        edges,
        np.array([2.0, 0.0]),
        np.array([3.0, 5.0]),
    )
    exact = 3 / 2 * (1 - np.exp(-1)) + np.exp(-1) * 5
    assert np.isclose(value, exact, atol=1e-14)


def check_first_flight_conservation():
    edges, volume = s.geometry(([0, 1, 2], [0, 1, 2], [0, 1, 2]))
    sigma = np.linspace(0.1, 4, 8)
    P = s.first_flight_matrix(s.sobol_cells(8, 7, 6), edges, sigma, volume)
    assert np.all(P >= 0)
    # Integrated collision probability of every source history is at most one.
    assert np.all((volume * sigma) @ P <= volume * (1 + 1e-13))


def check_void_source_has_no_scattering_residual():
    result = s.solve(
        ([0, 1, 2],) * 3,
        np.zeros(8),
        np.zeros(8),
        np.full(8, 0.125),
        N=2,
        inner=2,
        power=6,
        repeats=4,
        pilot=0,
    )
    assert np.all(np.isfinite(result["mean"]))
    assert np.all(result["Q"] == 0)
    assert result["weight_second"] == 0


def check_linear_source_characteristic(sigma):
    edges, _ = s.geometry(([0, 1],) * 3)
    actual = s.characteristic(
        np.array([0.2, 0.5, 0.5]),
        np.array([1.0, 0.0, 0.0]),
        edges,
        np.array([sigma]),
        np.array([3.0]),
        np.array([[1.0, 0.0, 0.0]]),
    )
    x, w = np.polynomial.legendre.leggauss(24)
    distance = 0.4 * (x + 1)
    exact = np.sum(0.4 * w * (2.4 + 2 * distance) * np.exp(-sigma * distance))
    assert np.isclose(actual, exact, rtol=1e-13, atol=1e-14)


def check_absorber_and_scattering_agree_with_smc(kind):
    edges = ([0, 1, 2],) * 3
    sigma = np.linspace(0.5, 2.0, 8)
    q = np.zeros(8)
    q[0] = 1.0
    for ratio in (0.0, 0.8):
        ref = s.smc(edges, sigma, ratio * sigma, q, N=150000, seed=72)
        runs = [
            s.solve(
                edges,
                sigma,
                ratio * sigma,
                q,
                N=4000,
                inner=16,
                power=8,
                repeats=8,
                seed=101 + k,
                forward=kind == "forward",
                targeted=kind == "targeted",
                split=kind == "split",
                linear=kind == "linear",
                antithetic=kind == "antithetic",
                fit_source=kind != "unfitted",
            )
            for k in range(8)
        ]
        mean = np.mean([r["mean"] for r in runs], axis=0)
        variance = np.sum([r["variance"] for r in runs], axis=0) / len(runs) ** 2
        z = (mean - ref["mean"]) / np.sqrt(variance + ref["variance"])
        assert np.max(np.abs(z)) < 5.0, (ratio, z)
        if ratio == 0.0:
            assert all(np.all(r["Q"] == 0) for r in runs)
            assert all(r["weight_second"] == 0 for r in runs)


if __name__ == "__main__":
    check_characteristic_matches_exact_heterogeneous_ray()
    check_first_flight_conservation()
    check_void_source_has_no_scattering_residual()
    for sigma in (0.0, 1e-9, 2.0):
        check_linear_source_characteristic(sigma)
    for kind in (
        "backward",
        "forward",
        "targeted",
        "split",
        "linear",
        "antithetic",
        "unfitted",
    ):
        check_absorber_and_scattering_agree_with_smc(kind)
        print(f"{kind}: independent SMC agreement checks passed", flush=True)
    print("Characteristic, conservation, void and estimator checks passed.")
