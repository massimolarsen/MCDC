"""
Infinite, homogeneous Al-27 with a constant scattering fraction sigma_s / sigma_t = 1/4.

Reproduces [D] Figs. 3.3-3.6 (see ../common.py for references):
  - flux_low.png        10^-4 to 10^3 eV, G = 100: RMC vs SMC (10^6 histories)
  - flux_high.png       10^6 to 10^7 eV, G = 1000: RMC vs SMC
  - flux_high_zoom.png  the same, 7 to 10 MeV
  - convergence.png     high-energy case: ||eps~||_2 and L-infinity to the SMC (and the
                        deterministic) reference per iteration

Problem ([D] Sec. 3.4.2): sigma_t(E) of Al-27 (MC/DC library, ENDF/B-VIII.1 at 293.6 K)
with sigma_s = sigma_t / 4 (isotropic elastic in the COM frame, A = 26.75) and
sigma_c = 3 sigma_t / 4; uniform isotropic source over the window; 2 polar bins;
RMC with 100 P/B/I for 20 iterations. Atom density 0.0602 /b-cm.

Differences from the dissertation:
  - Low-energy case: below 400 kT (3.4 meV at 0.1 K) MC/DC samples free-gas target
    motion, which RMC's target-at-rest kernels exclude. The low case is therefore run
    with all energies scaled up by 10^4 (data, window, source); target-at-rest
    slowing down is invariant under this scaling, so results are mapped back exactly
    (E / 10^4, flux x 10^4).
  - SMC histories for the high-energy case: N_SMC_HIGH below ([D] used 10^9).
  - A deterministic reference (../common.py) is included for both cases.
  - Low case: with the source uniform per eV over 10^-4 to 10^3 eV, few SMC histories
    reach the lowest decades (3/4 of collisions absorb), so SMC is very noisy there;
    RMC agrees with the deterministic reference to ~2e-5 (median). The flux shape
    differs from [D] Fig. 3.3 (which rises ~E^0.5); the source normalization and
    sigma_t data of [D] are not fully specified.

Run with MCDC_LIB pointing to the MC/DC library (writes output.h5; then plot.py):
    python input.py --mode=numba
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

DENSITY = 0.0602
SCALE_LOW = 1.0e4
CASES = {
    # name: (window [eV], G, energy scale)
    "low": ((1.0e-4, 1.0e3), 100, SCALE_LOW),
    "high": ((1.0e6, 1.0e7), 1000, 1.0),
}
N_ITERATION, PARTICLES = 20, 100
N_SMC_LOW, N_SMC_HIGH = 1e6, 1e7


def make_model_factory(name, window):
    def make_model():
        import mcdc

        material = mcdc.Material(nuclide_composition={name: DENSITY}, temperature=0.1)
        simulation = common.slab([(0.0, 1.0, material)])
        simulation.set_sources([common.volume_source(0.0, 1.0, window)])
        return simulation, [material]

    return make_model


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    energy, sigma_t = common.read_total_xs("Al27", 293.6)
    A = common.read_atomic_weight_ratio("Al27", 293.6)
    output = common.open_output()
    output.create_dataset("data/energy", data=energy)
    output.create_dataset("data/sigma_t", data=sigma_t)

    nuclides = {}
    for case, (window, G, scale) in CASES.items():
        name = f"AL{case.upper()}"
        common.write_nuclide(
            "data", name, A, energy * scale, sigma_t / 4, 3 * sigma_t / 4
        )
        nuclides[case] = (name, energy * scale)
    common.use_library("data")

    for case, (window, G, scale) in CASES.items():
        name, data_energy = nuclides[case]
        window = (window[0] * scale, window[1] * scale)
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        z_edges = [0.0, 1.0]
        make_model = make_model_factory(name, window)
        reference = common.slowing_down_reference(
            E_edges,
            [(A, DENSITY, data_energy, sigma_t / 4, sigma_t)],
            window,
            h=1e-4 if case == "low" else 1e-5,
        )
        output.create_dataset(f"reference/{case}", data=reference)
        Q = common.uniform_Q(z_edges, E_edges, [0], window)
        result = common.run_rmc(
            make_model,
            z_edges,
            E_edges,
            Q,
            N_ITERATION,
            PARTICLES / 2,
            cache_dir="cache",
        )
        common.save_rmc(output, case, result, energy_scale=scale)
        N = N_SMC_LOW if case == "low" else N_SMC_HIGH
        mean, sdev, wall = common.run_smc(make_model, z_edges, E_edges, N)
        common.save_smc(output, case, mean, sdev, wall, N, energy_scale=scale)
    output.close()


if __name__ == "__main__":
    main()
