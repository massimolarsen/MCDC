"""
1D fuel rod pin cell: H2O | UO2 | H2O slab with reflective boundaries.

Reproduces [D] Figs. 4.5-4.8 (see ../common.py for references):
  - flux_space_thermal.png    energy-integrated flux per z cell, thermal case
  - flux_energy_thermal.png   flux spectrum in the H2O and UO2 regions, thermal case
  - flux_space_fast.png       the same, fast case
  - flux_energy_fast.png
The residual-error histories are used by ../convergence/plot.py ([D] Figs. 4.9-4.10).

Problem ([D] Sec. 4.4.2): z in [0, 3] cm, H2O (1 g/cc) in [0, 1] and [2, 3], UO2
(10.2 g/cc, 4% U-235) in [1, 2]; 24 uniform z cells, 2 polar bins.
  - Thermal: window [1e-4, 1e-3] eV, G = 100, source uniform in energy over the whole
    slab, RMC with 1000 P/B/I for 10 iterations, SMC with 10^7 histories.
  - Fast: window [3, 10] MeV, G = 40, source uniform in energy in the UO2, RMC with
    10^4 P/B/I for 10 iterations, SMC with 10^7 histories ([D]: 10^5 P/B/I and 10^8,
    reduced for run time on the machine used).
P/B/I counts histories per energy bin per iteration, as in [D] (10^3 P/B/I with
G = 100 gives 10^5 histories per iteration).

Differences from the dissertation:
  - Thermal: MC/DC samples free-gas target motion below 400 kT, and real H in water
    would need S(alpha, beta). As in [D] (isotropic elastic, target at rest), the
    thermal case uses synthetic nuclides built from the 0.1 K cross sections of H-1,
    O-16, U-235 and U-238: isotropic COM elastic and capture, with U-235 fission
    counted as capture (fission neutrons are born far above the window and are killed).
    To stay above the free-gas threshold, all energies are scaled up by 10^4; target-
    at-rest slowing down is invariant under this scaling, and results are mapped back.
  - Fast: real ENDF/B-VIII.1 data at 293.6 K for all reactions (anisotropic elastic,
    all inelastic laws, the U-235 and U-238 fission spectra and nu(E), all prompt),
    where [D] used isotropic elastic scattering, a Watt spectrum and nu = 2.
  - UO2 composition in atom fractions (4 at.% U-235).

Run with MCDC_LIB pointing to the MC/DC library (writes output.h5; then plot.py):
    python input.py --mode=numba
FUEL_ROD_CASES=thermal (or fast) runs one case; FUEL_ROD_OUTPUT names the output file;
FUEL_ROD_RUNS=rmc (or smc) runs one method (default both).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

SCALE_THERMAL = 1.0e4
N_UO2 = 10.2 * 0.602214 / (0.04 * 235.044 + 0.96 * 238.051 + 2 * 15.995)
UO2 = {"U235": 0.04 * N_UO2, "U238": 0.96 * N_UO2, "O16": 2 * N_UO2}
H2O = {"H1": 2 * 0.033428, "O16": 0.033428}
Z_EDGES = np.linspace(0.0, 3.0, 25)
REGIONS = ("H2O", "UO2", "H2O")
CASES = {
    # name: (window [eV], G, P/B/I, source cells, SMC histories)
    "thermal": ((1.0e-4, 1.0e-3), 100, 1.0e3, range(0, 24), 1e7),
    "fast": ((3.0e6, 1.0e7), 40, 1.0e4, range(8, 16), 1e7),
}
N_ITERATION = 10


def make_model_factory(window, suffix, temperature, source_cells):
    def make_model():
        import mcdc

        def material(composition):
            return mcdc.Material(
                nuclide_composition={k + suffix: v for k, v in composition.items()},
                temperature=temperature,
            )

        h2o, uo2 = material(H2O), material(UO2)
        simulation = common.slab([(0.0, 1.0, h2o), (1.0, 2.0, uo2), (2.0, 3.0, h2o)])
        z_low, z_high = Z_EDGES[source_cells[0]], Z_EDGES[source_cells[-1] + 1]
        simulation.set_sources([common.volume_source(z_low, z_high, window)])
        cells = [h2o if z < 1.0 or z >= 2.0 else uo2 for z in Z_EDGES[:-1]]
        return simulation, cells

    return make_model


def write_thermal_nuclides():
    """Synthetic target-at-rest nuclides from the 0.1 K data, energies scaled."""
    for name in ("H1", "O16", "U235", "U238"):
        energy, elastic = common.read_xs(name, 0.1, "elastic_scattering")
        _, capture = common.read_xs(name, 0.1, "capture")
        _, fission = common.read_xs(name, 0.1, "fission")
        keep = (energy > 1.0e-5) & (energy < 1.0e-2)
        common.write_nuclide(
            "data",
            name + "T",
            common.read_atomic_weight_ratio(name, 0.1),
            energy[keep] * SCALE_THERMAL,
            elastic[keep],
            (capture + fission)[keep],
        )


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    real_library = common.real_library()
    if common.is_master():
        write_thermal_nuclides()
    common.barrier()
    output = common.open_output(os.environ.get("FUEL_ROD_OUTPUT", "output.h5"))
    cases = os.environ.get("FUEL_ROD_CASES", "thermal,fast").split(",")
    runs = os.environ.get("FUEL_ROD_RUNS", "rmc,smc").split(",")
    for name, (window, G, P, source_cells, N_smc) in CASES.items():
        if name not in cases:
            continue
        if name == "thermal":
            common.use_library("data")
            scale, suffix, temperature = SCALE_THERMAL, "T", 0.1
        else:
            common.use_library(real_library)
            scale, suffix, temperature = 1.0, "", 293.6
        window = (window[0] * scale, window[1] * scale)
        E_edges = np.logspace(np.log10(window[0]), np.log10(window[1]), G + 1)
        K, J = len(Z_EDGES) - 1, len(common.MU_EDGES) - 1
        Q = common.uniform_Q(Z_EDGES, E_edges, list(source_cells), window)
        make_model = make_model_factory(window, suffix, temperature, source_cells)
        if "rmc" in runs:
            result = common.run_rmc(
                make_model,
                Z_EDGES,
                E_edges,
                Q,
                N_ITERATION,
                P / (K * J),
                cache_dir="cache",
            )
            common.save_rmc(output, name, result, energy_scale=scale, P=P)
        if "smc" in runs:
            mean, sdev, wall = common.run_smc(make_model, Z_EDGES, E_edges, N_smc)
            common.save_smc(output, name, mean, sdev, wall, N_smc, energy_scale=scale)
    output.close()


if __name__ == "__main__":
    main()
