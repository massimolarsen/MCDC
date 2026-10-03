import os

import numpy as np
import pytest
from scipy import stats

import mcdc
import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_

DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../regression/mcdc-regression_test_data"
    )
)


# Statistical sampler-vs-kernel tests are too slow in pure-Python mode
numba_only = pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1",
    reason="slow statistical test; runs in numba mode",
)


def requires_nuclide(name):
    return pytest.mark.skipif(
        not os.path.exists(os.path.join(DATA_DIR, f"{name}-293.6K.h5")),
        reason="regression nuclear data not available",
    )


@pytest.fixture
def nuclide_simulation(prepare_simulation, monkeypatch):
    """Prepared (simulation, data, nuclide) for a single-nuclide infinite medium."""

    def _make(name):
        monkeypatch.setenv("MCDC_LIB", DATA_DIR)
        material = mcdc.Material(nuclide_composition={name: 0.05})
        container, data = prepare_simulation(cells=(mcdc.Cell(fill=material),))
        simulation = container[0]
        return simulation, data, simulation["nuclides"][0]

    return _make


@pytest.fixture
def tabulated_yield_simulation(prepare_simulation, monkeypatch, tmp_path):
    """U-235 whose MT-5 has a linear energy-dependent yield (0.3 to 1.3 over 0-30 MeV)."""
    import h5py
    import shutil

    path = tmp_path / "U235-293.6K.h5"
    shutil.copy(os.path.join(DATA_DIR, "U235-293.6K.h5"), path)
    with h5py.File(path, "r+") as f:
        group = f["neutron_reactions/inelastic_scattering/MT-005"]
        del group["multiplicity"]
        group.create_dataset("multiplicity", data=-1)
        table = group.create_group("multiplicity_table")
        table.attrs["type"] = "tabulated"
        table.create_dataset("energy", data=np.array([1.0e-11, 30.0]))  # MeV
        table.create_dataset("value", data=np.array([0.3, 1.3]))
    monkeypatch.setenv("MCDC_LIB", str(tmp_path))

    material = mcdc.Material(nuclide_composition={"U235": 0.05})
    container, data = prepare_simulation(cells=(mcdc.Cell(fill=material),))
    simulation = container[0]
    return simulation, data, simulation["nuclides"][0]


def inelastic_reactions(simulation, data, nuclide):
    """List of (base reaction, inelastic sub-record) pairs of a nuclide."""
    result = []
    for i in range(nuclide["N_neutron_inelastic_scattering_reaction"]):
        ID = mcdc_get.nuclide.neutron_inelastic_scattering_reaction_IDs(
            i, nuclide, data
        )
        base = simulation["neutron_reactions"][ID]
        sub = simulation["neutron_inelastic_scattering_reactions"][base["sub_ID"]]
        result.append((base, sub))
    return result


def spectrum_distribution(simulation, data, inelastic, n=0):
    ID = mcdc_get.neutron_inelastic_scattering_reaction.energy_spectrum_IDs(
        n, inelastic, data
    )
    return simulation["distributions"][ID]


def hashed_seed(i, base=20261002):
    """Per-history seed hashed like MC/DC's split_seed (sequential raw seeds give
    lattice-correlated LCG draws)."""
    import mcdc.transport.rng as rng

    return rng.split_seed(np.uint64(i), np.uint64(base))


def rng_container(seed):
    particles = np.zeros(1, type_.particle)
    particles[0]["rng_seed"] = seed
    return particles


def interior_energy(grid, idx, fraction=0.37):
    """An incident energy strictly inside [grid[idx], grid[idx+1]] (middle if None)."""
    if idx is None:
        idx = len(grid) // 2
    return grid[idx] + fraction * (grid[idx + 1] - grid[idx])


def bin_integral_1d(density, edges, N_sub=64):
    """Integral of density over each bin, Gauss-Legendre per bin."""
    x, w = np.polynomial.legendre.leggauss(N_sub)
    result = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        a, c = edges[b], edges[b + 1]
        xs = 0.5 * (c - a) * x + 0.5 * (c + a)
        result[b] = 0.5 * (c - a) * sum(wi * density(xi) for xi, wi in zip(xs, w))
    return result


def _composite_nodes(a, b, N_panel, N_point):
    """Composite Gauss-Legendre nodes/weights on [a, b] (resolves kinks/peaks)."""
    x, w = np.polynomial.legendre.leggauss(N_point)
    panels = np.linspace(a, b, N_panel + 1)
    nodes, weights = [], []
    for p0, p1 in zip(panels[:-1], panels[1:]):
        nodes.append(0.5 * (p1 - p0) * x + 0.5 * (p1 + p0))
        weights.append(0.5 * (p1 - p0) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def bin_integral_2d(density, edges_x, edges_y, N_panel_x=24, N_panel_y=4, N_point=6):
    result = np.zeros((len(edges_x) - 1, len(edges_y) - 1))
    for i in range(len(edges_x) - 1):
        xs, wx = _composite_nodes(edges_x[i], edges_x[i + 1], N_panel_x, N_point)
        for j in range(len(edges_y) - 1):
            ys, wy = _composite_nodes(edges_y[j], edges_y[j + 1], N_panel_y, N_point)
            total = 0.0
            for xi, wi in zip(xs, wx):
                for yj, wj in zip(ys, wy):
                    total += wi * wj * density(xi, yj)
            result[i, j] = total
    return result


def chi_square_pvalue(observed, probability):
    """Pearson chi-square p-value of counts against bin probabilities."""
    observed = np.asarray(observed, dtype=float).ravel()
    probability = np.asarray(probability, dtype=float).ravel()
    N = observed.sum()
    expected = N * probability
    keep = expected > 5.0
    # Lump sparse bins together
    obs = np.append(observed[keep], observed[~keep].sum())
    exp = np.append(expected[keep], expected[~keep].sum())
    if exp[-1] <= 5.0:
        obs, exp = obs[:-1], exp[:-1]
    chi2 = np.sum((obs - exp) ** 2 / exp)
    return stats.chi2.sf(chi2, len(obs) - 1)


def write_synthetic_nuclide(directory, name, A, energy, sigma_s, sigma_c):
    """
    Minimal nuclide in the MC/DC library format: elastic (isotropic COM) and capture
    with tabulated cross sections on `energy` [eV], at 0.1 K (no free-gas region in
    practice). Written as <name>-0.1K.h5.
    """
    import h5py

    energy_MeV = np.asarray(energy, dtype=float) * 1e-6
    path = os.path.join(directory, f"{name}-0.1K.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("nuclide_name", data=name)
        f.create_dataset("excitation_level", data=0)
        f.create_dataset("temperature", data=0.1).attrs["unit"] = "K"
        f.create_dataset("atomic_number", data=1)
        f.create_dataset("mass_number", data=1)
        f.create_dataset("atomic_weight_ratio", data=float(A))
        f.create_dataset("fissionable", data=False)
        reactions = f.create_group("neutron_reactions")
        reactions.create_dataset("xs_energy_grid", data=energy_MeV).attrs["unit"] = (
            "MeV"
        )

        elastic = reactions.create_group("elastic_scattering/MT-002")
        elastic.attrs["MT"] = 2
        xs = elastic.create_dataset("xs", data=np.asarray(sigma_s, dtype=float))
        xs.attrs["offset"] = 0
        elastic.create_dataset("reference_frame", data="COM")
        elastic.create_dataset("Q-value", data=0.0)
        angle = elastic.create_group("angular_cosine_distribution")
        angle.attrs["type"] = "tabulated"
        angle.create_dataset("energy", data=energy_MeV[[0, -1]])
        angle.create_dataset("offset", data=np.array([0, 2]))
        angle.create_dataset("value", data=np.array([-1.0, 1.0, -1.0, 1.0]))
        angle.create_dataset("pdf", data=np.array([0.5, 0.5, 0.5, 0.5]))

        capture = reactions.create_group("capture/MT-102")
        capture.attrs["MT"] = 102
        xs = capture.create_dataset("xs", data=np.asarray(sigma_c, dtype=float))
        xs.attrs["offset"] = 0
        capture.create_dataset("reference_frame", data="LAB")
        capture.create_dataset("Q-value", data=0.0)
    return path
