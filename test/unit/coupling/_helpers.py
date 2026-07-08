import numpy as np

from mcdc.constant import SCORE_CURRENT_IN, TALLY_SURFACE_CROSSING
from mcdc.coupling.geant4_config import Geant4HandoffConfig
from mcdc.numba_types import surface_crossing_tally as SURFACE_CROSSING_TALLY_DTYPE
from mcdc.numba_types import tally as TALLY_DTYPE


PARTICLE_DTYPE = np.dtype(
    [
        ("particle_type", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("ux", np.float64),
        ("uy", np.float64),
        ("uz", np.float64),
        ("E", np.float64),
        ("w", np.float64),
        ("t", np.float64),
    ]
)


def distribution_simulation_and_data(
    *,
    tally_name="source_tally",
    scores=None,
    filter_direction=True,
    filter_energy=True,
    weights=None,
    n_particle=1,
    surface_mesh=True,
    child_type=TALLY_SURFACE_CROSSING,
    time_bins=1,
):
    if scores is None:
        scores = [99, SCORE_CURRENT_IN]
    if weights is None:
        weights = np.arange(2 * 2 * 2 * 6 * 2 * 3, dtype=np.float64).reshape(
            2, 2, 2, 6, 2, 3
        ) + 1.0

    scores = np.asarray(scores, dtype=np.float64)
    mu = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    azi = np.asarray([-np.pi, 0.0, np.pi], dtype=np.float64)
    energy = np.asarray([0.0, 1.0e6, 2.0e6], dtype=np.float64)
    shape = np.asarray([2, 2, 2, time_bins, 6, 2, 3, len(scores)], dtype=np.float64)
    mean = np.zeros(tuple(shape.astype(int)), dtype=np.float64)
    if SCORE_CURRENT_IN in scores:
        current_idx = int(np.where(scores == SCORE_CURRENT_IN)[0][0])
        for i_time in range(time_bins):
            mean[:, :, :, i_time, :, :, :, current_idx] = weights

    offsets = {}
    chunks = []
    for name, values in (
        ("scores", scores),
        ("mu", mu),
        ("azi", azi),
        ("energy", energy),
        ("bin_shape", shape),
        ("bin_sum", mean.ravel()),
    ):
        offsets[name] = sum(len(chunk) for chunk in chunks)
        chunks.append(np.asarray(values, dtype=np.float64).ravel())
    data = np.concatenate(chunks)

    tally_values = {
        "name": tally_name,
        "filter_direction": filter_direction,
        "filter_energy": filter_energy,
        "scores_offset": offsets["scores"],
        "scores_length": len(scores),
        "mu_offset": offsets["mu"],
        "mu_length": len(mu),
        "azi_offset": offsets["azi"],
        "azi_length": len(azi),
        "energy_offset": offsets["energy"],
        "energy_length": len(energy),
        "bin_shape_offset": offsets["bin_shape"],
        "bin_shape_length": len(shape),
        "bin_sum_offset": offsets["bin_sum"],
        "bin_length": mean.size,
        "child_type": child_type,
        "child_ID": 0,
    }
    tallies = np.zeros(1, dtype=TALLY_DTYPE)
    for field, value in tally_values.items():
        tallies[0][field] = value

    surface_crossing_tallies = np.zeros(1, dtype=SURFACE_CROSSING_TALLY_DTYPE)
    surface_crossing_tallies[0]["use_surface_mesh"] = surface_mesh
    surface_crossing_tallies[0]["surface_mesh_Nu"] = 2
    surface_crossing_tallies[0]["surface_mesh_Nv"] = 3
    surface_crossing_tallies[0]["surface_mesh_x_min"] = -1.0
    surface_crossing_tallies[0]["surface_mesh_x_max"] = 1.0
    surface_crossing_tallies[0]["surface_mesh_y_min"] = -2.0
    surface_crossing_tallies[0]["surface_mesh_y_max"] = 2.0
    surface_crossing_tallies[0]["surface_mesh_z_min"] = -3.0
    surface_crossing_tallies[0]["surface_mesh_z_max"] = 3.0

    simulation = {
        "mpi_size": 1,
        "settings": {"N_particle": n_particle},
        "tallies": tallies,
        "surface_crossing_tallies": surface_crossing_tallies,
        "bank_handoff": {
            "size": np.asarray([0], dtype=np.int64),
            "particle_data": np.array([], dtype=PARTICLE_DTYPE),
        },
    }
    return simulation, data, weights


def distribution_config(**kwargs):
    values = {
        "name": "source_region",
        "source_mode": "distribution",
        "bridge_build_dir": "bridge_build",
        "n_geant4_particles": 12,
        "source_tally_name": "source_tally",
    }
    values.update(kwargs)
    return Geant4HandoffConfig(**values)
