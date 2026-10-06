from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mcdc.constant import (
    PARTICLE_ANY,
    PARTICLE_NEUTRON,
    SCORE_CURRENT_IN,
    TALLY_SURFACE_CROSSING,
)
from mcdc.coupling.geant4_config import Geant4HandoffConfig, distribution_sources
from mcdc.coupling.particles import PARTICLE_PDG, PARTICLE_TYPE_BY_NAME


def build_source_distribution_payload(
    simulation: np.ndarray,
    data: np.ndarray,
    cfg: Geant4HandoffConfig,
) -> dict[str, Any] | None:
    # one current-in source block per particle species
    blocks = []
    box_bounds_mm = None
    for source in distribution_sources(cfg):
        block, block_box_mm = _species_source_block(simulation, data, source)
        if box_bounds_mm is None:
            box_bounds_mm = block_box_mm
        elif not np.allclose(block_box_mm, box_bounds_mm):
            raise RuntimeError(
                "Distribution source tallies of one region must share the same "
                "surface_mesh box cell."
            )
        if block is not None:
            blocks.append(block)

    if not blocks:
        return None

    return {
        "box_bounds_mm": box_bounds_mm,
        "sources": blocks,
        "source_size": sum(int(block["n_events"]) for block in blocks),
        "source_total_weight": sum(float(block["total_weight"]) for block in blocks),
        "source_tally_name": ",".join(str(block["tally_name"]) for block in blocks),
    }


def _species_source_block(
    simulation: np.ndarray,
    data: np.ndarray,
    source: dict[str, Any],
) -> tuple[dict[str, Any] | None, np.ndarray]:
    tally_name = source["tally"]
    n_events = int(source["n_events"])

    # validate distribution source settings
    if n_events <= 0:
        if source["particle"] is None:
            raise RuntimeError(
                "Distribution source mode requires n_geant4_particles > 0."
            )
        raise RuntimeError(
            f"Distribution source for {source['particle']} requires n_events > 0."
        )
    if not tally_name:
        raise RuntimeError("Distribution source mode requires source_tally_name.")

    # find configured source tally
    tally = None
    for candidate in simulation["tallies"]:
        if str(candidate["name"]) == tally_name:
            tally = candidate
            break
    if tally is None:
        raise RuntimeError(f"Could not find source tally named '{tally_name}'.")

    if not bool(tally["filter_direction"]):
        raise RuntimeError("Distribution source tally must define mu/azi filters.")
    if not bool(tally["filter_energy"]):
        raise RuntimeError("Distribution source tally must define energy bins.")
    # The bridge samples mu about +z and azimuth from +x, the default reference.
    if not np.allclose(np.asarray(tally["polar_reference"]), [0.0, 0.0, 1.0]):
        raise RuntimeError(
            "Distribution source tally must use the default polar_reference (0, 0, 1)."
        )
    if int(tally["sub_type"]) != TALLY_SURFACE_CROSSING:
        raise RuntimeError("Distribution source tally must be a surface-mesh tally.")

    surface_tally = simulation["surface_crossing_tallies"][int(tally["sub_ID"])]
    if not bool(surface_tally["use_surface_mesh"]):
        raise RuntimeError("Distribution source tally must define surface_mesh.")

    # resolve the species: explicit entries must match the tally particle filter;
    # the shorthand takes the tally filter, with unfiltered tallies as neutrons
    tally_particle_type = int(tally["particle_type"])
    if source["particle"] is None:
        particle_type = (
            PARTICLE_NEUTRON
            if tally_particle_type == PARTICLE_ANY
            else tally_particle_type
        )
    else:
        particle_type = PARTICLE_TYPE_BY_NAME[source["particle"]]
        if tally_particle_type != particle_type:
            raise RuntimeError(
                f"Distribution source tally '{tally_name}' must set "
                f"particle_type='{source['particle']}'."
            )
    if particle_type not in PARTICLE_PDG:
        raise RuntimeError(
            f"Distribution source tally '{tally_name}' scores an unsupported "
            f"particle type {particle_type}."
        )

    # load score ids from the flat data array
    scores_offset = int(tally["scores_offset"])
    scores_length = int(tally["scores_length"])
    scores = data[scores_offset : scores_offset + scores_length].astype(int)
    current_in_matches = np.where(scores == SCORE_CURRENT_IN)[0]
    if len(current_in_matches) != 1:
        raise RuntimeError(
            "Distribution source tally must include exactly one current-in score."
        )
    current_in_idx = int(current_in_matches[0])

    # load angular and energy bin edges
    mu_offset = int(tally["mu_offset"])
    mu_length = int(tally["mu_length"])
    mu_edges = data[mu_offset : mu_offset + mu_length].astype(np.float64)

    azi_offset = int(tally["azi_offset"])
    azi_length = int(tally["azi_length"])
    azi_edges = data[azi_offset : azi_offset + azi_length].astype(np.float64)

    energy_offset = int(tally["energy_offset"])
    energy_length = int(tally["energy_length"])
    energy_edges_ev = data[energy_offset : energy_offset + energy_length].astype(
        np.float64
    )
    if len(mu_edges) < 2 or len(azi_edges) < 2 or len(energy_edges_ev) < 2:
        raise RuntimeError(
            "Distribution source tally grids must each have at least one bin."
        )

    # reshape tally means back to tally bin shape
    shape_offset = int(tally["bin_shape_offset"])
    shape_length = int(tally["bin_shape_length"])
    shape = tuple(int(x) for x in data[shape_offset : shape_offset + shape_length])

    mean_offset = int(tally["bin_mean_offset"])
    mean_length = int(tally["bin_length"])
    mean = data[mean_offset : mean_offset + mean_length].reshape(shape)
    current_in_mean = np.take(mean, current_in_idx, axis=-1)
    if current_in_mean.ndim != 7:
        raise RuntimeError(
            "Distribution source tally must have mu/azi/energy/time/face/u/v axes."
        )

    if current_in_mean.shape[3] > 1:
        warnings.warn(
            "Distribution source mode collapses tally time bins.",
            RuntimeWarning,
            stacklevel=3,
        )

    Nu = int(surface_tally["surface_mesh_Nu"])
    Nv = int(surface_tally["surface_mesh_Nv"])
    if current_in_mean.shape[4:] != (6, Nu, Nv):
        raise RuntimeError(
            "Distribution source tally surface_mesh shape must be face/u/v."
        )

    # convert the global mcdc source box to the local geant4 detector frame
    box_bounds_cm = np.asarray(
        [
            [
                surface_tally["surface_mesh_x_min"],
                surface_tally["surface_mesh_x_max"],
            ],
            [
                surface_tally["surface_mesh_y_min"],
                surface_tally["surface_mesh_y_max"],
            ],
            [
                surface_tally["surface_mesh_z_min"],
                surface_tally["surface_mesh_z_max"],
            ],
        ],
        dtype=np.float64,
    )
    box_center_cm = np.mean(box_bounds_cm, axis=1, keepdims=True)
    box_bounds_mm = (box_bounds_cm - box_center_cm) * 10.0

    # collapse time only. The bridge decodes this C-order flat array as
    # mu, azi, energy, face, u, v, with v fastest.
    weights = np.sum(current_in_mean, axis=3)
    weights = np.maximum(weights, 0.0)

    # The tally mean is normalized by the global source count, not the local
    # MPI work size. Multiply it back so weights carry the absolute integrated
    # current entering the volume, matching the raw weights used in bank mode.
    N_particle = int(simulation["settings"]["N_particle"])
    if N_particle <= 0:
        raise RuntimeError("Distribution source mode requires global N_particle > 0.")
    weights = weights * N_particle
    total = float(np.sum(weights))
    if total <= 0.0:
        return None, box_bounds_mm

    block = {
        "pdg": int(PARTICLE_PDG[particle_type]),
        "tally_name": tally_name,
        "mu_edges": mu_edges,
        "azi_edges": azi_edges,
        "energy_edges_mev": energy_edges_ev * 1.0e-6,
        "weights": np.ascontiguousarray(weights.ravel()),
        "Nu": Nu,
        "Nv": Nv,
        "n_events": n_events,
        "total_weight": total,
    }
    return block, box_bounds_mm
