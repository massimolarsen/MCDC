from __future__ import annotations

from typing import Any

import numpy as np

from mcdc.constant import SCORE_CURRENT_IN
from mcdc.coupling import geant4_config
from mcdc.coupling.geant4_config import Geant4HandoffConfig


def validate_distribution_config(cfg: Geant4HandoffConfig) -> None:
    if cfg.n_geant4_particles <= 0:
        raise RuntimeError("Distribution source mode requires n_geant4_particles > 0.")
    if not cfg.source_tally_name:
        raise RuntimeError("Distribution source mode requires source_tally_name.")
    if cfg.distribution_box_cm is None:
        raise RuntimeError("Distribution source mode requires distribution_box_cm.")
    if len(cfg.distribution_box_cm) != 3:
        raise RuntimeError("distribution_box_cm must contain x, y, and z bounds.")
    for axis, bounds in zip(("x", "y", "z"), cfg.distribution_box_cm):
        if len(bounds) != 2:
            raise RuntimeError(
                f"distribution_box_cm {axis} bounds must have two values."
            )
        lower = float(bounds[0])
        upper = float(bounds[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise RuntimeError(
                f"distribution_box_cm {axis} bounds must be finite with min < max."
            )


def build_source_distribution_payload(
    simulation: np.ndarray,
    data: np.ndarray,
    cfg: Geant4HandoffConfig,
) -> dict[str, Any]:
    validate_distribution_config(cfg)
    tally = None
    for candidate in simulation["tallies"]:
        if str(candidate["name"]) == cfg.source_tally_name:
            tally = candidate
            break
    if tally is None:
        raise RuntimeError(
            f"Could not find source tally named '{cfg.source_tally_name}'."
        )

    if not bool(tally["filter_direction"]):
        raise RuntimeError("Distribution source tally must define mu/azi filters.")
    if not bool(tally["filter_energy"]):
        raise RuntimeError("Distribution source tally must define energy bins.")

    # Tally grids and scores are stored in MCDC's flat data array; the tally record
    # carries the offsets needed to rebuild the pieces Geant4 samples from.
    scores_offset = int(tally["scores_offset"])
    scores_length = int(tally["scores_length"])
    scores = data[scores_offset : scores_offset + scores_length].astype(int)
    current_in_matches = np.where(scores == SCORE_CURRENT_IN)[0]
    if len(current_in_matches) != 1:
        raise RuntimeError(
            "Distribution source tally must include exactly one current-in score."
        )
    current_in_idx = int(current_in_matches[0])

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

    shape_offset = int(tally["bin_shape_offset"])
    shape_length = int(tally["bin_shape_length"])
    shape = tuple(int(x) for x in data[shape_offset : shape_offset + shape_length])

    mean_offset = int(tally["bin_sum_offset"])
    mean_length = int(tally["bin_length"])
    mean = data[mean_offset : mean_offset + mean_length].reshape(shape)
    current_in_mean = np.take(mean, current_in_idx, axis=-1)
    if current_in_mean.ndim < 4:
        raise RuntimeError(
            "Distribution source tally must have mu/azi/energy/time axes."
        )

    # V1 samples only the joint mu/azi/energy distribution. Any extra tally axes
    # such as time are collapsed before passing weights to the bridge.
    weights = np.sum(current_in_mean, axis=tuple(range(3, current_in_mean.ndim)))
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if not total > 0.0:
        raise RuntimeError(
            "Distribution source tally has zero total current-in weight."
        )

    box_bounds_mm = np.asarray(cfg.distribution_box_cm, dtype=np.float64) * 10.0
    return {
        "box_bounds_mm": box_bounds_mm,
        "mu_edges": mu_edges,
        "azi_edges": azi_edges,
        "energy_edges_mev": energy_edges_ev * 1.0e-6,
        "weights": np.ascontiguousarray(weights.ravel()),
        "n_events": int(cfg.n_geant4_particles),
        "total_weight": total,
        "tally_name": cfg.source_tally_name,
    }


def run_distribution_handoff(
    simulation: np.ndarray,
    data: np.ndarray,
    cfg: Geant4HandoffConfig,
) -> dict[str, Any]:
    payload = build_source_distribution_payload(simulation, data, cfg)

    # session is the geant4 bridge object
    session = geant4_config.create_session()
    try:
        session.load_source_distribution(
            payload["box_bounds_mm"],
            payload["mu_edges"],
            payload["azi_edges"],
            payload["energy_edges_mev"],
            payload["weights"],
            payload["n_events"],
        )
        session.beam_on()
        results = session.get_results()
        summary = {
            "handoff_bank_size": int(simulation["bank_handoff"]["size"][0]),
            "loaded_primaries": int(results.loaded_primaries),
            "events_run": int(results.last_events_run),
            "status": str(results.status),
            "source_mode": "distribution",
            "source_tally_name": payload["tally_name"],
            "source_total_weight": payload["total_weight"],
            "primary_summary": geant4_config.primary_summary(results),
        }
    finally:
        session.close()

    if summary["events_run"] != payload["n_events"]:
        raise RuntimeError(
            "Geant4 distribution coupling mismatch: events_run does not match n_geant4_particles."
        )

    return summary
