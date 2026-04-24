from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


EV_TO_MEV = 1.0e-6


def export_static_source(
    output_path: str | Path = "output.h5",
    destination: str | Path = ".",
    energy_tally: str = "handoff_energy",
    angle_tally: str = "handoff_angle",
    energy_score: str | None = None,
    angle_score: str | None = None,
    metadata: dict | None = None,
):
    output_path = Path(output_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    metadata = {} if metadata is None else dict(metadata)
    group_edges_mev = metadata.get("energy_bin_edges_mev", metadata.get("energy_group_edges_mev"))
    if group_edges_mev is None:
        raise ValueError("`metadata['energy_bin_edges_mev']` is required.")

    with h5py.File(output_path, "r") as h5_file:
        energy_score = _resolve_score_name(h5_file, energy_tally, energy_score)
        angle_score = _resolve_score_name(h5_file, angle_tally, angle_score)
        energy_group = h5_file[f"tallies/{energy_tally}/{energy_score}/mean"][()]
        angle_group = h5_file[f"tallies/{angle_tally}/{angle_score}/mean"][()]
        mu_grid = h5_file[f"tallies/{angle_tally}/grid/mu"][()]
        azi_grid = h5_file[f"tallies/{angle_tally}/grid/azi"][()] if f"tallies/{angle_tally}/grid/azi" in h5_file else None

    energy_weights = _flatten_weights(energy_group)
    theta_grid = np.arccos(np.clip(_bin_centers(mu_grid), -1.0, 1.0))
    theta_weights, phi_grid, phi_weights = _extract_angular_distributions(angle_group, mu_grid, azi_grid)

    spectrum_path = destination / "spectrum.mac"
    angle_path = destination / "angle_spectrum.mac"
    manifest_path = destination / "manifest.json"

    _write_energy_macro(spectrum_path, np.asarray(group_edges_mev, dtype=float), energy_weights)
    _write_angle_macro(angle_path, theta_grid, theta_weights, phi_grid=phi_grid, phi_weights=phi_weights)

    manifest = {
        "output_h5": str(output_path.resolve()),
        "energy_tally": energy_tally,
        "angle_tally": angle_tally,
        "energy_score": energy_score,
        "angle_score": angle_score,
        "energy_bin_edges_mev": list(np.asarray(group_edges_mev, dtype=float)),
        "energy_bin_centers_mev": list(_bin_centers(np.asarray(group_edges_mev, dtype=float))),
        "mu_bin_edges": list(np.asarray(mu_grid, dtype=float)),
        "theta_bin_centers_rad": list(theta_grid),
        "azi_bin_edges_rad": None if azi_grid is None else list(np.asarray(azi_grid, dtype=float)),
        "phi_bin_centers_rad": None if phi_grid is None else list(np.asarray(phi_grid, dtype=float)),
        **metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return {
        "destination": str(destination),
        "spectrum": str(spectrum_path),
        "angle": str(angle_path),
        "manifest": str(manifest_path),
    }


def _resolve_score_name(h5_file: h5py.File, tally_name: str, requested_score: str | None):
    tally_group = h5_file[f"tallies/{tally_name}"]
    if requested_score is not None:
        if requested_score not in tally_group:
            available = sorted(name for name in tally_group.keys() if name != "grid")
            raise ValueError(
                f"Tally {tally_name!r} does not contain score {requested_score!r}. "
                f"Available scores: {available}"
            )
        return requested_score

    preferred_scores = ("flux", "net-current", "density", "collision", "capture", "fission")
    for score_name in preferred_scores:
        if score_name in tally_group:
            return score_name

    available = sorted(name for name in tally_group.keys() if name != "grid")
    raise ValueError(f"Could not resolve a score dataset for tally {tally_name!r}. Available scores: {available}")


def _flatten_weights(values):
    weights = np.asarray(values, dtype=float)
    weights = np.squeeze(weights)
    if weights.ndim != 1:
        raise ValueError(f"Expected a 1D tally array after squeeze, got shape {weights.shape}")
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if np.all(weights == 0.0):
        raise ValueError("All tally weights are zero.")
    return weights / weights.sum()


def _bin_centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _extract_angular_distributions(values, mu_edges, azi_edges):
    weights = np.asarray(values, dtype=float)
    weights = np.squeeze(weights)
    weights = np.clip(weights, a_min=0.0, a_max=None)

    if weights.ndim == 1:
        if np.all(weights == 0.0):
            raise ValueError("All angular tally weights are zero.")
        return weights / weights.sum(), None, None

    if weights.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D angular tally array after squeeze, got shape {weights.shape}")

    if azi_edges is None:
        raise ValueError("A 2D angular tally requires an azimuth grid.")

    theta_weights = weights.sum(axis=1)
    phi_weights = weights.sum(axis=0)
    if np.all(theta_weights == 0.0) or np.all(phi_weights == 0.0):
        raise ValueError("Angular tally collapsed to zero in theta or phi.")

    phi_grid = _bin_centers(azi_edges)
    # Geant4 GPS user-defined phi histograms expect nonnegative angular positions.
    phi_grid = np.mod(phi_grid, 2.0 * np.pi)
    order = np.argsort(phi_grid)
    phi_grid = phi_grid[order]
    phi_weights = phi_weights[order]
    return theta_weights / theta_weights.sum(), phi_grid, phi_weights / phi_weights.sum()


def _write_energy_macro(output_path: Path, group_edges_mev: np.ndarray, weights: np.ndarray):
    centers_mev = _bin_centers(group_edges_mev)
    with output_path.open("w", encoding="utf-8") as mac:
        mac.write("# Automatically generated from MC/DC static handoff tally\n")
        mac.write("/gps/ene/type User\n")
        mac.write("/gps/hist/type energy\n")
        for energy_mev, weight in zip(centers_mev, weights):
            mac.write(f"/gps/hist/point {energy_mev:.8e} {weight:.8e}\n")


def _write_angle_macro(
    output_path: Path,
    theta_grid: np.ndarray,
    weights: np.ndarray,
    phi_grid: np.ndarray | None = None,
    phi_weights: np.ndarray | None = None,
):
    with output_path.open("w", encoding="utf-8") as mac:
        mac.write("# Automatically generated from MC/DC static handoff tally\n")
        mac.write("/gps/ang/type user\n")
        mac.write("/gps/hist/type theta\n")
        for theta, weight in zip(theta_grid, weights):
            mac.write(f"/gps/hist/point {theta:.8e} {weight:.8e}\n")
        if phi_grid is not None and phi_weights is not None:
            mac.write("/gps/hist/type phi\n")
            for phi, weight in zip(phi_grid, phi_weights):
                mac.write(f"/gps/hist/point {phi:.8e} {weight:.8e}\n")
