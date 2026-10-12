"""Run metadata for the CubeSat result plots: history counts, source cube, boxes.

Everything is read from the run directory (its archived input, payloads, and
outputs), so the plots work for any run without hard-coded numbers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import h5py
import numpy as np

REGIONS = ("obc", "eps", "adcs", "comms")
FACES = ("x-", "x+", "y-", "y+", "z- (down)", "z+ (up)")


def run_input(run_dir: Path) -> str:
    inputs = sorted(Path(run_dir).glob("inputCE_*.py"))
    if not inputs:
        raise FileNotFoundError(f"No archived inputCE_*.py in {run_dir}")
    return inputs[0].read_text()


def n_particle(run_dir: Path) -> int:
    output = next(Path(run_dir, "mcdc_h5").glob("*.h5"))
    with h5py.File(output, "r") as file:
        return int(file["settings/N_particle"][()])


def species_spectra(run_dir: Path) -> dict[str, Path]:
    """Source spectrum file per species, from the input's SOURCE_SPECTRA."""
    text = run_input(run_dir)
    pairs = re.findall(r'"(\w+)":\s*EXAMPLE_DIR\s*/\s*"([^"]+)"', text)
    return {particle: Path(run_dir, name) for particle, name in pairs}


def species_histories(run_dir: Path) -> dict[str, float]:
    """MC/DC histories per source species (SPECIES_HISTORIES, else an even split)."""
    text = run_input(run_dir)
    match = re.search(r"SPECIES_HISTORIES\s*=\s*(\{[^}]*\})", text)
    if match:
        return {k: float(v) for k, v in ast.literal_eval(match.group(1)).items()}
    species = list(species_spectra(run_dir)) or ["neutron"]
    return {s: n_particle(run_dir) / len(species) for s in species}


def source_cube_area_cm2(run_dir: Path) -> float:
    half_width = float(re.search(r"boundary_half_width\s*=\s*([0-9.]+)", run_input(run_dir)).group(1))
    return 6.0 * (2.0 * half_width) ** 2


def box_dims_cm(run_dir: Path, region: str) -> np.ndarray:
    with h5py.File(Path(run_dir, "geant4_payloads", f"{region}_payload.h5"), "r") as file:
        bounds = file["box_bounds_mm"][()]
    return (bounds[:, 1] - bounds[:, 0]) / 10.0


def face_areas_cm2(dims: np.ndarray) -> np.ndarray:
    dx, dy, dz = dims
    return np.array([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy])


def spectrum_bin_probability(spectrum: Path, edges_ev: np.ndarray) -> np.ndarray:
    """Probability of each tally energy bin under a source pdf (eV^-1)."""
    energy, pdf = np.loadtxt(spectrum, delimiter=",", skiprows=1, unpack=True)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(energy))])
    cdf /= cdf[-1]
    return np.diff(np.interp(edges_ev, energy, cdf, left=0.0, right=1.0))


def text_array(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]
