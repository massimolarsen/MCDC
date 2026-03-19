from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SUPPORTED_SCORES = {
    "flux",
    "density",
    "collision",
    "capture",
    "fission",
    "net-current",
}

GRID_AXIS_ORDER = ("mu", "azi", "energy", "time", "x", "y", "z")
GPS_ENERGY_SCORE_PRIORITY = {
    "net-current": 100,
    "flux": 80,
    "density": 60,
    "collision": 30,
    "capture": 20,
    "fission": 10,
}
GPS_ANGLE_SCORE_PRIORITY = {
    "flux": 100,
    "density": 70,
    "net-current": 50,
    "collision": 25,
    "capture": 15,
    "fission": 10,
}
CONVERSION_BONUS = {
    "energy": 40,
    "angle": 20,
}


@dataclass
class ScoreDescriptor:
    name: str
    mean_path: str
    sdev_path: str | None
    shape: list[int]


@dataclass
class TallyDescriptor:
    name: str
    group_path: str
    tally_class: str
    grids: dict[str, list[float]]
    scores: dict[str, ScoreDescriptor]
    warnings: list[str] = field(default_factory=list)

    @property
    def nondegenerate_axes(self) -> list[str]:
        axes: list[str] = []
        for axis in GRID_AXIS_ORDER:
            grid = self.grids.get(axis)
            if grid is None:
                continue
            if len(grid) - 1 > 1:
                axes.append(axis)
        return axes

    @property
    def available_axes(self) -> list[str]:
        axes: list[str] = []
        for axis in GRID_AXIS_ORDER:
            grid = self.grids.get(axis)
            if grid is None:
                continue
            if len(grid) >= 2:
                axes.append(axis)
        return axes


@dataclass
class ExportCandidate:
    id: str
    tally_name: str
    score_name: str
    conversion_type: str
    source_type: str
    confidence: int
    warnings: list[str]
    axes_present: list[str]
    axes_reduced: list[str]
    varied_axes: list[str]
    export_dir_name: str


def _sanitize_name(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unnamed"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_hdf5_tallies(hdf_path: str | Path) -> list[TallyDescriptor]:
    hdf_path = Path(hdf_path)
    descriptors: list[TallyDescriptor] = []

    with h5py.File(hdf_path, "r") as h5:
        tallies_group = h5.get("tallies")
        if tallies_group is None:
            return descriptors

        for tally_name in sorted(tallies_group.keys()):
            tally_group = tallies_group[tally_name]
            if not isinstance(tally_group, h5py.Group):
                continue

            grids: dict[str, list[float]] = {}
            scores: dict[str, ScoreDescriptor] = {}
            warnings: list[str] = []

            grid_group = tally_group.get("grid")
            if isinstance(grid_group, h5py.Group):
                for axis in GRID_AXIS_ORDER:
                    if axis in grid_group:
                        grids[axis] = np.asarray(grid_group[axis][()]).tolist()

            for key in sorted(tally_group.keys()):
                if key == "grid":
                    continue
                child = tally_group[key]
                if not isinstance(child, h5py.Group):
                    continue
                if key not in SUPPORTED_SCORES:
                    warnings.append(
                        f"Score group '{key}' is not part of the known MCDC tally score set."
                    )
                mean_ds = child.get("mean")
                if mean_ds is None:
                    warnings.append(f"Score group '{key}' is missing a mean dataset.")
                    continue
                sdev_ds = child.get("sdev")
                scores[key] = ScoreDescriptor(
                    name=key,
                    mean_path=mean_ds.name,
                    sdev_path=sdev_ds.name if sdev_ds is not None else None,
                    shape=list(mean_ds.shape),
                )

            tally_class = _infer_tally_class(grids, scores)
            descriptors.append(
                TallyDescriptor(
                    name=tally_name,
                    group_path=tally_group.name,
                    tally_class=tally_class,
                    grids=grids,
                    scores=scores,
                    warnings=warnings,
                )
            )

    return descriptors


def _infer_tally_class(
    grids: dict[str, list[float]], scores: dict[str, ScoreDescriptor]
) -> str:
    if any(axis in grids for axis in ("x", "y", "z")):
        return "mesh"
    if "net-current" in scores:
        return "surface-like"
    if any(axis in grids for axis in ("mu", "azi", "energy", "time")):
        return "phase-space"
    return "unknown"


def classify_candidates(tallies: list[TallyDescriptor]) -> tuple[list[ExportCandidate], list[dict[str, Any]]]:
    candidates: list[ExportCandidate] = []
    unsupported: list[dict[str, Any]] = []

    for tally in tallies:
        tally_supported = False
        axes_present = tally.available_axes
        varied_axes = tally.nondegenerate_axes

        for score_name in sorted(tally.scores.keys()):
            energy_candidate = _maybe_build_energy_candidate(
                tally, score_name, axes_present, varied_axes
            )
            if energy_candidate is not None:
                candidates.append(energy_candidate)
                tally_supported = True

            angle_candidate = _maybe_build_angle_candidate(
                tally, score_name, axes_present, varied_axes
            )
            if angle_candidate is not None:
                candidates.append(angle_candidate)
                tally_supported = True

        if not tally_supported:
            reasons: list[str] = []
            if "energy" not in tally.grids and "mu" not in tally.grids:
                reasons.append(
                    "No exportable source dimensions found: expected energy or mu grids."
                )
            if not tally.scores:
                reasons.append("No score groups with mean datasets were found.")
            if not reasons:
                reasons.append(
                    "Tally dimensions are present, but no supported source interpretation was considered safe."
                )
            unsupported.append(
                {
                    "tally_name": tally.name,
                    "tally_class": tally.tally_class,
                    "reasons": reasons + tally.warnings,
                }
            )

    candidates.sort(key=lambda item: (-item.confidence, item.id))
    return candidates, unsupported


def _maybe_build_energy_candidate(
    tally: TallyDescriptor,
    score_name: str,
    axes_present: list[str],
    varied_axes: list[str],
) -> ExportCandidate | None:
    if "energy" not in tally.grids:
        return None

    warnings: list[str] = []
    confidence = GPS_ENERGY_SCORE_PRIORITY.get(score_name, 0)
    if confidence <= 0:
        return None
    if len(tally.grids["energy"]) - 1 <= 1:
        return None

    if score_name not in {"net-current", "flux", "density"}:
        warnings.append(
            f"Energy candidate uses score '{score_name}', which is not a default source observable."
        )
        confidence -= 20
    confidence += CONVERSION_BONUS["energy"]

    axes_reduced = [axis for axis in varied_axes if axis != "energy"]
    if axes_reduced:
        warnings.append(
            "Extra dimensions were marginalized to form an energy spectrum."
        )

    export_name = f"{_sanitize_name(tally.name)}__{_sanitize_name(score_name)}__energy"
    return ExportCandidate(
        id=export_name,
        tally_name=tally.name,
        score_name=score_name,
        conversion_type="energy",
        source_type="gps-energy",
        confidence=max(confidence, 1),
        warnings=warnings,
        axes_present=axes_present,
        axes_reduced=axes_reduced,
        varied_axes=varied_axes,
        export_dir_name=export_name,
    )


def _maybe_build_angle_candidate(
    tally: TallyDescriptor,
    score_name: str,
    axes_present: list[str],
    varied_axes: list[str],
) -> ExportCandidate | None:
    if "mu" not in tally.grids:
        return None

    warnings: list[str] = []
    confidence = GPS_ANGLE_SCORE_PRIORITY.get(score_name, 0)
    if confidence <= 0:
        return None
    if len(tally.grids["mu"]) - 1 <= 1:
        return None

    if score_name not in {"flux", "density"}:
        warnings.append(
            f"Angular candidate uses score '{score_name}', which may not represent a directional source distribution cleanly."
        )
        confidence -= 20
    confidence += CONVERSION_BONUS["angle"]

    axes_reduced = [axis for axis in varied_axes if axis != "mu"]
    if "azi" in tally.grids and len(tally.grids["azi"]) - 1 > 1:
        warnings.append(
            "Azimuthal information is present and was marginalized because Geant4 GPS theta histograms do not preserve it directly."
        )
    if axes_reduced:
        warnings.append(
            "Extra dimensions were marginalized to form a polar-angle spectrum."
        )

    export_name = f"{_sanitize_name(tally.name)}__{_sanitize_name(score_name)}__angle"
    return ExportCandidate(
        id=export_name,
        tally_name=tally.name,
        score_name=score_name,
        conversion_type="angle",
        source_type="gps-angle",
        confidence=max(confidence, 1),
        warnings=warnings,
        axes_present=axes_present,
        axes_reduced=axes_reduced,
        varied_axes=varied_axes,
        export_dir_name=export_name,
    )


def _load_score_array(
    h5: h5py.File, tally: TallyDescriptor, score_name: str
) -> tuple[np.ndarray, list[str]]:
    array = np.asarray(h5[tally.scores[score_name].mean_path][()])
    axes = []
    for axis in GRID_AXIS_ORDER:
        grid = tally.grids.get(axis)
        if grid is None:
            continue
        nbin = len(grid) - 1
        if nbin > 1:
            axes.append(axis)
    if array.ndim != len(axes):
        raise ValueError(
            f"Unable to map score array for tally '{tally.name}' score '{score_name}': "
            f"array has {array.ndim} dimensions but inferred {len(axes)} varying axes {axes}."
        )
    return array, axes


def _sum_except(
    array: np.ndarray, axes: list[str], keep_axes: set[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    reduce_indices = [i for i, axis in enumerate(axes) if axis not in keep_axes]
    reduced_axes = [axes[i] for i in reduce_indices]
    if reduce_indices:
        reduced = array.sum(axis=tuple(reduce_indices))
    else:
        reduced = array
    kept_axes = [axis for axis in axes if axis in keep_axes]
    return np.asarray(reduced), kept_axes, reduced_axes


def export_candidates(
    hdf_path: str | Path,
    outdir: str | Path,
    candidates: list[ExportCandidate],
    tallies: list[TallyDescriptor],
    particle: str = "neutron",
    candidate_filter: str | None = None,
    best_only: bool = False,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tally_map = {tally.name: tally for tally in tallies}
    selected = candidates
    if candidate_filter is not None:
        selected = [candidate for candidate in selected if candidate.id == candidate_filter]
    if best_only and selected:
        selected = [selected[0]]

    exports_summary: list[dict[str, Any]] = []
    with h5py.File(hdf_path, "r") as h5:
        for candidate in selected:
            tally = tally_map[candidate.tally_name]
            export_dir = outdir / candidate.export_dir_name
            export_dir.mkdir(parents=True, exist_ok=True)
            artifacts = _export_candidate(
                h5=h5,
                tally=tally,
                candidate=candidate,
                export_dir=export_dir,
                particle=particle,
            )
            exports_summary.append(
                {
                    "candidate_id": candidate.id,
                    "tally_name": candidate.tally_name,
                    "score_name": candidate.score_name,
                    "conversion_type": candidate.conversion_type,
                    "confidence": candidate.confidence,
                    "warnings": candidate.warnings,
                    "artifacts": artifacts,
                }
            )

    return {"exports": exports_summary}


def _export_candidate(
    h5: h5py.File,
    tally: TallyDescriptor,
    candidate: ExportCandidate,
    export_dir: Path,
    particle: str,
) -> list[str]:
    array, axes = _load_score_array(h5, tally, candidate.score_name)
    artifacts: list[str] = []

    if candidate.conversion_type == "energy":
        reduced, kept_axes, reduced_axes = _sum_except(array, axes, {"energy"})
        artifacts.extend(
            _write_energy_export(
                export_dir=export_dir,
                tally=tally,
                candidate=candidate,
                reduced=np.asarray(reduced),
                kept_axes=kept_axes,
                reduced_axes=reduced_axes,
                particle=particle,
            )
        )
        return artifacts

    if candidate.conversion_type == "angle":
        reduced, kept_axes, reduced_axes = _sum_except(array, axes, {"mu"})
        artifacts.extend(
            _write_angle_export(
                export_dir=export_dir,
                tally=tally,
                candidate=candidate,
                reduced=np.asarray(reduced),
                kept_axes=kept_axes,
                reduced_axes=reduced_axes,
                particle=particle,
            )
        )
        return artifacts

    raise ValueError(f"Unsupported candidate conversion type: {candidate.conversion_type}")


def _write_energy_export(
    export_dir: Path,
    tally: TallyDescriptor,
    candidate: ExportCandidate,
    reduced: np.ndarray,
    kept_axes: list[str],
    reduced_axes: list[str],
    particle: str,
) -> list[str]:
    if kept_axes != ["energy"]:
        raise ValueError(
            f"Energy export expected kept axes ['energy']; got {kept_axes}."
        )

    weights = np.asarray(reduced, dtype=float)
    energy_grid = np.asarray(tally.grids["energy"], dtype=float)
    artifacts: list[str] = []

    mac_path = export_dir / "spectrum.mac"
    _write_energy_macro(mac_path, energy_grid, weights)
    artifacts.append(mac_path.name)
    raw_path = export_dir / "energy_histogram.npz"
    np.savez(raw_path, energy=energy_grid, weight=weights)
    artifacts.append(raw_path.name)

    metadata = {
        "particle": particle,
        "source_type": candidate.source_type,
        "conversion_type": candidate.conversion_type,
        "tally_name": tally.name,
        "score_name": candidate.score_name,
        "axes_kept": kept_axes,
        "axes_reduced": reduced_axes,
        "warnings": candidate.warnings,
        "units": {"energy": "eV"},
    }
    metadata_path = export_dir / "source_metadata.json"
    metadata_path.write_text(json.dumps(_to_builtin(metadata), indent=2))
    artifacts.append(metadata_path.name)
    return artifacts


def _write_angle_export(
    export_dir: Path,
    tally: TallyDescriptor,
    candidate: ExportCandidate,
    reduced: np.ndarray,
    kept_axes: list[str],
    reduced_axes: list[str],
    particle: str,
) -> list[str]:
    if kept_axes != ["mu"]:
        raise ValueError(
            f"Angle export expected kept axes ['mu']; got {kept_axes}."
        )

    weights = np.asarray(reduced, dtype=float)
    mu_grid = np.asarray(tally.grids["mu"], dtype=float)
    theta_grid = np.arccos(np.clip(mu_grid, -1.0, 1.0))
    artifacts: list[str] = []

    mac_path = export_dir / "angle_spectrum.mac"
    _write_angle_macro(mac_path, theta_grid, weights)
    artifacts.append(mac_path.name)
    raw_path = export_dir / "angle_histogram.npz"
    np.savez(raw_path, mu=mu_grid, theta=theta_grid, weight=weights)
    artifacts.append(raw_path.name)

    metadata = {
        "particle": particle,
        "source_type": candidate.source_type,
        "conversion_type": candidate.conversion_type,
        "tally_name": tally.name,
        "score_name": candidate.score_name,
        "axes_kept": kept_axes,
        "axes_reduced": reduced_axes,
        "warnings": candidate.warnings,
        "units": {"theta": "radians"},
        "azimuth_present": "azi" in tally.grids and len(tally.grids["azi"]) - 1 > 1,
    }
    metadata_path = export_dir / "source_metadata.json"
    metadata_path.write_text(json.dumps(_to_builtin(metadata), indent=2))
    artifacts.append(metadata_path.name)
    return artifacts


def _write_energy_macro(path: Path, energy: np.ndarray, weight: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("# Automatically generated energy spectrum from MCDC HDF5\n")
        handle.write("/gps/ene/type User\n")
        handle.write("/gps/hist/type energy\n")
        for e, w in zip(energy, weight):
            handle.write(f"/gps/hist/point {float(e):.8e} {float(w):.8e}\n")


def _write_angle_macro(path: Path, theta: np.ndarray, weight: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("# Automatically generated angular spectrum from MCDC HDF5\n")
        handle.write("/gps/ang/type user\n")
        handle.write("/gps/hist/type theta\n")
        for t, w in zip(theta, weight):
            handle.write(f"/gps/hist/point {float(t):.8e} {float(w):.8e}\n")


def build_manifest(
    tallies: list[TallyDescriptor],
    candidates: list[ExportCandidate],
    unsupported: list[dict[str, Any]],
    export_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tally_entries = []
    for tally in tallies:
        tally_entries.append(
            {
                "name": tally.name,
                "tally_class": tally.tally_class,
                "grids": {
                    axis: {"size": len(values), "bins": max(len(values) - 1, 0)}
                    for axis, values in tally.grids.items()
                },
                "scores": {
                    score_name: {"shape": descriptor.shape}
                    for score_name, descriptor in tally.scores.items()
                },
                "warnings": tally.warnings,
            }
        )

    candidate_entries = []
    for candidate in candidates:
        candidate_entries.append(
            {
                "id": candidate.id,
                "tally_name": candidate.tally_name,
                "score_name": candidate.score_name,
                "conversion_type": candidate.conversion_type,
                "source_type": candidate.source_type,
                "confidence": candidate.confidence,
                "axes_present": candidate.axes_present,
                "axes_reduced": candidate.axes_reduced,
                "varied_axes": candidate.varied_axes,
                "warnings": candidate.warnings,
            }
        )

    manifest = {
        "tallies": tally_entries,
        "candidates": candidate_entries,
        "unsupported_tallies": unsupported,
    }
    if export_summary is not None:
        manifest["exports"] = export_summary["exports"]
    return manifest


def write_manifest(outdir: str | Path, manifest: dict[str, Any]) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(_to_builtin(manifest), indent=2))
    return manifest_path


def inspect_and_export(
    hdf_path: str | Path,
    outdir: str | Path,
    particle: str = "neutron",
    export_all: bool = True,
    best_only: bool = False,
    inspect_only: bool = False,
    candidate_filter: str | None = None,
) -> dict[str, Any]:
    tallies = inspect_hdf5_tallies(hdf_path)
    candidates, unsupported = classify_candidates(tallies)
    export_summary = None
    if not inspect_only and export_all:
        export_summary = export_candidates(
            hdf_path=hdf_path,
            outdir=outdir,
            candidates=candidates,
            tallies=tallies,
            particle=particle,
            candidate_filter=candidate_filter,
            best_only=best_only,
        )
    manifest = build_manifest(tallies, candidates, unsupported, export_summary)
    write_manifest(outdir, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect MCDC HDF5 tallies and export Geant4 source candidates."
    )
    parser.add_argument("--input", required=True, help="Path to MCDC output.h5")
    parser.add_argument(
        "--outdir",
        default="geant4_exports",
        help="Directory to write manifest and exported source candidates",
    )
    parser.add_argument(
        "--particle",
        default="neutron",
        help="Particle name recorded in metadata",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect tallies and write only the manifest",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Export only one candidate by its manifest id",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Export only the highest-ranked candidate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inspect_and_export(
        hdf_path=args.input,
        outdir=args.outdir,
        particle=args.particle,
        export_all=not args.inspect_only,
        best_only=args.best_only,
        inspect_only=args.inspect_only,
        candidate_filter=args.candidate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
