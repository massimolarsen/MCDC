from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
MCDC_ROOT = EXAMPLE_DIR.parents[2]
DEFAULT_INPUT = MCDC_ROOT / "examples" / "testSiliconDiode" / "output.h5"
EXPORTER_PATH = MCDC_ROOT / "mcdc" / "geant4_export.py"
BUILD_DIR = EXAMPLE_DIR / "build"
EXPORT_DIR = BUILD_DIR / "mcdc_source"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("mcdc_geant4_export", EXPORTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_candidate(manifest: dict, conversion_type: str, source_name: str, target_name: str) -> Path | None:
    candidate = next(
        (
            item
            for item in manifest.get("candidates", [])
            if item.get("conversion_type") == conversion_type
        ),
        None,
    )
    if candidate is None:
        return None

    source_path = EXPORT_DIR / candidate["id"] / source_name
    if not source_path.exists():
        return None

    target_path = BUILD_DIR / target_name
    shutil.copyfile(source_path, target_path)
    return target_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    input_path = Path(argv[0]).resolve() if argv else DEFAULT_INPUT

    if not input_path.exists():
        raise FileNotFoundError(f"MCDC output not found: {input_path}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    exporter = _load_exporter()
    manifest = exporter.inspect_and_export(hdf_path=input_path, outdir=EXPORT_DIR)

    energy_path = _copy_candidate(manifest, "energy", "spectrum.mac", "spectrum.mac")
    angle_path = _copy_candidate(
        manifest, "angle", "angle_spectrum.mac", "angle_spectrum.mac"
    )

    print(f"Prepared source files from {input_path}")
    print(f"Manifest: {EXPORT_DIR / 'manifest.json'}")
    if energy_path is not None:
        print(f"Energy source: {energy_path}")
    else:
        print("Energy source: no supported energy candidate found")
    if angle_path is not None:
        print(f"Angle source: {angle_path}")
    else:
        print("Angle source: no supported angle candidate found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
