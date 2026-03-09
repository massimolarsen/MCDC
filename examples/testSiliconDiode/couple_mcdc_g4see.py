#!/usr/bin/env python3
"""End-to-end MCDC -> G4SEE coupling runner.

Pipeline:
1) Run an MCDC input Python file.
2) Convert MCDC HDF5 tallies into G4SEE GPS energy/angular macros.
3) Parse the same MCDC input geometry to build a G4SEE macro.
4) Run G4SEE with the generated macro.

This currently supports diode-like geometry defined by:
- one cylinder surface + two planes on its axis for the sensitive volume cell.
"""

from __future__ import annotations

import argparse
import ast
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class SurfaceDef:
    name: str
    kind: str  # CylinderX/Y/Z or PlaneX/Y/Z
    axis: str  # x/y/z
    value: float  # radius for cylinder, coordinate for plane


@dataclass
class CellDef:
    fill_name: str
    region_terms: Dict[str, int]  # surface_name -> sign (+1/-1)


@dataclass
class DiodeGeometry:
    radius_cm: float
    length_cm: float


class InputGeometryParser:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.constants: Dict[str, float] = {}
        self.surfaces: Dict[str, SurfaceDef] = {}
        self.cells: List[CellDef] = []

    def parse(self) -> None:
        tree = ast.parse(self.input_path.read_text(), filename=str(self.input_path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                self._handle_assign(node)
            elif isinstance(node, ast.Expr):
                self._handle_expr(node)

    def extract_diode_geometry(self, sensitive_fill: str) -> DiodeGeometry:
        sv_cell = None
        for cell in self.cells:
            if cell.fill_name == sensitive_fill:
                sv_cell = cell
                break

        if sv_cell is None:
            raise RuntimeError(
                f"No mcdc.Cell found with fill={sensitive_fill!r}. "
                "Use --sensitive-fill to select a different fill variable name."
            )

        cyl: Optional[SurfaceDef] = None
        planes_on_axis: List[SurfaceDef] = []

        for surf_name in sv_cell.region_terms:
            surf = self.surfaces.get(surf_name)
            if surf is None:
                continue
            if surf.kind.startswith("Cylinder"):
                cyl = surf

        if cyl is None:
            raise RuntimeError(
                f"Sensitive cell for fill={sensitive_fill!r} does not contain a cylinder surface."
            )

        for surf_name in sv_cell.region_terms:
            surf = self.surfaces.get(surf_name)
            if surf is None:
                continue
            if surf.kind.startswith("Plane") and surf.axis == cyl.axis:
                planes_on_axis.append(surf)

        if len(planes_on_axis) < 2:
            raise RuntimeError(
                "Could not infer diode length: expected two plane surfaces on the cylinder axis "
                f"({cyl.axis.upper()})."
            )

        # Use the two most extreme planes on the cylinder axis.
        values = sorted([p.value for p in planes_on_axis])
        length_cm = values[-1] - values[0]

        if length_cm <= 0:
            raise RuntimeError("Invalid inferred diode length (<= 0).")

        return DiodeGeometry(radius_cm=cyl.value, length_cm=length_cm)

    def _handle_assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return

        target = node.targets[0].id

        # Numeric constant assignment
        numeric_value = self._eval_num(node.value)
        if numeric_value is not None:
            self.constants[target] = numeric_value
            return

        # Surface assignment
        if isinstance(node.value, ast.Call):
            surface = self._parse_surface_call(target, node.value)
            if surface is not None:
                self.surfaces[target] = surface

    def _handle_expr(self, node: ast.Expr) -> None:
        if not isinstance(node.value, ast.Call):
            return

        call = node.value
        if not self._is_mcdc_attr_call(call.func, "Cell"):
            return

        fill_name = None
        region_terms: Dict[str, int] = {}

        for kw in call.keywords:
            if kw.arg == "fill" and isinstance(kw.value, ast.Name):
                fill_name = kw.value.id
            elif kw.arg == "region":
                region_terms = self._flatten_region(kw.value)

        if fill_name is None:
            return

        self.cells.append(CellDef(fill_name=fill_name, region_terms=region_terms))

    def _parse_surface_call(self, name: str, call: ast.Call) -> Optional[SurfaceDef]:
        func = call.func
        if not isinstance(func, ast.Attribute):
            return None

        kind = func.attr
        if kind not in {"CylinderX", "CylinderY", "CylinderZ", "PlaneX", "PlaneY", "PlaneZ"}:
            return None

        axis = kind[-1].lower()
        value = None

        if kind.startswith("Cylinder"):
            for kw in call.keywords:
                if kw.arg == "radius":
                    value = self._eval_num(kw.value)
                    break
        else:
            coord_arg = axis
            for kw in call.keywords:
                if kw.arg == coord_arg:
                    value = self._eval_num(kw.value)
                    break

        if value is None:
            return None

        return SurfaceDef(name=name, kind=kind, axis=axis, value=value)

    def _flatten_region(self, expr: ast.AST) -> Dict[str, int]:
        terms: Dict[str, int] = {}

        def walk(node: ast.AST) -> None:
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                walk(node.left)
                walk(node.right)
                return

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                sign = 1 if isinstance(node.op, ast.UAdd) else -1
                if isinstance(node.operand, ast.Name):
                    terms[node.operand.id] = sign
                return

            if isinstance(node, ast.Name):
                terms[node.id] = 1

        walk(expr)
        return terms

    def _eval_num(self, node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.Name):
            return self.constants.get(node.id)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = self._eval_num(node.operand)
            if val is None:
                return None
            return val if isinstance(node.op, ast.UAdd) else -val

        if isinstance(node, ast.BinOp):
            left = self._eval_num(node.left)
            right = self._eval_num(node.right)
            if left is None or right is None:
                return None

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right

        return None

    @staticmethod
    def _is_mcdc_attr_call(func: ast.AST, attr_name: str) -> bool:
        return (
            isinstance(func, ast.Attribute)
            and func.attr == attr_name
            and isinstance(func.value, ast.Name)
            and func.value.id == "mcdc"
        )


def generate_g4see_spectra(hdf_path: Path, spectrum_mac: Path, angle_mac: Path) -> None:
    with h5py.File(hdf_path, "r") as f:
        try:
            energies = f["tallies/surface_tally_0/grid/energy"][:]
            eng_flux = f["tallies/surface_tally_0/net-current/mean"][:]
        except Exception as exc:
            raise RuntimeError(
                "Could not find energy tally at tallies/surface_tally_0/grid/energy and net-current/mean"
            ) from exc

        try:
            mu = f["tallies/surface_tally_1/grid/mu"][:]
            mu_flux = f["tallies/surface_tally_1/flux/mean"][:]
        except Exception:
            mu = None
            mu_flux = None

    with spectrum_mac.open("w") as mac:
        mac.write("# Automatically generated energy spectrum from MCDC HDF5\n")
        mac.write("/gps/ene/type User\n")
        mac.write("/gps/hist/type energy\n")
        for e, w in zip(energies, eng_flux):
            mac.write(f"/gps/hist/point {e:.8e} {w:.8e}\n")

    if mu is None or mu_flux is None:
        return

    theta = np.arccos(np.clip(mu, -1.0, 1.0))
    with angle_mac.open("w") as mac:
        mac.write("# Automatically generated angular spectrum from MCDC HDF5\n")
        mac.write("/gps/ang/type user\n")
        mac.write("/gps/hist/type theta\n")
        for t, p in zip(theta, mu_flux):
            mac.write(f"/gps/hist/point {t:.8e} {p:.8e}\n")


def write_g4see_macro(
    macro_path: Path,
    spectrum_mac: Path,
    angle_mac: Path,
    geometry: DiodeGeometry,
    threads: int,
    beam_on: int,
) -> None:
    bulk_width_mm = max(1e-9, 2.0 * geometry.radius_cm * 10.0)
    bulk_thickness_mm = max(1e-9, geometry.length_cm * 10.0)

    # Keep SV slightly smaller than bulk to avoid geometric overlaps.
    margin_mm = 0.001  # 1 um radial margin
    sv_width_mm = max(1e-6, bulk_width_mm - 2.0 * margin_mm)
    sv_thickness_um = max(1e-3, bulk_thickness_mm * 1000.0 - 1.0)  # 1 um axial margin

    text = f"""# Auto-generated by couple_mcdc_g4see.py
/run/numberOfThreads                {threads}
/run/printProgress                  100
/tracking/verbose                   0

/SEE/geometry/setCylindricalGeometry   true
/SEE/geometry/Bulk                  G4_Si               {bulk_width_mm:.6g} mm   {bulk_thickness_mm:.6g} mm   true
/SEE/geometry/SV                    0 0 0 um            {sv_width_mm:.6g} {sv_width_mm:.6g} mm    {sv_thickness_um:.6g} um      true
/SEE/geometry/BEOL/addLayer         G4_Al               {bulk_width_mm:.6g} mm   500 nm      Metal   false

/SEE/biasing/biasParticle           neutron
/SEE/biasing/biasProcess            hadElastic
/SEE/biasing/biasProcess            neutronInelastic
/SEE/biasing/biasProcess            nCapture
/SEE/biasing/biasFactor             1000

/SEE/physics/addPhysics             G4EmStandardPhysics_option4
/SEE/physics/addPhysics             G4HadronElasticPhysicsHP
/SEE/physics/addPhysics             G4HadronPhysicsFTFP_BERT_HP

/SEE/cuts/hadrons           BEOL    1 um
/SEE/cuts/hadrons           Bulk    1 um
/SEE/cuts/hadrons           SV      1 um

/run/initialize

/gps/particle                       neutron
/control/execute                    {spectrum_mac}
/control/execute                    {angle_mac}
/gps/pos/centre                     0 0 1 mm
/gps/pos/type                       Plane
/gps/pos/shape                      Rectangle
/gps/pos/halfx                      {bulk_width_mm/2.0:.6g} mm
/gps/pos/halfy                      {bulk_width_mm/2.0:.6g} mm

/SEE/scoring/standard/Edep          0
/SEE/scoring/setHistogram           lin     200     0           15  MeV
/SEE/scoring/standard/Edep          1
/SEE/scoring/setHistogram           log     200     0.001       20  MeV

/SEE/scoring/standard/Ekin          2       neutron
/SEE/scoring/setHistogram           log     200     0.001       20  MeV
/SEE/scoring/standard/Ekin          3       gamma
/SEE/scoring/setHistogram           log     200     0.001       20  MeV

/SEE/scoring/dumpHistogramsAfter    5000

/run/beamOn                         {beam_on}
"""
    macro_path.write_text(text)


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def find_g4see_executable(explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "g4see" / "build" / "g4see"
    if candidate.exists():
        return str(candidate)

    from_path = shutil.which("g4see")
    if from_path:
        return from_path

    raise RuntimeError("Could not find g4see executable. Pass --g4see explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCDC->G4SEE coupled workflow")
    parser.add_argument("--input", default="input.py", help="Path to MCDC input Python file")
    parser.add_argument("--output-h5", default="output.h5", help="Expected MCDC output HDF5 filename")
    parser.add_argument("--sensitive-fill", default="SV", help="MCDC fill variable name used as sensitive volume")
    parser.add_argument("--g4see", default=None, help="Path to g4see executable")
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run the MCDC input",
    )
    parser.add_argument("--beam-on", type=int, default=200, help="G4SEE /run/beamOn value")
    parser.add_argument("--threads", type=int, default=4, help="G4SEE /run/numberOfThreads")
    parser.add_argument("--skip-mcdc", action="store_true", help="Skip MCDC execution and reuse output-h5")
    parser.add_argument("--skip-g4see", action="store_true", help="Generate files but do not run g4see")
    parser.add_argument(
        "--mcdc-arg",
        action="append",
        default=[],
        help="Extra arg forwarded to MCDC input execution (repeatable)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    workdir = input_path.parent
    output_h5 = workdir / args.output_h5

    parser_obj = InputGeometryParser(input_path)
    parser_obj.parse()
    geometry = parser_obj.extract_diode_geometry(args.sensitive_fill)

    # 1) Run MCDC input
    if not args.skip_mcdc:
        run_cmd([args.python_exe, str(input_path), *args.mcdc_arg], cwd=workdir)

    if not output_h5.exists():
        raise RuntimeError(f"Expected MCDC output file does not exist: {output_h5}")

    # 2) Convert output.h5 -> G4SEE source distributions
    spectrum_mac = workdir / "spectrum.mac"
    angle_mac = workdir / "angle_spectrum.mac"
    generate_g4see_spectra(output_h5, spectrum_mac, angle_mac)

    # 3) Generate G4SEE macro from inferred MCDC geometry
    g4_macro = workdir / "g4see_from_mcdc.mac"
    write_g4see_macro(
        macro_path=g4_macro,
        spectrum_mac=spectrum_mac,
        angle_mac=angle_mac,
        geometry=geometry,
        threads=args.threads,
        beam_on=args.beam_on,
    )

    # 4) Run G4SEE
    if args.skip_g4see:
        print(f"[ok] Generated: {spectrum_mac}")
        print(f"[ok] Generated: {angle_mac}")
        print(f"[ok] Generated: {g4_macro}")
        return

    g4see_exe = find_g4see_executable(args.g4see)
    out_dir = workdir / "g4see_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([g4see_exe, str(g4_macro), "-o", str(out_dir)], cwd=workdir)

    print("[ok] Coupled run finished")
    print(f"[ok] MCDC HDF5: {output_h5}")
    print(f"[ok] G4SEE macro: {g4_macro}")
    print(f"[ok] G4SEE output dir: {out_dir}")


if __name__ == "__main__":
    main()
