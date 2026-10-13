"""
Export the electron CubeSat mesh tallies and geometry to ParaView files.

  <out>/body_mesh.vtr    RectilinearGrid of the body mesh with cell arrays
                         flux             electron flux [e / cm^2 / s]
                         flux_rel_err     relative standard error of flux
                         edep             energy deposition rate [MeV / cm^3 / s]
                         edep_rel_err     relative standard error of edep
                         (negative deposits, see below, are set to 0 in edep)
  <out>/geometry.vtu     one hexahedron per CubeSat component box, with cell
                         arrays component_id and material_id (legend printed
                         and written to geometry_legend.txt)

Tallies are per source history; they are scaled to the AE9 environment with
R = Phi_e * A / 4 source electrons per second and divided by voxel volume,
as in plot_cubesat_tallies.py.

The geometry is read from the input deck itself: the deck is executed with
Simulation.run disabled, and each component cell's region (an intersection of
PlaneX/Y/Z half-spaces) is converted to its box bounds.

Usage:
    python export_paraview.py runs/<jobid>/mcdc_h5/cubesat_CE_electron.h5 [--out DIR]
        [--input input_CubeSat_electron.py] [--no-geometry]
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile
from xml.sax.saxutils import escape

import h5py
import numpy as np

from plot_cubesat_tallies import MEV_PER_EV, RATE

VTK_HEXAHEDRON = 12


# =============================================================================
# Mesh tallies
# =============================================================================


def load_mesh(path):
    with h5py.File(path, "r") as f:
        t = f["tallies"]
        x, y, z = (np.asarray(t[f"body_flux/grid/{a}"]) for a in "xyz")
        tl = np.asarray(t["body_flux/flux/mean"])
        tl_sd = np.asarray(t["body_flux/flux/sdev"])
        ed = np.asarray(t["body_edep/energy_deposition/mean"])
        ed_sd = np.asarray(t["body_edep/energy_deposition/sdev"])
    return x, y, z, tl, tl_sd, ed, ed_sd


def relative_error(mean, sdev):
    out = np.zeros_like(mean)
    nonzero = mean != 0.0
    out[nonzero] = sdev[nonzero] / np.abs(mean[nonzero])
    return out


def mesh_arrays(x, y, z, tl, tl_sd, ed, ed_sd):
    vol = np.diff(x)[:, None, None] * np.diff(y)[None, :, None] * np.diff(z)[None, None, :]
    n_negative = int((ed < 0.0).sum())
    arrays = {
        "flux": tl / vol * RATE,
        "flux_rel_err": relative_error(tl, tl_sd),
        "edep": np.clip(ed, 0.0, None) * MEV_PER_EV / vol * RATE,
        "edep_rel_err": relative_error(ed, ed_sd),
    }
    return arrays, n_negative


def data_array(name, values, vtk_type="Float64", components=1):
    flat = np.asarray(values).ravel()
    fmt = "%d" if vtk_type.startswith(("Int", "UInt")) else "%.9e"
    text = " ".join(fmt % v for v in flat)
    return (
        f'<DataArray type="{vtk_type}" Name="{escape(name)}" '
        f'NumberOfComponents="{components}" format="ascii">{text}</DataArray>'
    )


def write_vtr(path, x, y, z, arrays):
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    extent = f"0 {nx} 0 {ny} 0 {nz}"
    # VTK cell order: x fastest, then y, then z
    cells = "".join(data_array(k, v.ravel(order="F")) for k, v in arrays.items())
    coords = "".join(data_array(f"{a}_coordinates", g) for a, g in zip("xyz", (x, y, z)))
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">'
        f'<RectilinearGrid WholeExtent="{extent}"><Piece Extent="{extent}">'
        f"<CellData>{cells}</CellData><Coordinates>{coords}</Coordinates>"
        "</Piece></RectilinearGrid></VTKFile>\n"
    )


# =============================================================================
# Geometry
# =============================================================================


def load_deck(input_path):
    """Execute the input deck without running the simulation; return its globals."""
    import mcdc

    mcdc.Simulation.run = lambda self, *args, **kwargs: None
    argv = sys.argv
    run_dir = os.environ.get("MCDC_RUN_DIR")
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as scratch:
        # The deck creates RUN_DIR/mcdc_h5; keep that out of the run directory
        os.environ["MCDC_RUN_DIR"] = scratch
        sys.argv = [str(input_path)]
        try:
            namespace = {"__file__": str(input_path), "__name__": "__deck__"}
            exec(compile(input_path.read_text(), str(input_path), "exec"), namespace)
        finally:
            sys.argv = argv
            os.chdir(cwd)
            if run_dir is None:
                os.environ.pop("MCDC_RUN_DIR", None)
            else:
                os.environ["MCDC_RUN_DIR"] = run_dir
    return namespace


def box_bounds(region):
    """Bounds (x0, x1, y0, y1, z0, z1) of an intersection of axis-plane half-spaces."""
    from mcdc.constant import SURFACE_PLANE_X, SURFACE_PLANE_Y, SURFACE_PLANE_Z

    lower = [-np.inf] * 3
    upper = [np.inf] * 3
    axes = {SURFACE_PLANE_X: 0, SURFACE_PLANE_Y: 1, SURFACE_PLANE_Z: 2}
    stack = [region]
    while stack:
        r = stack.pop()
        if r.type == "intersection":
            stack.extend([r.A, r.B])
        elif r.type == "halfspace" and r.A.type in axes:
            axis = axes[r.A.type]
            position = -r.A.J  # plane is u + J = 0
            if r.B > 0:
                lower[axis] = max(lower[axis], position)
            else:
                upper[axis] = min(upper[axis], position)
        else:
            raise ValueError(f"Region is not an axis-aligned box ({r.type})")
    bounds = [v for pair in zip(lower, upper) for v in pair]
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"Region is not closed: {bounds}")
    return bounds


def write_geometry(out_dir, deck):
    cells = deck["all_component_cells"]
    materials = []
    points, material_ids = [], []
    for cell in cells:
        x0, x1, y0, y1, z0, z1 = box_bounds(cell.region)
        points += [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        name = cell.fill.name
        if name not in materials:
            materials.append(name)
        material_ids.append(materials.index(name))

    n = len(cells)
    connectivity = np.arange(8 * n)
    offsets = 8 * np.arange(1, n + 1)
    types = np.full(n, VTK_HEXAHEDRON)
    (out_dir / "geometry.vtu").write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
        f'<UnstructuredGrid><Piece NumberOfPoints="{8 * n}" NumberOfCells="{n}">'
        f"<Points>{data_array('points', points, components=3)}</Points>"
        "<Cells>"
        f"{data_array('connectivity', connectivity, 'Int64')}"
        f"{data_array('offsets', offsets, 'Int64')}"
        f"{data_array('types', types, 'UInt8')}"
        "</Cells>"
        "<CellData>"
        f"{data_array('component_id', np.arange(n), 'Int32')}"
        f"{data_array('material_id', material_ids, 'Int32')}"
        "</CellData>"
        "</Piece></UnstructuredGrid></VTKFile>\n"
    )
    legend = "\n".join(f"{i}: {m}" for i, m in enumerate(materials))
    (out_dir / "geometry_legend.txt").write_text(f"material_id\n{legend}\n")
    return n, materials


# =============================================================================


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="MC/DC output .h5")
    parser.add_argument("--out", type=Path, help="destination (default: <run>/paraview)")
    parser.add_argument("--input", type=Path,
                        help="input deck for the geometry (default: the run's archived "
                             "input_CubeSat_electron.py, else the example's)")
    parser.add_argument("--no-geometry", action="store_true")
    args = parser.parse_args()

    h5 = args.output.resolve()
    run_dir = h5.parent.parent
    out_dir = (args.out or run_dir / "paraview").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, z, *scores = load_mesh(h5)
    arrays, n_negative = mesh_arrays(x, y, z, *scores)
    write_vtr(out_dir / "body_mesh.vtr", x, y, z, arrays)
    print(f"Wrote {out_dir / 'body_mesh.vtr'} ({len(x) - 1}x{len(y) - 1}x{len(z) - 1} voxels)")
    if n_negative:
        print(f"  {n_negative} voxel(s) with negative energy deposition set to 0 in edep")

    if not args.no_geometry:
        deck_path = args.input
        if deck_path is None:
            archived = run_dir / "input_CubeSat_electron.py"
            deck_path = archived if archived.is_file() else here / "input_CubeSat_electron.py"
        n, materials = write_geometry(out_dir, load_deck(deck_path.resolve()))
        print(f"Wrote {out_dir / 'geometry.vtu'} ({n} component boxes from {deck_path})")
        for i, m in enumerate(materials):
            print(f"  material_id {i}: {m}")


if __name__ == "__main__":
    main()
