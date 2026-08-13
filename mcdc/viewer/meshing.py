"""Sampled-field geometry helpers for 3D visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from mcdc.constant import (
    BOOL_AND,
    BOOL_NOT,
    BOOL_OR,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_SPHERE,
    SURFACE_TORUS,
    SURFACE_TORUS_X,
    SURFACE_TORUS_Y,
    SURFACE_TORUS_Z,
)
from mcdc.viewer.types import Bounds3D

SAMPLE_BACKEND = "sampled-field"
DEFAULT_SAMPLE_RESOLUTION = 96


@dataclass(frozen=True)
class SampleGrid:
    """Regular point grid used to sample implicit geometry."""

    bounds: Bounds3D
    resolution: tuple[int, int, int]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray

    @property
    def dimensions(self):
        return self.resolution

    @property
    def origin(self):
        return (self.bounds.x[0], self.bounds.y[0], self.bounds.z[0])

    @property
    def spacing(self):
        nx, ny, nz = self.resolution
        return (
            (self.bounds.x[1] - self.bounds.x[0]) / max(nx - 1, 1),
            (self.bounds.y[1] - self.bounds.y[0]) / max(ny - 1, 1),
            (self.bounds.z[1] - self.bounds.z[0]) / max(nz - 1, 1),
        )


def normalize_sample_resolution(sample_resolution, minimum=3):
    """Return a three-axis sample resolution tuple."""

    if isinstance(sample_resolution, int):
        resolution = (sample_resolution, sample_resolution, sample_resolution)
    else:
        resolution = tuple(int(value) for value in sample_resolution)
        if len(resolution) != 3:
            raise ValueError("sample_resolution must be an int or a 3-item sequence.")
    if any(value < minimum for value in resolution):
        raise ValueError(f"sample_resolution values must be >= {minimum}.")
    return resolution


def make_grid(bounds, sample_resolution):
    """Create point coordinates and meshgrid arrays over bounds."""

    resolution = normalize_sample_resolution(sample_resolution)
    x = np.linspace(bounds.x[0], bounds.x[1], resolution[0])
    y = np.linspace(bounds.y[0], bounds.y[1], resolution[1])
    z = np.linspace(bounds.z[0], bounds.z[1], resolution[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return SampleGrid(bounds, resolution, x, y, z, X, Y, Z)


def _padded_axis(lo, hi, fraction=0.08):
    lo, hi = float(lo), float(hi)
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    pad = fraction * (hi - lo)
    return lo - pad, hi + pad


def _torus_axis(surface):
    if surface.type == SURFACE_TORUS_X:
        return np.array([1.0, 0.0, 0.0])
    if surface.type == SURFACE_TORUS_Y:
        return np.array([0.0, 1.0, 0.0])
    if surface.type == SURFACE_TORUS_Z:
        return np.array([0.0, 0.0, 1.0])
    return np.array([surface.nx, surface.ny, surface.nz], dtype=float)


def _surface_extent_candidates(surface, shift):
    values = {"x": [], "y": [], "z": []}
    shift = np.asarray(shift, dtype=float)

    if surface.type == SURFACE_PLANE_X:
        values["x"].append(-surface.J + shift[0])
    elif surface.type == SURFACE_PLANE_Y:
        values["y"].append(-surface.J + shift[1])
    elif surface.type == SURFACE_PLANE_Z:
        values["z"].append(-surface.J + shift[2])

    if surface.type == SURFACE_CYLINDER_X:
        cy0, cz0 = -0.5 * surface.H, -0.5 * surface.I
        radius_sq = cy0 * cy0 + cz0 * cz0 - surface.J
        if radius_sq > 0.0:
            cy, cz = cy0 + shift[1], cz0 + shift[2]
            radius = np.sqrt(radius_sq)
            values["y"].extend([cy - radius, cy + radius])
            values["z"].extend([cz - radius, cz + radius])
    elif surface.type == SURFACE_CYLINDER_Y:
        cx0, cz0 = -0.5 * surface.G, -0.5 * surface.I
        radius_sq = cx0 * cx0 + cz0 * cz0 - surface.J
        if radius_sq > 0.0:
            cx, cz = cx0 + shift[0], cz0 + shift[2]
            radius = np.sqrt(radius_sq)
            values["x"].extend([cx - radius, cx + radius])
            values["z"].extend([cz - radius, cz + radius])
    elif surface.type == SURFACE_CYLINDER_Z:
        cx0, cy0 = -0.5 * surface.G, -0.5 * surface.H
        radius_sq = cx0 * cx0 + cy0 * cy0 - surface.J
        if radius_sq > 0.0:
            cx, cy = cx0 + shift[0], cy0 + shift[1]
            radius = np.sqrt(radius_sq)
            values["x"].extend([cx - radius, cx + radius])
            values["y"].extend([cy - radius, cy + radius])
    elif surface.type == SURFACE_SPHERE:
        cx0 = -0.5 * surface.G
        cy0 = -0.5 * surface.H
        cz0 = -0.5 * surface.I
        radius_sq = cx0 * cx0 + cy0 * cy0 + cz0 * cz0 - surface.J
        if radius_sq > 0.0:
            cx, cy, cz = cx0 + shift[0], cy0 + shift[1], cz0 + shift[2]
            radius = np.sqrt(radius_sq)
            values["x"].extend([cx - radius, cx + radius])
            values["y"].extend([cy - radius, cy + radius])
            values["z"].extend([cz - radius, cz + radius])
    elif surface.type in (
        SURFACE_TORUS_X,
        SURFACE_TORUS_Y,
        SURFACE_TORUS_Z,
        SURFACE_TORUS,
    ):
        cx, cy, cz = surface.A + shift[0], surface.B + shift[1], surface.C + shift[2]
        radius = surface.R + surface.r
        values["x"].extend([cx - radius, cx + radius])
        values["y"].extend([cy - radius, cy + radius])
        values["z"].extend([cz - radius, cz + radius])

    return values


def bounds_from_planes_with_shift(simulation, surface_shift_map):
    """Infer padded world bounds from planes and finite surface extents."""

    values = {"x": [], "y": [], "z": []}
    fallback = 1.0
    for surface in simulation.surfaces:
        shift = surface_shift_map.get(surface.ID, np.zeros(3, dtype=float))
        candidates = _surface_extent_candidates(surface, shift)
        for axis in values:
            values[axis].extend(candidates[axis])
        fallback = max(
            fallback,
            abs(getattr(surface, "A", 0.0)),
            abs(getattr(surface, "B", 0.0)),
            abs(getattr(surface, "C", 0.0)),
            abs(getattr(surface, "J", 0.0)),
            abs(getattr(surface, "R", 0.0) + getattr(surface, "r", 0.0)),
        )

    def axis(axis_values):
        if len(axis_values) >= 2:
            return _padded_axis(np.min(axis_values), np.max(axis_values))
        if len(axis_values) == 1:
            delta = max(1.0, abs(axis_values[0]))
            return _padded_axis(-delta, delta)
        return _padded_axis(-fallback, fallback)

    return Bounds3D(x=axis(values["x"]), y=axis(values["y"]), z=axis(values["z"]))


def cell_bounds(cell, global_bounds, surface_shift_map):
    """Infer local sampling bounds for one cell, clipped to global bounds."""

    values = {"x": [], "y": [], "z": []}
    for surface in getattr(cell, "surfaces", []):
        shift = surface_shift_map.get(surface.ID, np.zeros(3, dtype=float))
        candidates = _surface_extent_candidates(surface, shift)
        for axis in values:
            values[axis].extend(candidates[axis])

    out = []
    for axis_name, global_axis in zip(
        ("x", "y", "z"), (global_bounds.x, global_bounds.y, global_bounds.z)
    ):
        axis_values = values[axis_name]
        if len(axis_values) >= 2:
            lo, hi = _padded_axis(
                np.min(axis_values), np.max(axis_values), fraction=0.04
            )
            lo = max(lo, global_axis[0])
            hi = min(hi, global_axis[1])
            if lo < hi:
                out.append((lo, hi))
                continue
        out.append(global_axis)
    return Bounds3D(x=out[0], y=out[1], z=out[2])


def quadric_field(surface, grid, shift):
    """Evaluate a linear or quadric MC/DC surface on a sampled grid."""

    x = grid.X - shift[0]
    y = grid.Y - shift[1]
    z = grid.Z - shift[2]
    return (
        surface.A * x * x
        + surface.B * y * y
        + surface.C * z * z
        + surface.D * x * y
        + surface.E * x * z
        + surface.F * y * z
        + surface.G * x
        + surface.H * y
        + surface.I * z
        + surface.J
    )


def torus_field(surface, grid, shift):
    """Evaluate an axis-aligned or arbitrary-axis torus on a sampled grid."""

    center = np.array([surface.A, surface.B, surface.C], dtype=float) + shift
    axis = _torus_axis(surface)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        raise ValueError("Torus axis must be nonzero.")
    axis /= axis_norm
    x = grid.X - center[0]
    y = grid.Y - center[1]
    z = grid.Z - center[2]
    p_dot_p = x * x + y * y + z * z
    p_dot_d = x * axis[0] + y * axis[1] + z * axis[2]
    radial_sq = p_dot_p - p_dot_d * p_dot_d
    q = p_dot_p + surface.R * surface.R - surface.r * surface.r
    return q * q - 4.0 * surface.R * surface.R * radial_sq


def surface_field(surface, grid, shift):
    """Evaluate one MC/DC surface field on a sampled grid."""

    if surface.type in (
        SURFACE_TORUS_X,
        SURFACE_TORUS_Y,
        SURFACE_TORUS_Z,
        SURFACE_TORUS,
    ):
        return torus_field(surface, grid, shift)
    return quadric_field(surface, grid, shift)


def region_mask_from_rpn(region_RPN_tokens, surface_fields, shape):
    """Evaluate compiled region RPN tokens into a boolean mask."""

    tokens = list(region_RPN_tokens)
    if len(tokens) == 0:
        return np.ones(shape, dtype=bool)
    stack = []
    for token in tokens:
        token = int(token)
        if token >= 0:
            stack.append(surface_fields[token] >= 0.0)
        elif token == BOOL_NOT:
            stack.append(~stack.pop())
        elif token == BOOL_AND:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(lhs & rhs)
        elif token == BOOL_OR:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(lhs | rhs)
        else:
            raise ValueError(f"Unsupported region RPN token {token}.")
    if len(stack) != 1:
        raise ValueError("Invalid region RPN expression.")
    return stack[0]


def _collect_region_surfaces(region, out):
    if region.type == "halfspace":
        out[region.A.ID] = region.A
    elif region.type in {"intersection", "union"}:
        _collect_region_surfaces(region.A, out)
        _collect_region_surfaces(region.B, out)
    elif region.type == "complement":
        _collect_region_surfaces(region.A, out)


def _region_mask(region, surface_fields, shape):
    if region.type == "all":
        return np.ones(shape, dtype=bool)
    if region.type == "halfspace":
        return (
            surface_fields[region.A.ID] >= 0.0
            if region.B > 0
            else surface_fields[region.A.ID] < 0.0
        )
    if region.type == "intersection":
        return _region_mask(region.A, surface_fields, shape) & _region_mask(
            region.B, surface_fields, shape
        )
    if region.type == "union":
        return _region_mask(region.A, surface_fields, shape) | _region_mask(
            region.B, surface_fields, shape
        )
    if region.type == "complement":
        return ~_region_mask(region.A, surface_fields, shape)
    raise ValueError(f"Unsupported region type {region.type!r}.")


def _image_data_from_points(pv, grid, values, name):
    image = pv.ImageData(
        dimensions=grid.dimensions, spacing=grid.spacing, origin=grid.origin
    )
    image.point_data[name] = np.asarray(values).ravel(order="F")
    return image


def _mask_to_polydata(mask, pv, grid):
    if not np.any(mask):
        return None
    if np.all(mask):
        return pv.Cube(
            center=[
                0.5 * (grid.bounds.x[0] + grid.bounds.x[1]),
                0.5 * (grid.bounds.y[0] + grid.bounds.y[1]),
                0.5 * (grid.bounds.z[0] + grid.bounds.z[1]),
            ],
            x_length=grid.bounds.x[1] - grid.bounds.x[0],
            y_length=grid.bounds.y[1] - grid.bounds.y[0],
            z_length=grid.bounds.z[1] - grid.bounds.z[0],
        )
    image = _image_data_from_points(pv, grid, mask.astype(np.float32), "inside")
    mesh = image.contour(isosurfaces=[0.5], scalars="inside")
    return mesh if getattr(mesh, "n_points", 0) > 0 else None


def _surface_fields_for_surfaces(surfaces, grid, surface_shift_map):
    return {
        surface.ID: surface_field(
            surface,
            grid,
            surface_shift_map.get(surface.ID, np.zeros(3, dtype=float)),
        )
        for surface in surfaces
    }


def cell_mesh(cell, global_bounds, pv, sample_resolution, surface_shift_map):
    """Sample and contour one cell on a local grid."""

    local_bounds = cell_bounds(cell, global_bounds, surface_shift_map)
    grid = make_grid(local_bounds, sample_resolution)
    surface_fields = _surface_fields_for_surfaces(
        getattr(cell, "surfaces", []), grid, surface_shift_map
    )
    mask = region_mask_from_rpn(cell.region_RPN_tokens, surface_fields, grid.dimensions)
    return _mask_to_polydata(mask, pv, grid)


def region_mesh(region, bounds, pv, sample_resolution, surface_shift_map):
    """Sample and contour a region expression on a grid."""

    grid = make_grid(bounds, sample_resolution)
    surfaces = {}
    _collect_region_surfaces(region, surfaces)
    surface_fields = _surface_fields_for_surfaces(
        surfaces.values(), grid, surface_shift_map
    )
    mask = _region_mask(region, surface_fields, grid.dimensions)
    return _mask_to_polydata(mask, pv, grid)


def _normalize_vtk_bounds(vtk_bounds, default_bounds):
    if vtk_bounds is None:
        return default_bounds
    if isinstance(vtk_bounds, Bounds3D):
        return vtk_bounds
    if isinstance(vtk_bounds, dict):
        return Bounds3D(
            x=tuple(vtk_bounds["x"]),
            y=tuple(vtk_bounds["y"]),
            z=tuple(vtk_bounds["z"]),
        )
    if len(vtk_bounds) != 3:
        raise ValueError("vtk_bounds must be Bounds3D, a dict, or three axis ranges.")
    return Bounds3D(
        x=tuple(vtk_bounds[0]), y=tuple(vtk_bounds[1]), z=tuple(vtk_bounds[2])
    )


def _cell_center_grid(bounds, sample_resolution):
    nx, ny, nz = normalize_sample_resolution(sample_resolution, minimum=1)
    x_edges = np.linspace(bounds.x[0], bounds.x[1], nx + 1)
    y_edges = np.linspace(bounds.y[0], bounds.y[1], ny + 1)
    z_edges = np.linspace(bounds.z[0], bounds.z[1], nz + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return SampleGrid(bounds, (nx, ny, nz), x, y, z, X, Y, Z)


def sampled_cell_arrays(simulation, bounds, shifts, sample_resolution):
    """Sample global cell/material IDs on voxel centers."""

    grid = _cell_center_grid(bounds, sample_resolution)
    shape = grid.dimensions
    cell_id = np.full(shape, -1, dtype=np.int32)
    material_id = np.full(shape, -1, dtype=np.int32)
    surface_fields = _surface_fields_for_surfaces(
        simulation.surfaces, grid, shifts.surface
    )
    for cell in simulation.cells:
        mask = region_mask_from_rpn(cell.region_RPN_tokens, surface_fields, shape)
        unassigned = mask & (cell_id < 0)
        cell_id[unassigned] = int(cell.ID)
        material_id[unassigned] = int(getattr(getattr(cell, "fill", None), "ID", -1))
    return grid, cell_id, material_id


def sampled_image_data(
    pv, simulation, bounds, shifts, sample_resolution, vtk_bounds=None
):
    """Build a PyVista ImageData object with cell and material IDs."""

    bounds = _normalize_vtk_bounds(vtk_bounds, bounds)
    grid, cell_id, material_id = sampled_cell_arrays(
        simulation, bounds, shifts, sample_resolution
    )
    nx, ny, nz = grid.resolution
    image = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(
            (bounds.x[1] - bounds.x[0]) / nx,
            (bounds.y[1] - bounds.y[0]) / ny,
            (bounds.z[1] - bounds.z[0]) / nz,
        ),
        origin=(bounds.x[0], bounds.y[0], bounds.z[0]),
    )
    image.cell_data["cell_id"] = cell_id.ravel(order="F")
    image.cell_data["material_id"] = material_id.ravel(order="F")
    return image


def save_vti(
    pv, simulation, bounds, shifts, sample_resolution, save_vtk_path, vtk_bounds=None
):
    """Write a sampled static geometry grid to VTI."""

    image = sampled_image_data(
        pv, simulation, bounds, shifts, sample_resolution, vtk_bounds=vtk_bounds
    )
    image.save(str(save_vtk_path))


def save_pvd_collection(save_vtk_path, frame_paths, times):
    """Write a ParaView collection file for a time series."""

    collection = ET.Element(
        "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
    )
    datasets = ET.SubElement(collection, "Collection")
    base_dir = Path(save_vtk_path).parent
    for path, time_value in zip(frame_paths, times):
        ET.SubElement(
            datasets,
            "DataSet",
            timestep=str(float(time_value)),
            group="",
            part="0",
            file=str(Path(path).relative_to(base_dir)),
        )
    tree = ET.ElementTree(collection)
    ET.indent(tree, space="  ")
    tree.write(save_vtk_path, encoding="utf-8", xml_declaration=True)


def save_vtk_time_series(
    pv,
    simulation,
    times,
    bounds_for_time,
    shifts_for_time,
    sample_resolution,
    save_vtk_path,
    vtk_bounds=None,
):
    """Write a VTI/PVD time series for sampled geometry."""

    save_path = Path(save_vtk_path)
    frame_dir = save_path.with_suffix("")
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index, (time_value, bounds, shifts) in enumerate(
        zip(times, bounds_for_time, shifts_for_time)
    ):
        frame_path = frame_dir / f"{save_path.stem}_{index:04d}.vti"
        save_vti(
            pv,
            simulation,
            bounds,
            shifts,
            sample_resolution,
            frame_path,
            vtk_bounds=vtk_bounds,
        )
        frame_paths.append(frame_path)
    save_pvd_collection(save_path, frame_paths, times)
