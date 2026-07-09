from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mcdc
from mcdc.constant import SURFACE_PLANE_X
from mcdc.object_.simulation import simulation
from mcdc.viewer.geometry import geo_viewer_3d
from mcdc.viewer.meshing import Bounds3D, bounds_from_planes_with_shift, region_mesh
from mcdc.viewer.motion import infer_time_steps, translation_at_time
from mcdc.viewer.render import build_frame_entry
from mcdc.viewer.types import ShiftState


class FakeMesh:
    def __init__(self):
        self.faces = np.array([[0, 1, 2]], dtype=np.int64)
        self.vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def __add__(self, other):
        return FakeMesh()

    def process(self, validate=True):
        return self

    def slice_plane(self, plane_origin, plane_normal, cap=True):
        return FakeMesh()

    def apply_translation(self, center):
        return None

    def apply_transform(self, transform):
        return None


class FakeCreation:
    @staticmethod
    def box(extents, transform):
        return FakeMesh()

    @staticmethod
    def icosphere(subdivisions, radius):
        return FakeMesh()

    @staticmethod
    def cylinder(radius, height, sections):
        return FakeMesh()


class FakeBoolean:
    @staticmethod
    def intersection(meshes, engine):
        return FakeMesh()

    @staticmethod
    def union(meshes, engine):
        return FakeMesh()

    @staticmethod
    def difference(meshes, engine):
        return FakeMesh()


class FakeTransformations:
    @staticmethod
    def rotation_matrix(angle, axis):
        return np.eye(4)


class FakeTrimesh:
    creation = FakeCreation()
    boolean = FakeBoolean()
    transformations = FakeTransformations()


class FakePolyData:
    n_points = 3


class FakePyVista:
    @staticmethod
    def PolyData(vertices, faces):
        return FakePolyData()

    @staticmethod
    def Sphere(radius, center):
        return FakePolyData()

    @staticmethod
    def Cube(center, x_length, y_length, z_length):
        return FakePolyData()


def test_viewer_imports_without_runtime_dependencies():
    repo = Path(__file__).resolve().parents[3]
    code = """
import builtins
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'pyvista', 'trimesh', 'manifold3d'}:
        raise AssertionError(f'unexpected import: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import mcdc
import mcdc.viewer
from mcdc.viewer import geo_viewer_3d
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_bounds_from_shifted_planes():
    surface = SimpleNamespace(ID=0, type=SURFACE_PLANE_X, J=-2.0)
    sim = SimpleNamespace(surfaces=[surface])

    bounds = bounds_from_planes_with_shift(
        sim,
        {surface.ID: np.array([1.0, 0.0, 0.0])},
    )

    assert bounds.x == pytest.approx((-3.48, 3.48))
    assert bounds.y == pytest.approx((-2.32, 2.32))
    assert bounds.z == pytest.approx((-2.32, 2.32))


def test_motion_translation_and_time_inference():
    moving = SimpleNamespace(
        moving=True,
        move_time_grid=np.array([0.0, 2.0, 5.0, np.inf]),
        move_translations=np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]
        ),
        move_velocities=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    source = SimpleNamespace(
        ID=0,
        moving=False,
        discrete_time=False,
        time_range=np.array([1.0, 4.0]),
    )
    sim = SimpleNamespace(surfaces=[moving], sources=[source])

    assert translation_at_time(moving, 3.5) == pytest.approx([2.0, 1.5, 0.0])
    assert infer_time_steps(sim) == [0.0, 1.0, 2.0, 4.0, 5.0]


def test_geo_viewer_dispatches_static_and_time(monkeypatch):
    calls = []

    def fake_static(**kwargs):
        calls.append(("static", kwargs))

    def fake_time(**kwargs):
        calls.append(("time", kwargs))

    monkeypatch.setattr("mcdc.viewer.geometry.geo_viewer_3d_static", fake_static)
    monkeypatch.setattr("mcdc.viewer.geometry.geo_viewer_3d_time", fake_time)

    geo_viewer_3d(SimpleNamespace(surfaces=[], sources=[]))
    geo_viewer_3d(SimpleNamespace(surfaces=[], sources=[]), time_steps=[0.0, 1.0])

    moving_surface = SimpleNamespace(
        moving=True,
        move_time_grid=np.array([0.0, 1.0, np.inf]),
    )
    geo_viewer_3d(SimpleNamespace(surfaces=[moving_surface], sources=[]))

    assert [name for name, _ in calls] == ["static", "time", "time"]


def test_region_mesh_uses_current_region_tree():
    x0 = mcdc.Surface.PlaneX(x=-1.0)
    x1 = mcdc.Surface.PlaneX(x=1.0)
    sphere = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=0.5)
    region = (+x0 & -x1) | ~(-sphere)

    mesh = region_mesh(
        region,
        Bounds3D(x=(-2.0, 2.0), y=(-2.0, 2.0), z=(-2.0, 2.0)),
        FakeTrimesh,
        primitive_resolution=16,
        surface_shift_map={surface.ID: np.zeros(3) for surface in simulation.surfaces},
    )

    assert mesh is not None


def test_torus_region_warns_and_supported_cells_still_render():
    material = mcdc.MaterialMG(name="mat", capture=np.array([1.0]))
    x0 = mcdc.Surface.PlaneX(x=-2.0)
    x1 = mcdc.Surface.PlaneX(x=2.0)
    y0 = mcdc.Surface.PlaneY(y=-2.0)
    y1 = mcdc.Surface.PlaneY(y=2.0)
    z0 = mcdc.Surface.PlaneZ(z=-2.0)
    z1 = mcdc.Surface.PlaneZ(z=2.0)
    torus = mcdc.Surface.TorusZ(R=1.0, r=0.25)

    supported = +x0 & -x1 & +y0 & -y1 & +z0 & -z1
    mcdc.Cell(name="box", region=supported, fill=material)
    mcdc.Cell(name="unsupported_torus", region=-torus, fill=material)

    shifts = ShiftState(
        surface={surface.ID: np.zeros(3) for surface in simulation.surfaces},
        source={},
    )
    with pytest.warns(RuntimeWarning, match="does not support surface"):
        frame = build_frame_entry(
            simulation=simulation,
            bounds=Bounds3D(x=(-3.0, 3.0), y=(-3.0, 3.0), z=(-3.0, 3.0)),
            tm=FakeTrimesh,
            pv=FakePyVista,
            primitive_resolution=16,
            shifts=shifts,
        )

    assert frame.has_geometry
    assert [cell.label for cell in frame.cells] == ["box"]


def test_source_entries_are_included_in_legend():
    source = SimpleNamespace(
        ID=0,
        name="source marker",
        point_source=True,
        point=np.array([0.0, 0.0, 0.0]),
    )
    sim = SimpleNamespace(cells=[], sources=[source])
    shifts = ShiftState(surface={}, source={})

    with pytest.warns(RuntimeWarning, match="No geometry rendered"):
        frame = build_frame_entry(
            simulation=sim,
            bounds=Bounds3D(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0)),
            tm=FakeTrimesh,
            pv=FakePyVista,
            primitive_resolution=16,
            shifts=shifts,
        )

    assert ("source marker", (1.0, 0.9, 0.2)) in frame.legend


def test_unknown_future_surface_type_warns():
    unknown_surface = SimpleNamespace(ID=123, type=999)
    region = SimpleNamespace(type="halfspace", A=unknown_surface, B=-1)

    with pytest.warns(RuntimeWarning, match="does not support surface"):
        mesh = region_mesh(
            region,
            Bounds3D(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(-1.0, 1.0)),
            FakeTrimesh,
            primitive_resolution=16,
            surface_shift_map={},
        )

    assert mesh is None
