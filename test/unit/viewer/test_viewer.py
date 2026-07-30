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
from mcdc.viewer.render import (
    add_global_opacity_slider,
    build_frame_entry,
    cell_display_properties,
    render_frame_entry,
)
from mcdc.viewer.time_view import geo_viewer_3d_time
from mcdc.viewer.types import CellVisual, FrameEntry, ShiftState


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
    center = (0.5, 0.5, 0.5)


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

    geo_viewer_3d(SimpleNamespace(surfaces=[], sources=[]), labels=True)
    geo_viewer_3d(
        SimpleNamespace(surfaces=[], sources=[]),
        time_steps=[0.0, 1.0],
        labels=True,
    )

    moving_surface = SimpleNamespace(
        moving=True,
        move_time_grid=np.array([0.0, 1.0, np.inf]),
    )
    geo_viewer_3d(SimpleNamespace(surfaces=[moving_surface], sources=[]), labels=True)

    assert [name for name, _ in calls] == ["static", "time", "time"]
    assert all(call[1]["labels"] is True for call in calls)


class FakePlotter:
    def __init__(self):
        self.labels = []
        self.meshes = []
        self.removed = []
        self.legend = None
        self.text = []
        self.key_events = {}
        self.slider_callback = None
        self.slider_args = None
        self.slider_kwargs = None
        self.render_count = 0

    def remove_actor(self, actor_name, reset_camera=False):
        self.removed.append(actor_name)

    def add_axes(self):
        return None

    def show_grid(self):
        return None

    def add_mesh(self, mesh, **kwargs):
        actor = FakeActor()
        self.meshes.append((mesh, kwargs))
        return actor

    def add_point_labels(self, points, labels, **kwargs):
        self.labels.append((points, labels, kwargs))

    def add_legend(self, legend, **kwargs):
        self.legend = legend

    def add_text(self, *args, **kwargs):
        self.text.append((args, kwargs))
        return None

    def add_slider_widget(self, callback, *args, **kwargs):
        self.slider_callback = callback
        self.slider_args = args
        self.slider_kwargs = kwargs

    def add_key_event(self, key, callback):
        self.key_events[key] = callback

    def render(self):
        self.render_count += 1

    def show(self, **kwargs):
        return None


class FakeProperty:
    def __init__(self):
        self.opacity = None

    def SetOpacity(self, value):
        self.opacity = value


class FakeActor:
    def __init__(self):
        self.property = FakeProperty()

    def GetProperty(self):
        return self.property


def cell_visual():
    return CellVisual(
        actor_name="cell_0",
        mesh=SimpleNamespace(center=(1.0, 2.0, 3.0)),
        color=(0.1, 0.2, 0.3),
        opacity=0.6,
        label="fuel",
    )


def single_cell_frame():
    return FrameEntry(
        cells=[cell_visual()],
        sources=[],
        legend=[("fuel", (0.1, 0.2, 0.3))],
        label=None,
        has_geometry=True,
    )


def test_render_frame_entry_adds_cell_labels_when_enabled():
    plotter = FakePlotter()
    frame = single_cell_frame()

    rendered, actor_names = render_frame_entry(
        plotter,
        frame,
        actor_names=[],
        labels=True,
    )

    assert rendered
    assert "mcdc_cell_labels" in actor_names
    assert plotter.labels[0][1] == ["fuel"]
    np.testing.assert_allclose(plotter.labels[0][0], [[1.0, 2.0, 3.0]])


def test_render_frame_entry_skips_cell_labels_by_default():
    plotter = FakePlotter()
    frame = single_cell_frame()

    _, actor_names = render_frame_entry(plotter, frame, actor_names=[])

    assert "mcdc_cell_labels" not in actor_names
    assert plotter.labels == []


def test_material_color_mode_uses_same_color_for_same_material():
    cells = [
        SimpleNamespace(ID=0, name="a", fill=SimpleNamespace(name="Silicon")),
        SimpleNamespace(ID=1, name="b", fill=SimpleNamespace(name="Silicon")),
        SimpleNamespace(ID=2, name="c", fill=SimpleNamespace(name="Copper")),
    ]
    props = cell_display_properties(SimpleNamespace(cells=cells), color_by="material")

    assert props[0]["color"] == props[1]["color"]
    assert props[0]["color"] != props[2]["color"]
    assert props[0]["legend"] == "Silicon"


def test_material_color_mode_keeps_first_palette_materials_distinct():
    cells = [
        SimpleNamespace(ID=index, name=f"cell_{index}", fill=SimpleNamespace(name=name))
        for index, name in enumerate(
            [
                "Aluminum",
                "Bakelite",
                "Copper",
                "Epoxy",
                "FR4",
                "Galactic",
                "LithiumCobaltOxide",
                "Silicon",
                "Steel",
                "Tantalum",
            ]
        )
    ]
    props = cell_display_properties(SimpleNamespace(cells=cells), color_by="material")

    assert len({props[index]["color"] for index in range(len(cells))}) == len(cells)


def test_cell_color_mode_keeps_cell_palette():
    cells = [
        SimpleNamespace(ID=0, name="a", fill=SimpleNamespace(name="Silicon")),
        SimpleNamespace(ID=1, name="b", fill=SimpleNamespace(name="Silicon")),
    ]
    props = cell_display_properties(SimpleNamespace(cells=cells), color_by="cell")

    assert props[0]["color"] != props[1]["color"]
    assert props[0]["legend"] == "a"


def test_time_view_only_binds_arrow_step_keys(monkeypatch):
    plotter = FakePlotter()

    class FakePyVistaWithPlotter(FakePyVista):
        @staticmethod
        def Plotter():
            return plotter

    monkeypatch.setattr(
        "mcdc.viewer.time_view.import_backends",
        lambda: (FakePyVistaWithPlotter, FakeTrimesh),
    )
    sim = SimpleNamespace(cells=[], sources=[], surfaces=[])

    with pytest.warns(RuntimeWarning, match="No geometry rendered"):
        geo_viewer_3d_time(
            sim,
            time_steps=[0.0],
            opacity_slider=False,
        )

    assert "Right" in plotter.key_events
    assert "Left" in plotter.key_events
    assert "p" not in plotter.key_events
    assert "n" not in plotter.key_events


def test_global_opacity_slider_updates_cell_actor_opacity():
    plotter = FakePlotter()
    render_state = SimpleNamespace(opacity_scale=1.0)
    render_frame_entry(plotter, single_cell_frame(), actor_names=[])
    actor = plotter._mcdc_cell_actors[0][0]

    add_global_opacity_slider(plotter, render_state)
    plotter.slider_callback(0.5)

    assert plotter.slider_args == ((0.02, 1.0),)
    assert plotter.slider_kwargs["pointa"] == (0.68, 0.08)
    assert plotter.slider_kwargs["pointb"] == (0.96, 0.08)
    assert plotter.slider_kwargs["style"] == "modern"
    assert plotter.slider_kwargs["fmt"] == "%.2f"
    assert render_state.opacity_scale == 0.5
    assert actor.property.opacity == pytest.approx(0.3)
    assert plotter.render_count == 1


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
