import h5py
import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.main import preparation
from mcdc.output import create_tally_dataset
from mcdc.transport.tally import score as score_module


def _box_cell(material_mg):
    xmin = mcdc.Surface.PlaneX(x=-1.0)
    xmax = mcdc.Surface.PlaneX(x=1.0)
    ymin = mcdc.Surface.PlaneY(y=-2.0)
    ymax = mcdc.Surface.PlaneY(y=2.0)
    zmin = mcdc.Surface.PlaneZ(z=-3.0)
    zmax = mcdc.Surface.PlaneZ(z=3.0)
    cell = mcdc.Cell(
        region=+xmin & -xmax & +ymin & -ymax & +zmin & -zmax,
        fill=material_mg,
    )
    return {
        "cell": cell,
        "surfaces": [xmin, xmax, ymin, ymax, zmin, zmax],
    }


def _particle(surface_ID, x, y, z, ux, uy, uz, w=2.0):
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]
    particle["alive"] = True
    particle["particle_type"] = PARTICLE_NEUTRON
    particle["surface_ID"] = surface_ID
    particle["material_ID"] = 0
    particle["g"] = 0
    particle["x"] = x
    particle["y"] = y
    particle["z"] = z
    particle["t"] = 0.0
    particle["ux"] = ux
    particle["uy"] = uy
    particle["uz"] = uz
    particle["w"] = w
    return particle_container


def _surface_mesh_bin_offset(surface_tally, mcdc_struct, face, u, v, score=0):
    tally_base = mcdc_struct["tallies"][surface_tally["parent_ID"]]
    return (
        tally_base["bin_offset"]
        + face * surface_tally["surface_mesh_stride_face"]
        + u * surface_tally["surface_mesh_stride_u"]
        + v * surface_tally["surface_mesh_stride_v"]
        + score
    )


def _surface_mesh_sum_offset(surface_tally, mcdc_struct, face, u, v, score=0):
    tally_base = mcdc_struct["tallies"][surface_tally["parent_ID"]]
    return (
        tally_base["bin_sum_offset"]
        + face * surface_tally["surface_mesh_stride_face"]
        + u * surface_tally["surface_mesh_stride_u"]
        + v * surface_tally["surface_mesh_stride_v"]
        + score
    )


def _score_current_in(surface_tally, mcdc_struct, data, surface, particle_container):
    surface_struct = mcdc_struct["surfaces"][surface.ID]
    target_cell_ID = surface_tally["spatial_filter_ID"]
    score_module.surface_tally(
        particle_container,
        surface_struct,
        surface_tally,
        -1,
        target_cell_ID,
        mcdc_struct,
        data,
    )


def test_cell_current_surface_mesh_constructor_fields(material_mg):
    box = _box_cell(material_mg)
    tally_obj = mcdc.Tally(
        cell=box["cell"],
        scores=["current-in"],
        surface_mesh=(2, 3),
    )

    assert tally_obj.bin_shape == [1, 1, 1, 1, 6, 2, 3, 1]

    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    surface_tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    assert surface_tally["use_surface_mesh"] == 1
    assert surface_tally["surface_mesh_Nu"] == 2
    assert surface_tally["surface_mesh_Nv"] == 3
    assert surface_tally["surface_mesh_stride_v"] == 1
    assert surface_tally["surface_mesh_stride_u"] == 3
    assert surface_tally["surface_mesh_stride_face"] == 6
    assert surface_tally["surface_mesh_x_min"] == -1.0
    assert surface_tally["surface_mesh_x_max"] == 1.0
    assert surface_tally["surface_mesh_y_min"] == -2.0
    assert surface_tally["surface_mesh_y_max"] == 2.0
    assert surface_tally["surface_mesh_z_min"] == -3.0
    assert surface_tally["surface_mesh_z_max"] == 3.0

    assert data is not None


def test_cell_current_surface_mesh_face_order(material_mg):
    box = _box_cell(material_mg)
    tally_obj = mcdc.Tally(
        cell=box["cell"],
        scores=["current-in"],
        surface_mesh=(1, 1),
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    surface_tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    crossings = [
        (0, box["surfaces"][0], (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (1, box["surfaces"][1], (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        (2, box["surfaces"][2], (0.0, -2.0, 0.0), (0.0, 1.0, 0.0)),
        (3, box["surfaces"][3], (0.0, 2.0, 0.0), (0.0, -1.0, 0.0)),
        (4, box["surfaces"][4], (0.0, 0.0, -3.0), (0.0, 0.0, 1.0)),
        (5, box["surfaces"][5], (0.0, 0.0, 3.0), (0.0, 0.0, -1.0)),
    ]

    for face, surface, position, direction in crossings:
        particle_container = _particle(surface.ID, *position, *direction)
        _score_current_in(surface_tally, mcdc_struct, data, surface, particle_container)
        idx = _surface_mesh_bin_offset(surface_tally, mcdc_struct, face, 0, 0)
        assert np.isclose(data[idx], 2.0)


@pytest.mark.parametrize(
    "surface_index, position, direction, expected",
    [
        (0, (-1.0, -1.0, 2.0), (1.0, 0.0, 0.0), (0, 0, 2)),
        (2, (0.75, -2.0, -2.5), (0.0, 1.0, 0.0), (2, 1, 0)),
        (4, (-0.75, 1.9, -3.0), (0.0, 0.0, 1.0), (4, 0, 2)),
    ],
    ids=["x_face_yz", "y_face_xz", "z_face_xy"],
)
def test_cell_current_surface_mesh_local_uv_bins(
    material_mg,
    surface_index,
    position,
    direction,
    expected,
):
    box = _box_cell(material_mg)
    tally_obj = mcdc.Tally(
        cell=box["cell"],
        scores=["current-in"],
        surface_mesh=(2, 3),
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    surface_tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    surface = box["surfaces"][surface_index]
    particle_container = _particle(surface.ID, *position, *direction)
    _score_current_in(surface_tally, mcdc_struct, data, surface, particle_container)

    face, u, v = expected
    idx = _surface_mesh_bin_offset(surface_tally, mcdc_struct, face, u, v)
    assert np.isclose(data[idx], 2.0)


def test_cell_current_surface_mesh_rejects_invalid_selectors(material_mg, capsys):
    box = _box_cell(material_mg)

    with pytest.raises(SystemExit):
        mcdc.Tally(
            surface=box["surfaces"][0],
            scores=["net-current"],
            surface_mesh=(2, 2),
        )
    out = capsys.readouterr().out
    assert "surface_mesh is supported only for cell-filtered current tallies." in out

    with pytest.raises(SystemExit):
        mcdc.Tally(cell=box["cell"], scores=["flux"], surface_mesh=(2, 2))
    out = capsys.readouterr().out
    assert "surface_mesh is supported only for cell-filtered current tallies." in out


def test_cell_current_surface_mesh_rejects_non_box_cell(material_mg, capsys):
    cylinder = mcdc.Surface.CylinderZ(center=(0.0, 0.0), radius=1.0)
    cell = mcdc.Cell(region=-cylinder, fill=material_mg)

    with pytest.raises(SystemExit):
        mcdc.Tally(cell=cell, scores=["current-in"], surface_mesh=(2, 2))
    out = capsys.readouterr().out
    assert "surface_mesh currently supports only axis-aligned box cells" in out


def test_cell_current_surface_mesh_hdf5_output_metadata(material_mg, tmp_path):
    box = _box_cell(material_mg)
    tally_obj = mcdc.Tally(
        name="handoff_source",
        cell=box["cell"],
        scores=["current-in"],
        surface_mesh=(2, 3),
    )
    mcdc_container, data = preparation()
    mcdc_struct = mcdc_container[0]
    surface_tally = mcdc_struct["surface_tallies"][tally_obj.child_ID]

    idx = _surface_mesh_sum_offset(surface_tally, mcdc_struct, 1, 1, 2)
    data[idx] = 7.0

    output_path = tmp_path / "surface_mesh_tally.h5"
    with h5py.File(output_path, "w") as file:
        create_tally_dataset(file, mcdc_struct, data)

    with h5py.File(output_path, "r") as file:
        group = file["tallies/handoff_source"]
        faces = [face.decode("ascii") for face in group["grid/face"][:]]
        assert faces == ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        assert group["grid/u"].shape == (6, 3)
        assert group["grid/v"].shape == (6, 4)
        assert np.allclose(group["grid/u"][0], [-2.0, 0.0, 2.0])
        assert np.allclose(group["grid/u"][2], [-1.0, 0.0, 1.0])
        assert np.allclose(group["grid/v"][0], [-3.0, -1.0, 1.0, 3.0])

        mean = group["current-in/mean"][:]
        assert mean.shape == (1, 1, 1, 1, 6, 2, 3)
        assert mean[0, 0, 0, 0, 1, 1, 2] == 7.0
