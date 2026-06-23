from pathlib import Path

import numpy as np


COUPLING_TEST_DIR = Path(__file__).resolve().parent


def mcdc_output_name(name):
    return f"mcdc_h5/{name}"


def geant4_output_path(filename):
    return str(COUPLING_TEST_DIR / "geant4_h5" / filename)


def bridge_build_dir():
    return str(Path(__file__).resolve().parents[3] / "couple_mcdc_g4" / "build")


def box_region(mcdc, name, x_span_cm, y_span_cm, z_span_cm, boundary="none"):
    xmin = mcdc.Surface.PlaneX(
        x=x_span_cm[0], name=f"{name}_xmin", boundary_condition=boundary
    )
    xmax = mcdc.Surface.PlaneX(
        x=x_span_cm[1], name=f"{name}_xmax", boundary_condition=boundary
    )
    ymin = mcdc.Surface.PlaneY(
        y=y_span_cm[0], name=f"{name}_ymin", boundary_condition=boundary
    )
    ymax = mcdc.Surface.PlaneY(
        y=y_span_cm[1], name=f"{name}_ymax", boundary_condition=boundary
    )
    zmin = mcdc.Surface.PlaneZ(
        z=z_span_cm[0], name=f"{name}_zmin", boundary_condition=boundary
    )
    zmax = mcdc.Surface.PlaneZ(
        z=z_span_cm[1], name=f"{name}_zmax", boundary_condition=boundary
    )
    region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
    return {
        "region": region,
        "x_span_cm": x_span_cm,
        "y_span_cm": y_span_cm,
        "z_span_cm": z_span_cm,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


def g4_world_size_from_handoff_zone(zone, padding_scale=1.2):
    zone_size_mm = (
        10.0 * (zone["x_span_cm"][1] - zone["x_span_cm"][0]),
        10.0 * (zone["y_span_cm"][1] - zone["y_span_cm"][0]),
        10.0 * (zone["z_span_cm"][1] - zone["z_span_cm"][0]),
    )
    return tuple(padding_scale * length_mm for length_mm in zone_size_mm)


def build_vacuum_handoff_model(mcdc, source_energy_ev, handoff=False):
    vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"Si28": 0.0})

    world = box_region(
        mcdc,
        "world",
        (-6.0, 6.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
        boundary="vacuum",
    )
    target = box_region(mcdc, "target", (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    outside_region = world["region"] & ~target["region"]

    target_cell = mcdc.Cell(
        name="Target Handoff Box",
        region=target["region"],
        fill=vacuum,
        handoff=handoff,
    )
    mcdc.Cell(name="Vacuum Outside Target", region=outside_region, fill=vacuum)

    mcdc.Source(
        name="Monoenergetic +x Neutron Source",
        x=[-5.0, -4.9],
        y=[-0.75, 0.75],
        z=[-0.75, 0.75],
        direction=[1.0, 0.0, 0.0],
        energy=source_energy_ev,
    )

    return target, target_cell


def build_cube_in_cube_model(mcdc, handoff=False):
    vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"Si28": 0.0})
    aluminum = mcdc.Material(
        name="Aluminum", nuclide_composition={"Si28": 0.06022694744}
    )
    fr4 = mcdc.Material(
        name="FR4",
        nuclide_composition={
            "O16": 0.02843711896,
            "Si28": 0.00992531514,
        },
    )

    world = box_region(
        mcdc,
        "world",
        (-30.0, 30.0),
        (-30.0, 30.0),
        (-30.0, 30.0),
        boundary="vacuum",
    )
    wall = box_region(mcdc, "wall", (-10.0, -9.7), (-15.0, 15.0), (-15.0, 15.0))
    box_outer = box_region(mcdc, "box_outer", (-3.0, 3.0), (-7.0, 7.0), (-5.0, 5.0))
    box_inner = box_region(mcdc, "box_inner", (-2.8, 2.8), (-6.8, 6.8), (-4.8, 4.8))
    pcb = box_region(mcdc, "pcb", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5))
    sram = box_region(mcdc, "sram", (-1.0, 1.0), (-4.2, 4.2), (-4.2, 4.2))

    shield_thickness = 2.0
    pcb_shield_inner = box_region(
        mcdc, "pcb_shield_inner", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5)
    )
    pcb_shield_outer = box_region(
        mcdc,
        "pcb_shield_outer",
        (-0.08 - shield_thickness, 0.08 + shield_thickness),
        (-6.8, 6.8),
        (-4.5, 4.5),
    )

    shell_region = box_outer["region"] & ~box_inner["region"]
    pcb_region = pcb["region"] & ~sram["region"]
    pcb_shield_region = (
        pcb_shield_outer["region"] & ~pcb_shield_inner["region"]
    ) & ~sram["region"]
    vacuum_region = (
        world["region"]
        & ~wall["region"]
        & ~shell_region
        & ~pcb_region
        & ~pcb_shield_region
        & ~sram["region"]
    )

    mcdc.Cell(name="Outer Wall", region=wall["region"], fill=aluminum)
    mcdc.Cell(name="Electronics Shell", region=shell_region, fill=aluminum)
    mcdc.Cell(name="PCB", region=pcb_region, fill=fr4)
    mcdc.Cell(name="Local Shield", region=pcb_shield_region, fill=aluminum)
    sram_cell = mcdc.Cell(
        name="Inner SRAM",
        region=sram["region"],
        fill=vacuum,
        handoff=handoff,
    )
    mcdc.Cell(name="Vacuum", region=vacuum_region, fill=vacuum)

    mcdc.Source(
        name="Incident Neutron Source",
        x=[-11.0, -10.9],
        y=[-15.0, 15.0],
        z=[-15.0, 15.0],
        direction=[1.0, 0.0, 0.0],
        polar_cosine=[0.0, 1.0],
        azimuthal=[0.0, 2.0 * np.pi],
        energy=14.0e6,
    )

    return sram, sram_cell
