from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc

ENERGY_BINS_MEV = np.concatenate(([0.0], np.logspace(-6.0, np.log10(15.0), 19)))
MU_BINS = np.linspace(-1.0, 1.0, 5)
AZI_BINS = np.linspace(-np.pi, np.pi, 5)


def box_region(name, x_span_cm, y_span_cm, z_span_cm, boundary="none"):
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
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


# Keep nuclides consistent with available local CE library.
vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"Si28": 0.0})
aluminum = mcdc.Material(name="Aluminum", nuclide_composition={"Si28": 0.06022694744})
fr4 = mcdc.Material(
    name="FR4",
    nuclide_composition={
        "O16": 0.02843711896,
        "Si28": 0.00992531514,
    },
)

world = box_region(
    "world", (-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0), boundary="vacuum"
)
wall = box_region("wall", (-10.0, -9.7), (-15.0, 15.0), (-15.0, 15.0))
box_outer = box_region("box_outer", (-3.0, 3.0), (-7.0, 7.0), (-5.0, 5.0))
box_inner = box_region("box_inner", (-2.8, 2.8), (-6.8, 6.8), (-4.8, 4.8))
pcb = box_region("pcb", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5))
sram = box_region("sram", (-1.0, 1.0), (-4.2, 4.2), (-4.2, 4.2))

shield_thickness = 2.0
pcb_shield_inner = box_region(
    "pcb_shield_inner", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5)
)
pcb_shield_outer = box_region(
    "pcb_shield_outer",
    (-0.08 - shield_thickness, 0.08 + shield_thickness),
    (-6.8, 6.8),
    (-4.5, 4.5),
)

shell_region = box_outer["region"] & ~box_inner["region"]
pcb_region = pcb["region"] & ~sram["region"]
pcb_shield_region = (pcb_shield_outer["region"] & ~pcb_shield_inner["region"]) & ~sram[
    "region"
]
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
    energy=14.0,
)

# Geant4-oriented handoff source distribution over SRAM boundary crossings.
mcdc.Tally(
    name="sram_handoff_source_current_in",
    cell=sram_cell,
    scores=["current-in"],
    surface_mesh=(4, 4),
    mu=MU_BINS,
    azi=AZI_BINS,
    energy=ENERGY_BINS_MEV,
)

mcdc.settings.N_particle = 1000
mcdc.settings.output_name = "cube_in_cube_handoff_source_tally"

mcdc.run()
