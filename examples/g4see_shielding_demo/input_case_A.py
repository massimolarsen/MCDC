from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc


MM_TO_CM = 0.1
ENERGY_BINS_MEV = np.concatenate(([0.0], np.logspace(-6.0, np.log10(15.0), 19)))
MU_BINS = np.linspace(-1.0, 1.0, 20)


def box_region(name, x_span_cm, y_span_cm, z_span_cm, boundary="none"):
    xmin = mcdc.Surface.PlaneX(x=x_span_cm[0], name=f"{name}_xmin", boundary_condition=boundary)
    xmax = mcdc.Surface.PlaneX(x=x_span_cm[1], name=f"{name}_xmax", boundary_condition=boundary)
    ymin = mcdc.Surface.PlaneY(y=y_span_cm[0], name=f"{name}_ymin", boundary_condition=boundary)
    ymax = mcdc.Surface.PlaneY(y=y_span_cm[1], name=f"{name}_ymax", boundary_condition=boundary)
    zmin = mcdc.Surface.PlaneZ(z=z_span_cm[0], name=f"{name}_zmin", boundary_condition=boundary)
    zmax = mcdc.Surface.PlaneZ(z=z_span_cm[1], name=f"{name}_zmax", boundary_condition=boundary)
    region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax
    return {
        "region": region,
        "xmin": xmin,
    }


vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"H1": 0.0})
aluminum = mcdc.Material(name="Aluminum", nuclide_composition={"Al27": 0.06022694744})
fr4 = mcdc.Material(
    name="FR4",
    nuclide_composition={
        "H1": 0.00755003747,
        "C12": 0.02594406027,
        "O16": 0.02843711896,
        "Si28": 0.00992531514,
    },
)

world = box_region("world", (-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0), boundary="vacuum")
wall = box_region("wall", (-10.0, -9.7), (-15.0, 15.0), (-15.0, 15.0))
box_outer = box_region("box_outer", (-3.0, 3.0), (-7.0, 7.0), (-5.0, 5.0))
box_inner = box_region("box_inner", (-2.8, 2.8), (-6.8, 6.8), (-4.8, 4.8))
pcb = box_region("pcb", (-0.08, 0.08), (-7.0, 7.0), (-4.5, 4.5))
sram = box_region("sram", (-0.2, 0.4), (-1.2, 1.2), (-1.2, 1.2))

shell_region = box_outer["region"] & ~box_inner["region"]
pcb_region = pcb["region"] & ~sram["region"]
vacuum_region = world["region"] & ~wall["region"] & ~shell_region & ~pcb_region & ~sram["region"]

mcdc.Cell(name="Outer Wall", region=wall["region"], fill=aluminum)
mcdc.Cell(name="Electronics Shell", region=shell_region, fill=aluminum)
mcdc.Cell(name="PCB", region=pcb_region, fill=fr4)
mcdc.Cell(name="SRAM", region=sram["region"], fill=vacuum)
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

mcdc.TallySurface(
    name="handoff_energy",
    surface=sram["xmin"],
    scores=["net-current"],
    energy=ENERGY_BINS_MEV,
)
mcdc.TallySurface(
    name="handoff_angle",
    surface=sram["xmin"],
    scores=["net-current"],
    mu=MU_BINS,
    polar_reference=[1.0, 0.0, 0.0],
)

mesh = mcdc.MeshStructured(
    x=np.linspace(-1.0, 1.0, 21),
    y=np.linspace(-3.0, 3.0, 25),
    z=np.linspace(-3.0, 3.0, 25),
)
#mcdc.TallyMesh(name="local_flux", mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 1000
mcdc.settings.output_name = "case_A_none"

# Uncomment to save a geometry visualization before transport:
from mcdc import simulation
from mcdc.visualization.geometry import geo_viewer_3d
geo_viewer_3d(simulation=simulation, save_animation_path=str(Path(__file__).with_name("case_A_geometry.gif")),)

#mcdc.run()
