from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mcdc


ENERGY_BINS_MEV = np.concatenate(([0.0], np.logspace(-6.0, np.log10(15.0), 19)))
MU_BINS = np.linspace(-1.0, 1.0, 20)

# Run this input in accelerated mode with:
# python input_case_D.py --mode=numba --caching --N_particle=100000
# The --mode flag is handled by mcdc.config at startup.
# Without --caching, MC/DC clears the Numba cache each run and the initial
# compile can look like a hang for this continuous-energy case.


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
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "zmin": zmin,
        "zmax": zmax,
    }


vacuum = mcdc.Material(name="Vacuum", nuclide_composition={"H1": 0.0})
aluminum = mcdc.Material(name="Aluminum", nuclide_composition={"Al27": 0.06022694744})
tungsten = mcdc.Material(
    name="Tungsten",
    # Natural tungsten at 19.3 g/cc converted to atom densities in atoms/barn-cm.
    nuclide_composition={
        "W182": 0.01675169470,
        "W183": 0.00904446297,
        "W184": 0.01936195834,
        "W186": 0.01795885304,
    },
)
polyethylene = mcdc.Material(
    name="Polyethylene",
    # CH2 at 0.95 g/cc converted to atom densities in atoms/barn-cm.
    nuclide_composition={
        "H1": 0.08157725817,
        "C12": 0.04078862909,
    },
)
fr4 = mcdc.Material(
    name="FR4",
    # MC/DC continuous-energy materials use nuclide atom densities in
    # atoms/barn-cm, not normalized fractions, so these entries do not sum to 1.
    nuclide_composition={
        "H1": 0.00755003747,
        #"H1": 0.055003747,
        "C12": 0.02594406027,
        "O16": 0.02843711896,
        "Si28": 0.00992531514,
    },
)

world = box_region("world", (-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0), boundary="vacuum")
wall = box_region("wall", (-10.0, -9.7), (-15.0, 15.0), (-15.0, 15.0))
box_outer = box_region("box_outer", (-3.0, 3.0), (-7.0, 7.0), (-5.0, 5.0))
box_inner = box_region("box_inner", (-2.8, 2.8), (-6.8, 6.8), (-4.8, 4.8))
pcb = box_region("pcb", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5))
sram = box_region("sram", (-0.2, 0.4), (-1.2, 1.2), (-1.2, 1.2))
handoff_probe = box_region("handoff_probe", (-0.2000001, -0.2), (-1.2, 1.2), (-1.2, 1.2))

shield_thickness = 2.0
pcb_shield_inner = box_region("pcb_shield_inner", (-0.08, 0.08), (-6.8, 6.8), (-4.5, 4.5))
pcb_shield_outer = box_region(
    "pcb_shield_outer",
    (-0.08 - shield_thickness, 0.08 + shield_thickness),
    (-6.8, 6.8),
    (-4.5, 4.5),
)

shell_region = box_outer["region"] & ~box_inner["region"]
pcb_region = pcb["region"] & ~sram["region"]
pcb_shield_region = (
    (pcb_shield_outer["region"] & ~pcb_shield_inner["region"])
    & ~sram["region"]
    & ~handoff_probe["region"]
)
vacuum_region = (
    world["region"]
    & ~wall["region"]
    & ~shell_region
    & ~pcb_region
    & ~pcb_shield_region
    & ~handoff_probe["region"]
    & ~sram["region"]
)

mcdc.Cell(name="Outer Wall", region=wall["region"], fill=aluminum)
mcdc.Cell(name="Electronics Shell", region=shell_region, fill=aluminum)
mcdc.Cell(name="PCB", region=pcb_region, fill=fr4)
mcdc.Cell(name="Local Shield", region=pcb_shield_region, fill=polyethylene)
handoff_probe_cell = mcdc.Cell(name="Handoff probe", region=handoff_probe["region"], fill=vacuum)
sram_cell = mcdc.Cell(name="Handoff zone", region=sram["region"], fill=vacuum)
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

mcdc.TallyCell(
    name="sram_energy",
    cell=sram_cell,
    scores=["flux"],
    energy=ENERGY_BINS_MEV,
)
mcdc.TallyCell(
    name="sram_angle",
    cell=sram_cell,
    scores=["flux"],
    mu=MU_BINS,
    polar_reference=[1.0, 0.0, 0.0],
)

mcdc.TallyCell(
    name="face_energy",
    cell=handoff_probe_cell,
    scores=["flux"],
    energy=ENERGY_BINS_MEV,
)

mcdc.TallySurface(
    name="face_angle",
    surface=sram["xmin"],
    scores=["net-current"],
    mu=MU_BINS,
    polar_reference=[1.0, 0.0, 0.0],
)

# Case D is one of the primary coupling cases because it uses volumetric
# flux tallies in the SRAM region rather than a surface net-current handoff.

mesh = mcdc.MeshStructured(
    x=np.linspace(-1.0, 1.0, 21),
    y=np.linspace(-3.0, 3.0, 25),
    z=np.linspace(-3.0, 3.0, 25),
)
#mcdc.TallyMesh(name="local_flux", mesh=mesh, scores=["flux"])

mcdc.settings.N_particle = 10000
mcdc.settings.output_name = "case_E_polyethelene_shield"

# Uncomment to save a geometry visualization before transport:
from mcdc import simulation
from mcdc.visualization.geometry import geo_viewer_3d
#geo_viewer_3d(simulation=simulation,save_animation_path=str(Path(__file__).with_name("case_D_geometry.gif")))

mcdc.run()
