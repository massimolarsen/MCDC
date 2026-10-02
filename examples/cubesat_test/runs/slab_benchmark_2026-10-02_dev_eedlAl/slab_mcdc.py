"""Electron slab benchmark: pencil beam -> Al6061 slab -> Si detector (MC/DC).

Port of ../slab_benchmark_2026-09-28/slab_mcdc.py to the upstream-dev API used by
PR #548 (mcdc.Simulation, no surface_mesh). Geometry, materials, source and energy
bins are unchanged.

Geometry (cm): beam starts at x=-0.5 heading +x.
  slab     : Al6061, x in [0, T], y,z in [-10, 10]   (T = 0.3 = CubeSat shear panel)
  detector : Si, x in [T+0.7, T+2.5], y,z in [-2.25, 2.25] (ADCS SV size, 0.7 cm gap)
  void elsewhere; vacuum outer boundary.
Env: E [eV], T [cm], N, OUT
"""
import os, time
import numpy as np
import mcdc

E = float(os.environ["E"]); T = float(os.environ.get("T", "0.3"))
N = int(os.environ.get("N", "1000")); OUT = os.environ["OUT"]

simulation = mcdc.Simulation("slab")

m_void = mcdc.Material(name="Void", element_composition={"Si": 0.0})
# Al6061 exactly as in inputE_SUBMIT.py
if os.environ.get("PURE_AL") == "1":
    # Isolation case: Al only (Mg/Si have only EPRDATA14 data with the 0.256-10 MeV gap)
    m_al = mcdc.Material(name="Al", element_composition={"Al": 0.059057360465045})
else:
    m_al = mcdc.Material(name="Al6061", element_composition={
        "Al": 0.059057360465045, "Mg": 0.00081349602416576, "Si": 0.00046494828605733})
m_si = mcdc.Material(name="Silicon", element_composition={"Si": 0.050132618436459})

def box(x0, x1, y0, y1, z0, z1, bc="none"):
    s = [mcdc.Surface.PlaneX(x=x0, boundary_condition=bc), mcdc.Surface.PlaneX(x=x1, boundary_condition=bc),
         mcdc.Surface.PlaneY(y=y0, boundary_condition=bc), mcdc.Surface.PlaneY(y=y1, boundary_condition=bc),
         mcdc.Surface.PlaneZ(z=z0, boundary_condition=bc), mcdc.Surface.PlaneZ(z=z1, boundary_condition=bc)]
    return +s[0] & -s[1] & +s[2] & -s[3] & +s[4] & -s[5], s

G = 0.7; DX = 1.8; H = 2.25
outer, _ = box(-1.0, T + G + DX + 1.0, -11.0, 11.0, -11.0, 11.0, bc="vacuum")
slab_r, slab_s = box(0.0, T, -10.0, 10.0, -10.0, 10.0)
det_r, det_s = box(T + G, T + G + DX, -H, H, -H, H)
slab = mcdc.Cell(region=slab_r, fill=m_al)
det = mcdc.Cell(region=det_r, fill=m_si)
void = mcdc.Cell(region=outer & ~slab_r & ~det_r, fill=m_void)
# Workaround for upstream dev: a cell-only surface tally reads cell.surfaces before
# the cell is compiled. These are the same bounding surfaces the compile step lists.
det.surfaces = det_s
simulation.set_model([slab, det, void])

source = mcdc.Source(position=[-0.5, 0.0, 0.0], direction=[1.0, 0.0, 0.0], energy=E, particle_type="electron")
simulation.set_sources([source])

ebins = np.array([0.0, 1.0e3, 1.0e4, 5.0e4, 1.0e5, 2.5e5, 5.0e5, 1.0e6, 2.0e6, 4.0e6, 6.0e6, 8.0e6, 1.2e7])
# x-min face = backscatter out, x-max face = transmitted out (dev has no surface_mesh)
simulation.set_tallies([
    mcdc.Tally(name="detector", cell=det, scores=["current-in"], energy=ebins),
    mcdc.Tally(name="slab_back", cell=slab, surface=slab_s[0], scores=["current-out"], energy=ebins),
    mcdc.Tally(name="slab_trans", cell=slab, surface=slab_s[1], scores=["current-out"], energy=ebins),
])

# Electrons are activated from the source (set_transported_particles removed in PR #548 8984ccf1)
simulation.settings.N_particle = N
simulation.settings.active_bank_buffer = 100000
simulation.settings.output_name = OUT
simulation.settings.use_progress_bar = False
t0 = time.time()
simulation.run()
print(f"WALLTIME {OUT} {time.time() - t0:.1f} s N={N}")
