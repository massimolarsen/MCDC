import numpy as np

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.transport.simulation import surface_crossing


def _slab_with_handoff_cell():
    material = mcdc.Material.multigroup(capture=np.array([1.0]))
    s_left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    s_mid = mcdc.Surface.PlaneX(x=0.0)
    s_right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    outside = mcdc.Cell(region=+s_left & -s_mid, fill=material)
    device = mcdc.Cell(region=+s_mid & -s_right, fill=material, handoff=True)
    return outside, device, s_mid


def _crossing_particle(surface_ID, ux, cell_ID):
    particle_container = np.zeros(1, type_.particle)
    particle = particle_container[0]
    particle["alive"] = True
    particle["particle_type"] = PARTICLE_NEUTRON
    particle["surface_ID"] = surface_ID
    particle["cell_ID"] = cell_ID
    particle["ux"] = ux
    particle["E"] = 1.0
    particle["w"] = 0.5
    return particle_container


def test_entering_handoff_cell_banks_once_and_kills(prepare_simulation):
    outside, device, s_mid = _slab_with_handoff_cell()
    mcdc_container, data = prepare_simulation(cells=[outside, device])
    mcdc_struct = mcdc_container[0]
    assert mcdc_struct["N_handoff_cell"] == 1

    particle_container = _crossing_particle(s_mid.ID, 1.0, outside.ID)
    surface_crossing(particle_container, mcdc_struct, data)

    bank = mcdc_struct["bank_handoff"]
    assert bank["size"][0] == 1
    assert not particle_container[0]["alive"]
    banked = bank["particle_data"][0]
    assert banked["w"] == 0.5
    assert banked["ux"] == 1.0


def test_leaving_handoff_cell_is_not_banked(prepare_simulation):
    outside, device, s_mid = _slab_with_handoff_cell()
    mcdc_container, data = prepare_simulation(cells=[outside, device])
    mcdc_struct = mcdc_container[0]

    particle_container = _crossing_particle(s_mid.ID, -1.0, device.ID)
    surface_crossing(particle_container, mcdc_struct, data)

    assert mcdc_struct["bank_handoff"]["size"][0] == 0
    assert particle_container[0]["alive"]
