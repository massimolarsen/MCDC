import numpy as np

import mcdc
import mcdc.numba_types as type_
from mcdc.constant import PARTICLE_ELECTRON, PARTICLE_NEUTRON
from mcdc.transport.tally.filter import get_filter_indices


def test_particle_type_filter(prepare_simulation):
    tally_object = mcdc.Tally(scores=["flux"], particle_type="electron")
    simulation_container, data = prepare_simulation(tallies=[tally_object])
    tally = simulation_container[0]["tallies"][tally_object.ID]
    particle_container = np.zeros(1, type_.particle)

    particle_container[0]["particle_type"] = PARTICLE_ELECTRON
    assert get_filter_indices(particle_container, tally, data) == (0, 0, 0, 0)
    particle_container[0]["particle_type"] = PARTICLE_NEUTRON
    assert get_filter_indices(particle_container, tally, data) == (-1, -1, -1, -1)
