# ======================================================================================
# Simulation building blocks
# ======================================================================================

# The simulation
from mcdc.object_.simulation import simulation

# The settings
settings = simulation.settings

# The objects
from mcdc.object_.cell import Cell, Universe, Lattice
from mcdc.object_.material import Material, MaterialMG
from mcdc.object_.mesh import MeshUniform, MeshStructured
from mcdc.object_.source import Source
from mcdc.object_.surface import Surface
from mcdc.object_.tally import Tally

# ======================================================================================
# Runners
# ======================================================================================

from mcdc.main import run
from mcdc.visualize import visualize

# ======================================================================================
# Misc.
# ======================================================================================

import mcdc.config
from mcdc.output import recombine_tallies
from mcdc.coupling import geant4_handoff as _geant4_handoff


def enable_geant4_handoff(**kwargs):
    _geant4_handoff.configure(enabled=True, **kwargs)


def disable_geant4_handoff():
    _geant4_handoff.configure(enabled=False)
    _geant4_handoff.reset()
