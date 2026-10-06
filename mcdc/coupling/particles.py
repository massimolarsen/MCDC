"""MC/DC particle types handed to Geant4, and their PDG codes.

Kept apart from geant4_config, which the standalone Geant4 worker imports:
importing mcdc there would initialize MPI inside the worker subprocess.
"""

from mcdc.constant import PARTICLE_ELECTRON, PARTICLE_NEUTRON, PARTICLE_PROTON

PARTICLE_PDG = {
    PARTICLE_NEUTRON: 2112,
    PARTICLE_ELECTRON: 11,
    PARTICLE_PROTON: 2212,
}
PARTICLE_TYPE_BY_NAME = {
    "neutron": PARTICLE_NEUTRON,
    "electron": PARTICLE_ELECTRON,
    "proton": PARTICLE_PROTON,
}
