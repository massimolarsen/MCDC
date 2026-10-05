import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.transport.util import find_bin, make_direction_basis


@njit
def evaluate_neutron_xs_energy_grid(e, nuclide, data):
    energy_grid = mcdc_get.nuclide.neutron_xs_energy_grid_all(nuclide, data)

    idx = find_bin(e, energy_grid)

    # Off-grid energy: use the edge bin instead of indexing out of bounds
    if idx == -1:
        if e < energy_grid[0]:
            idx = 0
        else:
            idx = len(energy_grid) - 2

    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def evaluate_electron_xs_energy_grid(e, element, data):
    energy_grid = mcdc_get.element.electron_xs_energy_grid_all(element, data)
    idx = find_bin(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def evaluate_proton_xs_energy_grid(e, nuclide, data):
    offset = nuclide["proton_xs_energy_grid_offset"]
    length = nuclide["proton_xs_energy_grid_length"]
    energy_grid = data[offset : offset + length]

    idx = find_bin(e, energy_grid)
    e0 = energy_grid[idx]
    e1 = energy_grid[idx + 1]
    return idx, e0, e1


@njit
def scatter_direction(ux, uy, uz, mu0, azi):
    """Rotate a unit direction by polar cosine mu0 and azimuth azi in radians."""
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    sin_polar = math.sqrt(max(0.0, 1.0 - mu0 * mu0))
    u1x, u1y, u1z, u2x, u2y, u2z = make_direction_basis(ux, uy, uz)

    ux_new = sin_polar * cos_azi * u1x + sin_polar * sin_azi * u2x + mu0 * ux
    uy_new = sin_polar * cos_azi * u1y + sin_polar * sin_azi * u2y + mu0 * uy
    uz_new = sin_polar * cos_azi * u1z + sin_polar * sin_azi * u2z + mu0 * uz

    return ux_new, uy_new, uz_new
