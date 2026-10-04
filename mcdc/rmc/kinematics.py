"""
Two-body kinematics for Residual Monte Carlo kernel evaluation.

Notation: E_in is the incident (lab) energy, E_out the outgoing (lab) energy,
mu_lab the lab scattering cosine, and (E_cm, mu_cm) the outgoing energy and
scattering cosine in the center-of-mass (COM) frame. Energies are in eV.

Two COM conventions mirror the MC/DC samplers exactly:
  - Inelastic and fission (`sample_inelastic_scattering`, `sample_fission`) use
    the classical MCNP relations (MCNP Eqs. 2.43-2.44).
  - Elastic scattering (`sample_elastic_scattering`) adds velocities using the
    relativistic speed-energy relation (`particle_speed`,
    `particle_energy_from_speed`).
"""

import math

from numba import njit

####

from mcdc.constant import LIGHT_SPEED, NEUTRON_MASS, PI

# ======================================================================================
# Classical COM <-> lab (inelastic and fission)
# ======================================================================================


@njit(cache=True)
def com_to_lab(E_in, E_cm, mu_cm, A):
    """COM (E_cm, mu_cm) to lab (E_out, mu_lab), as in `sample_inelastic_scattering`."""
    E_out = E_cm + (E_in + 2 * mu_cm * (A + 1) * math.sqrt(E_in * E_cm)) / (A + 1) ** 2
    mu_lab = mu_cm * math.sqrt(E_cm / E_out) + math.sqrt(E_in / E_out) / (A + 1)
    return E_out, mu_lab


@njit(cache=True)
def lab_to_com(E_in, E_out, mu_lab, A):
    """Inverse of `com_to_lab`. Returns E_cm <= 0 if the lab point is unreachable."""
    E_cm = (
        E_out + E_in / (A + 1) ** 2 - 2.0 * mu_lab * math.sqrt(E_in * E_out) / (A + 1)
    )
    if E_cm <= 0.0:
        return 0.0, 0.0
    mu_cm = (mu_lab * math.sqrt(E_out) - math.sqrt(E_in) / (A + 1)) / math.sqrt(E_cm)
    return E_cm, mu_cm


@njit(cache=True)
def com_to_lab_jacobian(E_out, E_cm):
    """
    Joint density factor f_L(E_out, mu_lab) = f_C(E_cm, mu_cm) * sqrt(E_out / E_cm).

    From translation invariance of the 3D velocity density and
    d^3v ~ sqrt(E) dE dmu dphi (the azimuth is shared by both frames).
    """
    return math.sqrt(E_out / E_cm)


@njit(cache=True)
def level_mu_cm(E_in, E_out, E_cm, A):
    """COM cosine giving lab E_out for a fixed E_cm (E_out is affine in mu_cm)."""
    return ((A + 1) ** 2 * (E_out - E_cm) - E_in) / (
        2.0 * (A + 1) * math.sqrt(E_in * E_cm)
    )


@njit(cache=True)
def level_dmu_cm_dE_out(E_in, E_cm, A):
    """|d mu_cm / d E_out| for a fixed E_cm."""
    return (A + 1) / (2.0 * math.sqrt(E_in * E_cm))


@njit(cache=True)
def level_E_out_range(E_in, E_cm_low, E_cm_high, A):
    """
    Lab E_out support for E_cm in [E_cm_low, E_cm_high], over all mu_cm.

    For fixed E_cm, E_out = E_cm + s^2 + 2 mu_cm s sqrt(E_cm) with s = sqrt(E_in)/(A+1),
    so E_out ranges over [(sqrt(E_cm) - s)^2, (sqrt(E_cm) + s)^2].
    """
    s = math.sqrt(E_in) / (A + 1)
    E_high = (math.sqrt(E_cm_high) + s) ** 2
    if E_cm_low > s * s:
        E_low = (math.sqrt(E_cm_low) - s) ** 2
    elif E_cm_high < s * s:
        E_low = (s - math.sqrt(E_cm_high)) ** 2
    else:
        E_low = 0.0
    return E_low, E_high


# ======================================================================================
# Relativistic-speed elastic kinematics (target at rest)
# ======================================================================================


@njit(cache=True)
def speed_squared(E):
    """Squared speed [cm^2/s^2] of a neutron with kinetic energy E, as `particle_speed`."""
    m = NEUTRON_MASS
    v = LIGHT_SPEED * math.sqrt(E * (E + 2.0 * m)) / (E + m)
    return v * v


@njit(cache=True)
def energy_from_speed_squared(v2):
    """Kinetic energy from squared speed, as `particle_energy_from_speed`."""
    beta2 = v2 / (LIGHT_SPEED * LIGHT_SPEED)
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    return NEUTRON_MASS * (gamma - 1.0)


@njit(cache=True)
def elastic_E_out(E_in, mu_cm, A):
    """Lab outgoing energy for COM cosine mu_cm: |v_out|^2 = v^2 (1+A^2+2A mu_cm)/(1+A)^2."""
    v2 = speed_squared(E_in)
    return energy_from_speed_squared(
        v2 * (1.0 + A * A + 2.0 * A * mu_cm) / (1.0 + A) ** 2
    )


@njit(cache=True)
def elastic_mu_cm(E_in, E_out, A):
    """COM cosine that yields lab E_out (inverse of `elastic_E_out`)."""
    ratio = speed_squared(E_out) / speed_squared(E_in)
    return ((1.0 + A) ** 2 * ratio - 1.0 - A * A) / (2.0 * A)


@njit(cache=True)
def elastic_mu_lab(mu_cm, A):
    """Lab scattering cosine for COM cosine mu_cm."""
    return (1.0 + A * mu_cm) / math.sqrt(1.0 + A * A + 2.0 * A * mu_cm)


@njit(cache=True)
def elastic_dmu_cm_dE_out(E_in, E_out, A):
    """|d mu_cm / d E_out|, with d(v^2)/dE = 2 c^2 m^2 / (E + m)^3."""
    m = NEUTRON_MASS
    dv2_dE = 2.0 * LIGHT_SPEED * LIGHT_SPEED * m * m / (E_out + m) ** 3
    return (1.0 + A) ** 2 / (2.0 * A * speed_squared(E_in)) * dv2_dE


@njit(cache=True)
def elastic_E_out_range(E_in, A):
    """Lab E_out support of target-at-rest elastic scattering: mu_cm in [-1, 1]."""
    return elastic_E_out(E_in, -1.0, A), E_in


# ======================================================================================
# Azimuthal integration about the slab axis
# ======================================================================================
# For fixed incident polar cosine mu_in and lab scattering cosine mu0, the outgoing
# polar cosine is mu_out = mu_in mu0 + s_in s0 cos(phi), phi uniform on [0, 2pi),
# where s = sqrt(1 - mu^2). Its density is the arcsine density below.


@njit(cache=True)
def azimuthal_kernel(mu_in, mu_out, mu0):
    """Density of mu_out given (mu_in, mu0); zero outside its support."""
    D = 1.0 - mu_in * mu_in - mu_out * mu_out - mu0 * mu0 + 2.0 * mu_in * mu_out * mu0
    if D <= 0.0:
        return 0.0
    return 1.0 / (PI * math.sqrt(D))


@njit(cache=True)
def azimuthal_bin_probability(mu_in, mu0, mu_low, mu_high):
    """Probability that mu_out lies in [mu_low, mu_high] given (mu_in, mu0)."""
    center = mu_in * mu0
    half_width = math.sqrt(max(0.0, 1.0 - mu_in * mu_in)) * math.sqrt(
        max(0.0, 1.0 - mu0 * mu0)
    )

    # Degenerate: mu_out = center exactly
    if half_width == 0.0:
        return 1.0 if mu_low <= center < mu_high else 0.0

    t_low = min(1.0, max(-1.0, (mu_low - center) / half_width))
    t_high = min(1.0, max(-1.0, (mu_high - center) / half_width))
    return (math.asin(t_high) - math.asin(t_low)) / PI


@njit(cache=True)
def azimuthal_bin_moments(mu_in, mu0, mu_low, mu_high, out):
    """
    out[b] = average over the azimuth of P_b(y(mu_out)) 1[mu_out in bin], b < len(out),
    with y = (2 mu_out - mu_low - mu_high) / (mu_high - mu_low) and
    mu_out = c + d cos(theta), c = mu_in mu0, d = sqrt(1 - mu_in^2) sqrt(1 - mu0^2):

        out[0] = (theta_hi - theta_lo) / pi       (azimuthal_bin_probability),
        out[1] = 2 / (pi dmu) [(c - mu_center)(theta_hi - theta_lo)
                               + d (sin theta_hi - sin theta_lo)],

    theta_lo / theta_hi the azimuths where mu_out leaves the bin at its top / bottom.
    The azimuthal kernel is symmetric in (mu_in, mu_out), so the same expression with
    the roles swapped gives the moments over an incident bin.
    """
    out[0] = azimuthal_bin_probability(mu_in, mu0, mu_low, mu_high)
    if len(out) < 2:
        return
    center = mu_in * mu0
    half_width = math.sqrt(max(0.0, 1.0 - mu_in * mu_in)) * math.sqrt(
        max(0.0, 1.0 - mu0 * mu0)
    )
    width = mu_high - mu_low
    mid = 0.5 * (mu_low + mu_high)
    if half_width == 0.0:
        out[1] = 2.0 * (center - mid) / width if mu_low <= center < mu_high else 0.0
        return
    t_low = min(1.0, max(-1.0, (mu_low - center) / half_width))
    t_high = min(1.0, max(-1.0, (mu_high - center) / half_width))
    # theta_hi = acos(t_low), theta_lo = acos(t_high); sin(acos(t)) = sqrt(1 - t^2)
    out[1] = (
        2.0
        / (PI * width)
        * (
            (center - mid) * PI * out[0]
            + half_width
            * (math.sqrt(1.0 - t_low * t_low) - math.sqrt(1.0 - t_high * t_high))
        )
    )


@njit(cache=True)
def mu0_from_theta(mu_in, mu_out, theta):
    """
    Substitution mu0 = mu_in mu_out + s_in s_out cos(theta), theta in [0, pi].

    Under it, azimuthal_kernel(mu_in, mu_out, mu0) dmu0 = dtheta / pi, which removes
    the inverse-square-root endpoint singularities from the mu0 integral.
    """
    s_in = math.sqrt(max(0.0, 1.0 - mu_in * mu_in))
    s_out = math.sqrt(max(0.0, 1.0 - mu_out * mu_out))
    return mu_in * mu_out + s_in * s_out * math.cos(theta)
