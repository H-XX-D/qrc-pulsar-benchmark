"""Order-reduced Einstein–Infeld–Hoffmann acceleration, through first PN order.

All masses, positions and velocities use SI units in one inertial frame. This
includes the three-body potential terms; it is not a sum of isolated binary
precession formulas. Newtonian accelerations replace the implicit accelerations
inside the 1/c² terms, consistently discarding O(c^-4). See Newhall, Standish &
Williams (1983), A&A 125, 150, Eq. (1), beta=gamma=1.
"""

import numpy as np
from .constants import G, c

try:
    from numba import njit
except (ImportError, AttributeError):

    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=True)
def eih_correction(positions, velocities, masses, speed_of_light=c, gravity=G):
    """Return only the first-PN acceleration, not the Newtonian acceleration."""
    count = len(masses)
    potential = np.zeros(count)
    newton = np.zeros_like(positions)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            separation = positions[i] - positions[j]
            radius = np.sqrt(np.dot(separation, separation))
            if radius == 0:
                raise ValueError("Point-mass collision in EIH force")
            potential[i] += gravity * masses[j] / radius
            newton[i] -= gravity * masses[j] * separation / radius**3
    correction = np.zeros_like(positions)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            separation = positions[i] - positions[j]
            radius = np.sqrt(np.dot(separation, separation))
            vi, vj = velocities[i], velocities[j]
            radial = (
                4 * potential[i]
                + potential[j]
                - np.dot(vi, vi)
                - 2 * np.dot(vj, vj)
                + 4 * np.dot(vi, vj)
                + 1.5 * (np.dot(separation, vj) / radius) ** 2
                + 0.5 * np.dot(separation, newton[j])
            )
            velocity = np.dot(separation, 4 * vi - 3 * vj) * (vi - vj)
            correction[i] += (
                gravity
                * masses[j]
                / speed_of_light**2
                * (
                    (radial * separation + velocity) / radius**3
                    + 3.5 * newton[j] / radius
                )
            )
    return correction


@njit(cache=True)
def timing_rhs(t, y, masses, post_newtonian=False):
    """Jacobi 12-state + clock + Newtonian-COM position/velocity (19 values).

    The Newtonian mass-weighted COM need not move uniformly under EIH forces.
    Its six states are retained for reconstruction, rather than silently fixing
    a moving origin. At Newtonian order their derivatives vanish from rest.
    """
    m1, m2, m3 = masses
    m12 = m1 + m2
    mt = m12 + m3
    f2 = m2 / m12
    F3 = m3 / mt
    r1, v1, r2, v2 = y[:3], y[3:6], y[6:9], y[9:12]
    d13 = r2 + f2 * r1
    d23 = r2 - (1 - f2) * r1
    R1 = np.sqrt(np.dot(r1, r1))
    R13 = np.sqrt(np.dot(d13, d13))
    R23 = np.sqrt(np.dot(d23, d23))
    if min(R1, R13, R23) == 0:
        raise ValueError("Point-mass collision")
    a1 = -G * m12 * r1 / R1**3 + G * m3 * (d23 / R23**3 - d13 / R13**3)
    a2 = -G * mt * ((1 - f2) * d13 / R13**3 + f2 * d23 / R23**3)
    vp = y[16:19] - F3 * v2 - f2 * v1
    derivative = np.zeros(19)
    derivative[:3] = v1
    derivative[3:6] = a1
    derivative[6:9] = v2
    derivative[9:12] = a2
    derivative[12] = (0.5 * np.dot(vp, vp) + G * m2 / R1 + G * m3 / R13) / c**2
    derivative[13:16] = y[16:19]
    if post_newtonian:
        positions = np.empty((3, 3))
        velocities = np.empty((3, 3))
        positions[0] = -F3 * r2 - f2 * r1
        positions[1] = -F3 * r2 + (1 - f2) * r1
        positions[2] = (1 - F3) * r2
        velocities[0] = vp
        velocities[1] = y[16:19] - F3 * v2 + (1 - f2) * v1
        velocities[2] = y[16:19] + (1 - F3) * v2
        correction = eih_correction(positions, velocities, masses)
        inner_com = (1 - f2) * correction[0] + f2 * correction[1]
        derivative[3:6] += correction[1] - correction[0]
        derivative[9:12] += correction[2] - inner_com
        derivative[16:19] = (1 - F3) * inner_com + F3 * correction[2]
    return derivative
