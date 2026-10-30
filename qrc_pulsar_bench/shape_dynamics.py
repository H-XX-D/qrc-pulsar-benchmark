"""Direct planar shape-sphere evolution with scale and rotation reconstruction.

For mass-weighted complex Jacobi vectors q1,q2 define u=q2/q1. This is a
stereographic chart on the oriented shape sphere:
    w = (1-|u|², 2 Re(u), 2 Im(u))/(1+|u|²).
The integrated coordinates are log(|q1|/L), rotation theta, Re(u), Im(u),
and their four rates, plus an optional pulsar clock integral. No Cartesian
positions/velocities are ODE state variables. Newtonian forces are evaluated
algebraically from the shape, scale and masses at each RHS call.

Differentiating q2=u*q1 and z=log(q1) gives
    z''=f1/q1-(z')²; u''=(f2-u*f1)/q1-2*u'*z'.
These equations retain nonzero angular momentum and pass through collinearity.
This implementation is for motion in one fixed plane. Nonplanar input is
rejected; it is never silently projected. Binary collision q1=0 is outside
this chart; collisions are not regularized. Geometry: Montgomery (2014),
https://arxiv.org/abs/1402.0841; the chain-rule evolution above is explicit.
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from .constants import G, c
from .relativity import njit


@njit(cache=True)
def _shape_rhs(t, y, masses, weights, length):
    m1, m2, m3 = masses
    m12 = m1 + m2
    mt = m12 + m3
    f2 = m2 / m12
    F3 = m3 / mt
    q1 = length * np.exp(y[0])
    u = complex(y[2], y[3])
    zdot = complex(y[4], y[5])
    udot = complex(y[6], y[7])
    r1 = q1 / weights[0]
    r2 = q1 * u / weights[1]
    d13 = r2 + f2 * r1
    d23 = r2 - (1 - f2) * r1
    if min(abs(r1), abs(d13), abs(d23)) == 0:
        raise ValueError("Collision is outside the shape chart")
    a1 = -G * m12 * r1 / abs(r1) ** 3 + G * m3 * (
        d23 / abs(d23) ** 3 - d13 / abs(d13) ** 3
    )
    a2 = -G * mt * ((1 - f2) * d13 / abs(d13) ** 3 + f2 * d23 / abs(d23) ** 3)
    f1 = weights[0] * a1
    f2_force = weights[1] * a2
    zddot = f1 / q1 - zdot * zdot
    uddot = (f2_force - u * f1) / q1 - 2 * udot * zdot
    v1 = r1 * zdot
    v2 = q1 * (udot + u * zdot) / weights[1]
    vp = -F3 * v2 - f2 * v1
    clock = (0.5 * abs(vp) ** 2 + G * m2 / abs(r1) + G * m3 / abs(d13)) / c**2
    return np.array(
        [y[4], y[5], y[6], y[7], zddot.real, zddot.imag, uddot.real, uddot.imag, clock]
    )


class PlanarShapeSolution:
    """A dense shape trajectory with explicit reconstruction for observables."""

    def __init__(self, solution, basis, weights, length, wall):
        self.solution = solution
        self.basis = basis
        self.weights = weights
        self.length = length
        self.wall_s = wall
        self.success = solution.success
        self.message = solution.message

    def shape(self, t):
        y = self.solution.sol(t)
        u = y[2] + 1j * y[3]
        norm = 1 + np.abs(u) ** 2
        return np.array(
            [(1 - np.abs(u) ** 2) / norm, 2 * u.real / norm, 2 * u.imag / norm]
        )

    def sol(self, t):
        """Reconstruct the 19-component timing adapter state at requested times."""
        scalar = np.ndim(t) == 0
        y = self.solution.sol(np.atleast_1d(t))
        q1 = self.length * np.exp(y[0] + 1j * y[1])
        u = y[2] + 1j * y[3]
        zdot = y[4] + 1j * y[5]
        udot = y[6] + 1j * y[7]
        values = [
            q1 / self.weights[0],
            q1 * zdot / self.weights[0],
            q1 * u / self.weights[1],
            q1 * (udot + u * zdot) / self.weights[1],
        ]
        result = np.zeros((19, len(y[0])))
        for i, value in enumerate(values):
            result[3 * i : 3 * i + 3] = self.basis @ np.array([value.real, value.imag])
        result[12] = y[8]
        return result[:, 0] if scalar else result


def integrate_planar_shape(
    state, end_seconds, rtol=1e-12, atol=1e-14, planarity_tolerance=1e-12
):
    """Evolve a planar RelationalState directly in a shape-sphere chart.

    Times are seconds since the supplied initial state. ``atol`` applies to
    dimensionless chart coordinates and their rates in inverse seconds. The
    common-plane check normalizes each nonzero vector separately so position
    and velocity units do not affect admission.
    """
    vectors = np.column_stack([state.r1, state.v1, state.r2, state.v2])
    norms = np.linalg.norm(vectors, axis=0)
    normalized = vectors[:, norms > 0] / norms[norms > 0]
    basis, singular, _ = np.linalg.svd(normalized, full_matrices=True)
    if len(singular) > 2 and singular[2] > planarity_tolerance * singular[0]:
        raise ValueError(
            "Direct planar shape evolution requires coplanar positions AND velocities; input was not projected."
        )
    basis = basis[:, :2]
    masses = np.array([state.m1, state.m2, state.m3])
    weights = np.sqrt(np.array([state.mu1, state.mu2]) / state.M_tot)
    projected = basis.T @ vectors
    complex_values = projected[0] + 1j * projected[1]
    q1, v1 = weights[0] * complex_values[:2]
    q2, v2 = weights[1] * complex_values[2:]
    if abs(q1) == 0:
        raise ValueError("q1=0 is outside this shape chart")
    u = q2 / q1
    zdot = v1 / q1
    udot = (v2 - u * v1) / q1
    length = abs(q1)
    initial = np.array(
        [
            0.0,
            np.angle(q1),
            u.real,
            u.imag,
            zdot.real,
            zdot.imag,
            udot.real,
            udot.imag,
            0.0,
        ]
    )
    start = time.perf_counter()
    solution = solve_ivp(
        _shape_rhs,
        (0.0, float(end_seconds)),
        initial,
        args=(masses, weights, length),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return PlanarShapeSolution(
        solution, basis, weights, length, time.perf_counter() - start
    )


@njit(cache=True)
def _spatial_shape_rhs(t, y, masses, length):
    """Triangle body-frame reconstruction, with independent spatial rotation.

    x1=a(1,0,0), x2=a(u,v,0), r=R(q)x. Differentiate twice:
    R^-1 r''=x''+2 omega×x'+omega'×x+omega×(omega×x).
    Solve these six equations for rho'',u'',v'',omega'. This chart requires
    v>0 (a noncollinear triangle); rotation carries spatial orientation.
    """
    m1, m2, m3 = masses
    m12 = m1 + m2
    mt = m12 + m3
    f2 = m2 / m12
    F3 = m3 / mt
    a = length * np.exp(y[0])
    u, v = y[1:3]
    q = y[3:7]
    rates = y[7:10]
    omega = y[10:13]
    if v <= 1e-12:
        raise ValueError("Spatial shape chart is singular at collinearity")
    r1 = np.array([a, 0.0, 0.0])
    r2 = np.array([a * u, a * v, 0.0])
    d13 = r2 + f2 * r1
    d23 = r2 - (1 - f2) * r1
    a1 = -G * m12 * r1 / a**3 + G * m3 * (
        d23 / np.linalg.norm(d23) ** 3 - d13 / np.linalg.norm(d13) ** 3
    )
    a2 = (
        -G
        * mt
        * (
            (1 - f2) * d13 / np.linalg.norm(d13) ** 3
            + f2 * d23 / np.linalg.norm(d23) ** 3
        )
    )
    dx1 = np.array([a * rates[0], 0.0, 0.0])
    dx2 = np.array([a * (rates[0] * u + rates[1]), a * (rates[0] * v + rates[2]), 0.0])
    centrifugal1 = np.cross(omega, np.cross(omega, r1))
    centrifugal2 = np.cross(omega, np.cross(omega, r2))
    rho_ddot = (a1[0] - centrifugal1[0]) / a - rates[0] ** 2
    omega_dot = np.zeros(3)
    omega_dot[2] = (a1[1] - 2 * omega[2] * dx1[0] - centrifugal1[1]) / a
    omega_dot[1] = -(a1[2] + 2 * omega[1] * dx1[0] - centrifugal1[2]) / a
    coriolis2 = 2 * np.cross(omega, dx2)
    omega_dot[0] = (a2[2] - coriolis2[2] - centrifugal2[2] + omega_dot[1] * a * u) / (
        a * v
    )
    ddx2 = a2 - coriolis2 - np.cross(omega_dot, r2) - centrifugal2
    dd_u = ddx2[0] / a - (rho_ddot + rates[0] ** 2) * u - 2 * rates[0] * rates[1]
    dd_v = ddx2[1] / a - (rho_ddot + rates[0] ** 2) * v - 2 * rates[0] * rates[2]
    qdot = np.empty(4)
    qdot[0] = -0.5 * np.dot(q[1:], omega)
    qdot[1:] = 0.5 * (q[0] * omega + np.cross(q[1:], omega))
    vp = -F3 * (dx2 + np.cross(omega, r2)) - f2 * (dx1 + np.cross(omega, r1))
    clock = (0.5 * np.dot(vp, vp) + G * m2 / a + G * m3 / np.linalg.norm(d13)) / c**2
    return np.concatenate(
        (rates, qdot, np.array([rho_ddot, dd_u, dd_v]), omega_dot, np.array([clock]))
    )


class SpatialShapeSolution:
    """Noncollinear spatial shape/scale/rotation trajectory; no hidden Cartesian ODE."""

    def __init__(self, solution, length, weights, wall):
        self.solution = solution
        self.length = length
        self.weights = weights
        self.wall_s = wall
        self.success = solution.success
        self.message = solution.message

    def shape(self, t):
        y = self.solution.sol(t)
        ratio = self.weights[1] / self.weights[0]
        u = ratio * (y[1] + 1j * y[2])
        denominator = 1 + np.abs(u) ** 2
        return np.array(
            [
                (1 - np.abs(u) ** 2) / denominator,
                2 * u.real / denominator,
                2 * u.imag / denominator,
            ]
        )

    def sol(self, t):
        scalar = np.ndim(t) == 0
        y = self.solution.sol(np.atleast_1d(t))
        n = y.shape[1]
        a = self.length * np.exp(y[0])
        u, v = y[1:3]
        rates = y[7:10]
        omega = y[10:13].T
        r1 = np.column_stack([a, np.zeros(n), np.zeros(n)])
        r2 = np.column_stack([a * u, a * v, np.zeros(n)])
        v1 = np.column_stack([a * rates[0], np.zeros(n), np.zeros(n)]) + np.cross(
            omega, r1
        )
        v2 = np.column_stack(
            [a * (rates[0] * u + rates[1]), a * (rates[0] * v + rates[2]), np.zeros(n)]
        ) + np.cross(omega, r2)
        rotation = Rotation.from_quat(y[[4, 5, 6, 3]].T)
        result = np.zeros((19, n))
        for index, vector in enumerate([r1, v1, r2, v2]):
            result[3 * index : 3 * index + 3] = rotation.apply(vector).T
        result[12] = y[13]
        return result[:, 0] if scalar else result


def integrate_spatial_shape(state, end_seconds, rtol=1e-12, atol=1e-15):
    """Evolve spatial shape, scale and orientation; exact collinearity is excluded.

    The spatial configuration quotient is an upper-hemisphere shape chart.
    Unlike the planar oriented sphere, reflection of a triangle is a rotation
    in 3D. Orientation/rotation rates are retained to distinguish full motions.
    """
    a = np.linalg.norm(state.r1)
    e1 = state.r1 / a
    parallel = np.dot(state.r2, e1)
    transverse = state.r2 - parallel * e1
    magnitude = np.linalg.norm(transverse)
    if magnitude / a <= 1e-12:
        raise ValueError("Spatial shape chart is singular at collinearity")
    e2 = transverse / magnitude
    e3 = np.cross(e1, e2)
    rotation = np.column_stack([e1, e2, e3])
    u = parallel / a
    v = magnitude / a
    b1 = rotation.T @ state.v1
    b2 = rotation.T @ state.v2
    rho_dot = b1[0] / a
    omega_z = b1[1] / a
    omega_y = -b1[2] / a
    omega_x = (b2[2] + a * u * omega_y) / (a * v)
    u_dot = b2[0] / a + v * omega_z - rho_dot * u
    v_dot = b2[1] / a - u * omega_z - rho_dot * v
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    initial = np.r_[
        0.0, u, v, quaternion, rho_dot, u_dot, v_dot, omega_x, omega_y, omega_z, 0.0
    ]
    masses = np.array([state.m1, state.m2, state.m3])
    weights = np.sqrt(np.array([state.mu1, state.mu2]) / state.M_tot)
    start = time.perf_counter()
    solution = solve_ivp(
        _spatial_shape_rhs,
        (0.0, float(end_seconds)),
        initial,
        args=(masses, a),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return SpatialShapeSolution(solution, a, weights, time.perf_counter() - start)
