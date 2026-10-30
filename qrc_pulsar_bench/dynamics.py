"""Newtonian point-mass three-body evolution in Jacobi and Cartesian coordinates.

Both solvers use SciPy DOP853. Jacobi vectors remove center-of-mass translation,
not rotation, chaos, or physical degrees of freedom of the relative orbit.
The unweighted shape coordinates are diagnostics derived from the trajectory;
no equations are integrated on S². Roemer delay is geometric only: no PN forces,
Einstein/Shapiro delays, radiation reaction, pulsar timing fit, or SEP fit.
"""

from numbers import Integral
import time
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G, c, M_sun, day_s
from .systems import HierarchicalTripleSystem


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite three-vector.")
    return value.copy()


def _unit_los(value):
    vector = _vector(value, 'line_of_sight')
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError("line_of_sight must be nonzero.")
    vector /= scale
    return vector / np.linalg.norm(vector)


def _distance(vector):
    distance = np.linalg.norm(vector)
    if not np.isfinite(distance) or distance == 0:
        raise ValueError("Newtonian point-mass collision or nonfinite separation; no collision regularization is implemented.")
    return distance


def compute_eih_accelerations(positions: np.ndarray, velocities: np.ndarray, masses: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1PN Einstein-Infeld-Hoffmann (EIH) accelerations for an N-body system.

    Evaluates the first post-Newtonian (1PN, O(1/c²)) equations of motion in standard
    harmonic barycentric coordinates following Einstein, Infeld, & Hoffmann (1938)
    and Poisson & Will (2014) Eq. (9.124).

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) containing barycentric Cartesian positions in metres.
    velocities : np.ndarray
        Array of shape (N, 3) containing barycentric Cartesian velocities in metres/second.
    masses : list or np.ndarray
        List of length N containing masses in kilograms.

    Returns
    -------
    a_total : np.ndarray of shape (N, 3)
        Total physical acceleration = a_Newtonian + (1/c²) * a_1PN.
    a_N : np.ndarray of shape (N, 3)
        Leading Newtonian gravitational acceleration.
    a_1pn_over_c2 : np.ndarray of shape (N, 3)
        1PN relativistic acceleration correction (divided by c²).
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    masses = [float(m) for m in masses]
    N = len(masses)
    if positions.shape != (N, 3) or velocities.shape != (N, 3):
        raise ValueError(f"positions and velocities must have shape ({N}, 3).")

    a_N = np.zeros((N, 3), dtype=float)
    for A in range(N):
        for B in range(N):
            if A == B:
                continue
            r_vec = positions[B] - positions[A]
            r = _distance(r_vec)
            a_N[A] += (G * masses[B] / (r**2)) * (r_vec / r)

    a_1pn = np.zeros((N, 3), dtype=float)
    inv_c2 = 1.0 / (c**2)

    for A in range(N):
        for B in range(N):
            if A == B:
                continue
            r_BA = positions[A] - positions[B]
            r_AB = _distance(r_BA)
            n_AB = r_BA / r_AB
            n_BA = -n_AB

            v_A = velocities[A]
            v_B = velocities[B]
            v_A_sq = float(np.dot(v_A, v_A))
            v_B_sq = float(np.dot(v_B, v_B))
            v_A_dot_v_B = float(np.dot(v_A, v_B))
            n_AB_dot_v_B = float(np.dot(n_AB, v_B))

            phi_A = sum(G * masses[C] / _distance(positions[A] - positions[C]) for C in range(N) if C != A)
            phi_B = sum(G * masses[C] / _distance(positions[B] - positions[C]) for C in range(N) if C != B)

            x_diff = positions[B] - positions[A]
            x_dot_a_B = float(np.dot(x_diff, a_N[B]))

            scalar_term = (
                v_A_sq + 2.0 * v_B_sq - 4.0 * v_A_dot_v_B - 1.5 * (n_AB_dot_v_B**2)
                - 4.0 * phi_A - phi_B + 0.5 * x_dot_a_B
            )

            term1 = (G * masses[B] / (r_AB**2)) * n_BA * scalar_term
            term2_dot = float(np.dot(n_AB, 4.0 * v_A - 3.0 * v_B))
            term2 = (G * masses[B] / (r_AB**2)) * term2_dot * (v_A - v_B)
            term3 = 3.5 * (G * masses[B] / r_AB) * a_N[B]

            a_1pn[A] += term1 + term2 + term3

    a_1pn_over_c2 = inv_c2 * a_1pn
    return a_N + a_1pn_over_c2, a_N, a_1pn_over_c2


def orbital_elements_to_vectors(a, e, inc_rad, Omega_rad, omega_rad, f_rad, GM):
    """Bound Kepler orbit in SI units; inclination is measured from the xy plane.

    Omega is ascending node, omega argument of periapsis, f true anomaly (radians).
    A fixed observer along +z sees the usual sin(inclination) projection.
    """
    a, GM = _positive(a, 'a'), _positive(GM, 'GM')
    if not np.isfinite(e) or not 0 <= e < 1:
        raise ValueError("eccentricity must satisfy 0 <= e < 1.")
    if not np.all(np.isfinite([inc_rad, Omega_rad, omega_rad, f_rad])):
        raise ValueError("Orbital angles must be finite radians.")
    radius = a * (1.0-e**2) / (1.0+e*np.cos(f_rad))
    pos = np.array([radius*np.cos(f_rad), radius*np.sin(f_rad), 0.0])
    h = np.sqrt(GM*a*(1.0-e**2))
    vel = GM/h * np.array([-np.sin(f_rad), e+np.cos(f_rad), 0.0])
    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    ci, si = np.cos(inc_rad), np.sin(inc_rad)
    cw, sw = np.cos(omega_rad), np.sin(omega_rad)
    rotation = np.array([
        [cO*cw-sO*sw*ci, -cO*sw-sO*cw*ci, sO*si],
        [sO*cw+cO*sw*ci, -sO*sw+cO*cw*ci, -cO*si],
        [sw*si, cw*si, ci],
    ])
    return rotation @ pos, rotation @ vel


class RelationalState:
    """Jacobi relative state in metres, metres/second and kilograms."""

    def __init__(self, m1, m2, m3, r1, v1, r2, v2):
        self.m1, self.m2, self.m3 = [_positive(m, 'mass') for m in (m1, m2, m3)]
        self.M12 = self.m1+self.m2
        self.M_tot = self.M12+self.m3
        self.mu1 = self.m1*(self.m2/self.M12)
        self.mu2 = self.M12*(self.m3/self.M_tot)
        self.r1 = _vector(r1, 'r1')
        self.v1 = _vector(v1, 'v1')
        self.r2 = _vector(r2, 'r2')
        self.v2 = _vector(v2, 'v2')

    def to_barycentric_cartesian(self):
        """Reconstruct the COM rest frame, up to floating-point roundoff."""
        f1, f2 = self.m1/self.M12, self.m2/self.M12
        F3, F12 = self.m3/self.M_tot, self.M12/self.M_tot
        return (-F3*self.r2-f2*self.r1, -F3*self.v2-f2*self.v1,
                -F3*self.r2+f1*self.r1, -F3*self.v2+f1*self.v1,
                F12*self.r2, F12*self.v2)

    def compute_shape_coordinates(self) -> Tuple[float, float, float, float]:
        """Unweighted, unsigned triangle shape, matching ShapeSphereReduction.

        Sides a=r12, b=r23, c=r31; w3 >= 0 folds mirror orientations together.
        R is sqrt(a²+b²+c²), not a mass-weighted dynamical hyper-radius.
        """
        x12 = self.r1
        x13 = self.r2+self.m2/self.M12*self.r1
        x23 = self.r2-self.m1/self.M12*self.r1
        scale = max(np.max(np.abs(v)) for v in (x12, x13, x23))
        if scale == 0 or not np.isfinite(scale):
            raise ValueError("Shape is undefined at total collision or nonfinite scale.")
        u12, u13, u23 = x12/scale, x13/scale, x23/scale
        a2, b2, c2 = np.dot(u12,u12), np.dot(u23,u23), np.dot(u13,u13)
        radius2 = a2+b2+c2
        area = 0.5*np.linalg.norm(np.cross(u12, u13))
        return (float((2*c2-a2-b2)/radius2), float(np.sqrt(3)*(a2-b2)/radius2),
                float(4*np.sqrt(3)*area/radius2), float(scale*np.sqrt(radius2)))

    def total_energy(self):
        d12 = _distance(self.r1)
        d13 = _distance(self.r2+self.m2/self.M12*self.r1)
        d23 = _distance(self.r2-self.m1/self.M12*self.r1)
        kinetic = 0.5*self.mu1*np.dot(self.v1,self.v1)+0.5*self.mu2*np.dot(self.v2,self.v2)
        potential = -G*(self.m1*self.m2/d12+self.m1*self.m3/d13+self.m2*self.m3/d23)
        return float(kinetic+potential)

    def total_angular_momentum(self):
        return self.mu1*np.cross(self.r1,self.v1)+self.mu2*np.cross(self.r2,self.v2)

    def pulsar_roemer_delay_s(self, line_of_sight):
        x1 = -self.m3/self.M_tot*self.r2-self.m2/self.M12*self.r1
        return float(-np.dot(x1, _unit_los(line_of_sight))/c)


def _initial_state(triple, inner_angles_rad, outer_angles_rad):
    if not isinstance(triple, HierarchicalTripleSystem):
        raise ValueError("Time evolution requires a HierarchicalTripleSystem.")
    masses = [_positive(m, 'solar mass')*M_sun for m in (triple.m1_msun,triple.m2_msun,triple.m3_msun)]
    periods = [_positive(p, 'orbital period')*day_s for p in (triple.Pb1_days,triple.Pb2_days)]
    for value in (triple.cos_i1, triple.cos_i2):
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("cos(inclination) must lie in [-1, 1].")
    inner_angles = _vector(inner_angles_rad, 'inner_angles_rad (node, periapsis, true anomaly)')
    outer_angles = _vector(outer_angles_rad, 'outer_angles_rad (node, periapsis, true anomaly)')
    gm1, gm2 = G*sum(masses[:2]), G*sum(masses)
    a1, a2 = (gm1*(periods[0]/(2*np.pi))**2)**(1/3), (gm2*(periods[1]/(2*np.pi))**2)**(1/3)
    r1, v1 = orbital_elements_to_vectors(a1,triple.e1,np.arccos(triple.cos_i1),*inner_angles,gm1)
    r2, v2 = orbital_elements_to_vectors(a2,triple.e2,np.arccos(triple.cos_i2),*outer_angles,gm2)
    state = RelationalState(*masses,r1,v1,r2,v2)
    state.total_energy()  # Reject coincident initial bodies before integration.
    inputs = dict(masses_kg=masses, periods_s=periods, eccentricities=[triple.e1,triple.e2],
                  inclinations_rad=[float(np.arccos(triple.cos_i1)),float(np.arccos(triple.cos_i2))],
                  inner_angles_rad=inner_angles.tolist(), outer_angles_rad=outer_angles.tolist(),
                  note="Illustrative osculating elements at t=0. Default node, periapsis and true anomaly are zero; no observed epoch or fitted orbital phase is supplied.")
    return state, periods[0], inputs


def _integrate(rhs, y0, masses, period, orbits, points, rtol, atol, enable_1pn=False):
    duration = _positive(orbits, 'num_inner_orbits')*period
    if not np.isfinite(duration):
        raise ValueError("Integration duration must be finite.")
    if isinstance(points, bool) or not isinstance(points, Integral) or points < 2:
        raise ValueError("num_output_points must be an integer >= 2.")
    rtol, atol = _positive(rtol, 'rtol'), _positive(atol, 'atol')
    times = np.linspace(0., duration, points)
    start = time.perf_counter()
    sol = solve_ivp(lambda t,y: rhs(t,y,*masses,enable_1pn=enable_1pn), (0.,duration), y0, t_eval=times,
                    method='DOP853', rtol=rtol, atol=atol)
    wall_time = time.perf_counter()-start
    if not sol.success:
        raise RuntimeError(f"Three-body integration failed: {sol.message}")
    if (sol.y.shape != (len(y0),points) or not np.array_equal(sol.t,times)
            or not np.all(np.isfinite(sol.y))):
        raise RuntimeError("Three-body integration returned an incomplete or nonfinite trajectory.")
    return sol, wall_time


def _observables(jacobi, cartesian, masses, los):
    count = jacobi.shape[1]
    shapes, energies, angular = np.empty((4,count)), np.empty(count), np.empty((3,count))
    for idx in range(count):
        state = RelationalState(*masses,jacobi[:3,idx],jacobi[3:6,idx],jacobi[6:9,idx],jacobi[9:12,idx])
        shapes[:,idx] = state.compute_shape_coordinates()
        energies[idx] = state.total_energy()
        angular[:,idx] = state.total_angular_momentum()
    weights = np.asarray(masses)/sum(masses)
    com = sum(weights[i]*cartesian[6*i:6*i+3] for i in range(3))
    velocity_com = sum(weights[i]*cartesian[6*i+3:6*i+6] for i in range(3))
    energy_delta = float(np.max(np.abs(energies-energies[0])))
    angular_delta = float(np.max(np.linalg.norm(angular-angular[:,[0]],axis=0)))
    return dict(
        w1=shapes[0],w2=shapes[1],w3=shapes[2],hyper_radius_km=shapes[3]/1e3,
        roemer_delay_us=-(los @ cartesian[:3])/c*1e6,
        r1_norm_km=np.linalg.norm(jacobi[:3],axis=0)/1e3,
        energy_j=energies, angular_momentum=angular,
        energy_conservation_err=energy_delta/abs(energies[0]) if energies[0] != 0 else None,
        angular_momentum_conservation_err=angular_delta/np.linalg.norm(angular[:,0]) if np.linalg.norm(angular[:,0]) != 0 else None,
        sphere_invariance_err=float(np.max(np.abs(np.sum(shapes[:3]**2,axis=0)-1))),
        com_offset_m=float(np.max(np.linalg.norm(com,axis=0))),
        com_drift_m=float(np.max(np.linalg.norm(com-com[:,[0]],axis=0))),
        max_com_speed_m_s=float(np.max(np.linalg.norm(velocity_com,axis=0))),
    )


class RelationalThreeBodyIntegrator:
    """Newtonian or 1PN Einstein-Infeld-Hoffmann integration of 12 Jacobi components."""

    @staticmethod
    def _jacobi_derivatives(t, y, m1, m2, m3, enable_1pn=False):
        r1, v1, r2, v2 = y[:3], y[3:6], y[6:9], y[9:12]
        m12, mt = m1 + m2, m1 + m2 + m3
        if not enable_1pn:
            x13, x23 = r2 + m2/m12*r1, r2 - m1/m12*r1
            u12, u13, u23 = r1/_distance(r1)**3, x13/_distance(x13)**3, x23/_distance(x23)**3
            # Algebraic Jacobi transformation of pairwise Newtonian accelerations.
            a_r1 = -G*m12*u12 + G*m3*(u23 - u13)
            a_r2 = -G*mt*(m1/m12*u13 + m2/m12*u23)
            return np.concatenate([v1, a_r1, v2, a_r2])

        f1, f2 = m1 / m12, m2 / m12
        F3, F12 = m3 / mt, m12 / mt
        x1 = -F3 * r2 - f2 * r1
        v1_c = -F3 * v2 - f2 * v1
        x2 = -F3 * r2 + f1 * r1
        v2_c = -F3 * v2 + f1 * v1
        x3 = F12 * r2
        v3_c = F12 * v2

        pos = np.array([x1, x2, x3])
        vel = np.array([v1_c, v2_c, v3_c])
        a_tot, _, _ = compute_eih_accelerations(pos, vel, [m1, m2, m3])

        a_r1 = a_tot[1] - a_tot[0]
        a_r2 = a_tot[2] - (f1 * a_tot[0] + f2 * a_tot[1])
        return np.concatenate([v1, a_r1, v2, a_r2])

    @classmethod
    def integrate(cls, triple, num_inner_orbits=10., num_output_points=500, rtol=1e-10, atol=1e-10,
                  *, inner_angles_rad=(0.,0.,0.), outer_angles_rad=(0.,0.,0.), line_of_sight=(0.,0.,1.),
                  enable_1pn=False):
        """Integrate illustrative initial elements; angles are (node, periapsis, f).

        Times are seconds since an arbitrary t=0. atol applies separately to SI
        position and velocity components; equal tolerances across coordinate
        systems do not guarantee equal trajectory error. Extrema use output samples.
        """
        state, period, inputs = _initial_state(triple, inner_angles_rad, outer_angles_rad)
        los = _unit_los(line_of_sight)
        masses = [state.m1, state.m2, state.m3]
        sol, wall = _integrate(cls._jacobi_derivatives, np.concatenate([state.r1, state.v1, state.r2, state.v2]),
                               masses, period, num_inner_orbits, num_output_points, rtol, atol, enable_1pn=enable_1pn)
        f1, f2, F3, F12 = state.m1/state.M12, state.m2/state.M12, state.m3/state.M_tot, state.M12/state.M_tot
        cart = np.vstack([-F3*sol.y[6:9]-f2*sol.y[:3], -F3*sol.y[9:12]-f2*sol.y[3:6],
                          -F3*sol.y[6:9]+f1*sol.y[:3], -F3*sol.y[9:12]+f1*sol.y[3:6],
                          F12*sol.y[6:9], F12*sol.y[9:12]])
        return dict(system=triple.name, num_orbits=num_inner_orbits, span_days=sol.t[-1]/day_s,
                    wall_time_s=wall, nfev=sol.nfev, num_steps=sol.nfev, t_eval_s=sol.t,
                    sol_y=sol.y, cartesian_y=cart, r1_m=sol.y[:3], r2_m=sol.y[6:9],
                    initial_conditions=inputs, line_of_sight=los.tolist(), enable_1pn=enable_1pn,
                    **_observables(sol.y, cart, masses, los))


class CartesianThreeBodyIntegrator:
    """18-component Newtonian or 1PN Einstein-Infeld-Hoffmann reference using DOP853."""

    @staticmethod
    def _cartesian_derivatives(t, y, m1, m2, m3, enable_1pn=False):
        x1, v1, x2, v2, x3, v3 = y[:3], y[3:6], y[6:9], y[9:12], y[12:15], y[15:18]
        if not enable_1pn:
            u12, u13, u23 = x2-x1, x3-x1, x3-x2
            u12, u13, u23 = u12/_distance(u12)**3, u13/_distance(u13)**3, u23/_distance(u23)**3
            return np.concatenate([v1, G*(m2*u12+m3*u13), v2, G*(-m1*u12+m3*u23), v3, -G*(m1*u13+m2*u23)])

        pos = np.array([x1, x2, x3])
        vel = np.array([v1, v2, v3])
        a_tot, _, _ = compute_eih_accelerations(pos, vel, [m1, m2, m3])
        return np.concatenate([v1, a_tot[0], v2, a_tot[1], v3, a_tot[2]])

    @classmethod
    def integrate(cls, triple, num_inner_orbits=10., num_output_points=500, rtol=1e-10, atol=1e-10,
                  *, inner_angles_rad=(0.,0.,0.), outer_angles_rad=(0.,0.,0.), line_of_sight=(0.,0.,1.),
                  enable_1pn=False):
        state, period, inputs = _initial_state(triple, inner_angles_rad, outer_angles_rad)
        masses, los = [state.m1, state.m2, state.m3], _unit_los(line_of_sight)
        sol, wall = _integrate(cls._cartesian_derivatives, np.concatenate(state.to_barycentric_cartesian()),
                               masses, period, num_inner_orbits, num_output_points, rtol, atol, enable_1pn=enable_1pn)
        f1, f2 = state.m1/state.M12, state.m2/state.M12
        jacobi = np.vstack([sol.y[6:9]-sol.y[:3], sol.y[9:12]-sol.y[3:6],
                            sol.y[12:15]-f1*sol.y[:3]-f2*sol.y[6:9], sol.y[15:18]-f1*sol.y[3:6]-f2*sol.y[9:12]])
        observables = _observables(jacobi, sol.y, masses, los)
        return dict(system=triple.name, wall_time_s=wall, nfev=sol.nfev, num_steps=sol.nfev,
                    t_eval_s=sol.t, sol_y=sol.y, jacobi_y=jacobi, initial_conditions=inputs,
                    line_of_sight=los.tolist(), max_com_drift_m=observables['com_drift_m'], enable_1pn=enable_1pn, **observables)


class DynamicalBenchmarkSuite:
    """Measure both solvers and compare complete vector trajectories on one grid."""

    @classmethod
    def compare(cls,triple,num_orbits=10.,num_output_points=500,tolerance=1e-10,enable_1pn=False,**initial_options):
        options = dict(num_inner_orbits=num_orbits,num_output_points=num_output_points,
                       rtol=tolerance,atol=tolerance,enable_1pn=enable_1pn,**initial_options)
        rel = RelationalThreeBodyIntegrator.integrate(triple,**options)
        cart = CartesianThreeBodyIntegrator.integrate(triple,**options)
        if not np.array_equal(rel['t_eval_s'],cart['t_eval_s']):
            raise RuntimeError("Cannot compare trajectories on different time grids.")
        f1 = triple.m1_msun/(triple.m1_msun+triple.m2_msun)
        f2 = 1-f1
        # Use actual Cartesian vectors. Rebuilding direction from the reference
        # and comparing only radii would hide orbital phase/orientation errors.
        r1_cart = cart['sol_y'][6:9]-cart['sol_y'][:3]
        r2_cart = cart['sol_y'][12:15]-f1*cart['sol_y'][:3]-f2*cart['sol_y'][6:9]
        inner_error = float(np.max(np.linalg.norm(r1_cart-rel['r1_m'],axis=0)))
        outer_error = float(np.max(np.linalg.norm(r2_cart-rel['r2_m'],axis=0)))
        position_error = max(float(np.max(np.linalg.norm(cart['sol_y'][i:i+3]-rel['cartesian_y'][i:i+3],axis=0))) for i in (0,6,12))
        velocity_error = max(float(np.max(np.linalg.norm(cart['sol_y'][i:i+3]-rel['cartesian_y'][i:i+3],axis=0))) for i in (3,9,15))
        los = np.asarray(rel['line_of_sight'])
        cart_delay = -(los @ cart['sol_y'][:3])/c*1e6
        return dict(
            system=triple.name,num_orbits=num_orbits,
            relational_wall_time_ms=rel['wall_time_s']*1000,cartesian_wall_time_ms=cart['wall_time_s']*1000,
            relational_com_drift_m=rel['com_drift_m'],cartesian_com_drift_m=cart['com_drift_m'],
            energy_conservation_err=rel['energy_conservation_err'],cartesian_energy_conservation_err=cart['energy_conservation_err'],
            relational_angular_momentum_err=rel['angular_momentum_conservation_err'],cartesian_angular_momentum_err=cart['angular_momentum_conservation_err'],
            shape_sphere_invariance_err=rel['sphere_invariance_err'],
            max_inner_vector_difference_m=inner_error,max_outer_vector_difference_m=outer_error,
            max_inner_relative_difference=inner_error/np.linalg.norm(rel['r1_m'][:,0]),
            max_outer_relative_difference=outer_error/np.linalg.norm(rel['r2_m'][:,0]),
            max_barycentric_position_difference_m=position_error,max_barycentric_velocity_difference_m_s=velocity_error,
            max_roemer_difference_us=float(np.max(np.abs(rel['roemer_delay_us']-cart_delay))),
            roemer_delay_min_us=float(np.min(rel['roemer_delay_us'])),roemer_delay_max_us=float(np.max(rel['roemer_delay_us'])),
            roemer_amplitude_us=float(0.5*np.ptp(rel['roemer_delay_us'])),
            note="Coordinate-consistency test with the same Newtonian force law and DOP853 solver; not an independent physical model or observational validation. Wall times cover integration only; COM reconstruction and sphere norm are not accuracy proofs.",
        )
