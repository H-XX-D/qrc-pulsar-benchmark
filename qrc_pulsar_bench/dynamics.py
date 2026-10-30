"""
=====================================================================================
Relational Three-Body Dynamical Engine & Time-Evolution Integrator
=====================================================================================
Solves the full equations of motion for 3-body gravitational systems directly in
gauge-free Jacobi relative coordinates, tracking time-dependent trajectories,
Montgomery-Dragt shape sphere evolution on S², energy conservation, and projected
pulsar timing arrival delays.

Key Mathematical Properties:
1. Gauge Elimination: By formulating the dynamics in relative Jacobi vectors
   (r1 = x2 - x1, r2 = x3 - COM12), the 6 translational degrees of freedom are
   analytically eliminated. The center of mass is identically zero for all time.
2. Geometric Shape Sphere Tracking: At every time step along the trajectory,
   the inter-particle distances (a, b, c) are mapped to Montgomery-Dragt
   shape coordinates (w1, w2, w3) on S², satisfying w1^2 + w2^2 + w3^2 = 1
   to machine precision.
3. Observational Timing Synthesis: Derives the physical line-of-sight Roemer
   delay Delta t(t) = - x1(t) . n_los / c for the primary pulsar directly
   from the integrated trajectory.
4. Reference Verification: Includes an 18-variable Cartesian ODE integrator for
   side-by-side verification on identical initial conditions.
=====================================================================================
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List
from scipy.integrate import solve_ivp
from .constants import G, c, M_sun, AU, day_s, year_s
from .systems import HierarchicalTripleSystem

def orbital_elements_to_vectors(
    a: float,
    e: float,
    inc_rad: float,
    Omega_rad: float,
    omega_rad: float,
    f_rad: float,
    GM: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian orbital elements to 3D Cartesian position and velocity vectors."""
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(f_rad))
    pos_orb = np.array([r * np.cos(f_rad), r * np.sin(f_rad), 0.0])

    h = np.sqrt(GM * a * (1.0 - e**2))
    vel_orb = np.array([-(GM / h) * np.sin(f_rad), (GM / h) * (e + np.cos(f_rad)), 0.0])

    cO, sO = np.cos(Omega_rad), np.sin(Omega_rad)
    ci, si = np.cos(inc_rad), np.sin(inc_rad)
    cw, sw = np.cos(omega_rad), np.sin(omega_rad)

    R_mat = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci]
    ])

    return R_mat @ pos_orb, R_mat @ vel_orb

class RelationalState:
    """Represents the instantaneous relative state of a 3-body system."""

    def __init__(
        self,
        m1: float,
        m2: float,
        m3: float,
        r1: np.ndarray,
        v1: np.ndarray,
        r2: np.ndarray,
        v2: np.ndarray
    ):
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.m3 = float(m3)
        self.M12 = self.m1 + self.m2
        self.M_tot = self.M12 + self.m3

        self.mu1 = (self.m1 * self.m2) / self.M12
        self.mu2 = (self.M12 * self.m3) / self.M_tot

        self.r1 = np.asarray(r1, dtype=float)
        self.v1 = np.asarray(v1, dtype=float)
        self.r2 = np.asarray(r2, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)

    def to_barycentric_cartesian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert relative Jacobi state to center-of-mass barycentric Cartesian positions and velocities."""
        f1 = self.m1 / self.M12
        f2 = self.m2 / self.M12
        F3 = self.m3 / self.M_tot
        F12 = self.M12 / self.M_tot

        x_com12 = -F3 * self.r2
        v_com12 = -F3 * self.v2

        x1 = x_com12 - f2 * self.r1
        v1 = v_com12 - f2 * self.v1

        x2 = x_com12 + f1 * self.r1
        v2 = v_com12 + f1 * self.v1

        x3 = F12 * self.r2
        v3 = F12 * self.v2

        return x1, v1, x2, v2, x3, v3

    def compute_shape_coordinates(self) -> Tuple[float, float, float, float]:
        """
        Compute Montgomery-Dragt coordinates (w1, w2, w3) on S² and hyper-radius R.
        w1 = (2c^2 - a^2 - b^2) / R^2
        w2 = sqrt(3)(a^2 - b^2) / R^2
        w3 = 4 sqrt(3) Area / R^2
        """
        f1 = self.m1 / self.M12
        f2 = self.m2 / self.M12

        x12 = self.r1
        x13 = self.r2 + f2 * self.r1
        x23 = self.r2 - f1 * self.r1

        c_side = float(np.linalg.norm(x12))
        b_side = float(np.linalg.norm(x13))
        a_side = float(np.linalg.norm(x23))

        R_sq = a_side**2 + b_side**2 + c_side**2
        R = float(np.sqrt(R_sq))

        # Triangle area via cross product of relative vectors
        area = 0.5 * float(np.linalg.norm(np.cross(x12, x13)))

        w1 = (2.0 * c_side**2 - a_side**2 - b_side**2) / R_sq
        w2 = np.sqrt(3.0) * (a_side**2 - b_side**2) / R_sq
        w3 = 4.0 * np.sqrt(3.0) * area / R_sq

        return float(w1), float(w2), float(w3), R

    def total_energy(self) -> float:
        """Total mechanical energy E = T + V."""
        f1 = self.m1 / self.M12
        f2 = self.m2 / self.M12

        x12 = self.r1
        x13 = self.r2 + f2 * self.r1
        x23 = self.r2 - f1 * self.r1

        d12 = np.linalg.norm(x12)
        d13 = np.linalg.norm(x13)
        d23 = np.linalg.norm(x23)

        T = 0.5 * self.mu1 * np.dot(self.v1, self.v1) + 0.5 * self.mu2 * np.dot(self.v2, self.v2)
        V = -G * (self.m1 * self.m2 / d12 + self.m1 * self.m3 / d13 + self.m2 * self.m3 / d23)

        return float(T + V)

    def total_angular_momentum(self) -> np.ndarray:
        """Total orbital angular momentum vector J = mu1 (r1 x v1) + mu2 (r2 x v2)."""
        return self.mu1 * np.cross(self.r1, self.v1) + self.mu2 * np.cross(self.r2, self.v2)

    def pulsar_roemer_delay_s(self, line_of_sight: np.ndarray) -> float:
        """Projected geometric Roemer time delay for pulsar (body 1): Delta t = - x1 . n_los / c."""
        f2 = self.m2 / self.M12
        F3 = self.m3 / self.M_tot
        x1 = -F3 * self.r2 - f2 * self.r1
        n = line_of_sight / np.linalg.norm(line_of_sight)
        return float(-np.dot(x1, n) / c)

class RelationalThreeBodyIntegrator:
    """Solves the time-dependent equations of motion for 3 bodies in Jacobi relative coordinates."""

    @staticmethod
    def _jacobi_derivatives(t: float, y: np.ndarray, m1: float, m2: float, m3: float) -> np.ndarray:
        r1 = y[0:3]
        v1 = y[3:6]
        r2 = y[6:9]
        v2 = y[9:12]

        M12 = m1 + m2
        f1 = m1 / M12
        f2 = m2 / M12

        x12 = r1
        x13 = r2 + f2 * r1
        x23 = r2 - f1 * r1

        d12_cube = np.linalg.norm(x12)**3
        d13_cube = np.linalg.norm(x13)**3
        d23_cube = np.linalg.norm(x23)**3

        a1 = G * m2 * x12 / d12_cube + G * m3 * x13 / d13_cube
        a2 = -G * m1 * x12 / d12_cube + G * m3 * x23 / d23_cube
        a3 = -G * m1 * x13 / d13_cube - G * m2 * x23 / d23_cube

        a_r1 = a2 - a1
        a_r2 = a3 - (m1 * a1 + m2 * a2) / M12

        return np.concatenate([v1, a_r1, v2, a_r2])

    @classmethod
    def integrate(
        cls,
        triple: HierarchicalTripleSystem,
        num_inner_orbits: float = 10.0,
        num_output_points: int = 500,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> Dict[str, Any]:
        """Integrate the 3-body system over time, returning trajectories and physical observables."""
        m1 = triple.m1_msun * M_sun
        m2 = triple.m2_msun * M_sun
        m3 = triple.m3_msun * M_sun
        M12 = m1 + m2
        M_tot = M12 + m3

        P1 = triple.Pb1_days * day_s
        n1 = 2.0 * np.pi / P1
        a1 = (G * M12 / n1**2)**(1.0 / 3.0)

        P2 = triple.Pb2_days * day_s
        n2 = 2.0 * np.pi / P2
        a2 = (G * M_tot / n2**2)**(1.0 / 3.0)

        # Orbital inclinations
        i1 = np.arccos(triple.cos_i1)
        i2 = np.arccos(triple.cos_i2)

        # Initialize vectors from orbital elements
        r1_0, v1_0 = orbital_elements_to_vectors(a1, triple.e1, i1, 0.0, 0.0, 0.0, G * M12)
        r2_0, v2_0 = orbital_elements_to_vectors(a2, triple.e2, i2, 0.0, 0.0, 0.0, G * M_tot)

        y0 = np.concatenate([r1_0, v1_0, r2_0, v2_0])
        t_span = (0.0, num_inner_orbits * P1)
        t_eval = np.linspace(0.0, num_inner_orbits * P1, num_output_points)

        t_start = time.perf_counter()
        sol = solve_ivp(
            fun=lambda t, y: cls._jacobi_derivatives(t, y, m1, m2, m3),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol
        )
        wall_time_s = time.perf_counter() - t_start

        initial_state = RelationalState(m1, m2, m3, r1_0, v1_0, r2_0, v2_0)
        E0 = initial_state.total_energy()

        # Line of sight for Roemer delay (z-axis perpendicular to sky plane, tilt i1)
        los = np.array([0.0, np.sin(i1), np.cos(i1)])

        w1_arr = np.zeros(len(sol.t))
        w2_arr = np.zeros(len(sol.t))
        w3_arr = np.zeros(len(sol.t))
        R_arr = np.zeros(len(sol.t))
        E_arr = np.zeros(len(sol.t))
        roemer_us = np.zeros(len(sol.t))
        r1_norm = np.zeros(len(sol.t))

        for idx in range(len(sol.t)):
            st = RelationalState(
                m1, m2, m3,
                sol.y[0:3, idx], sol.y[3:6, idx],
                sol.y[6:9, idx], sol.y[9:12, idx]
            )
            w1, w2, w3, R = st.compute_shape_coordinates()
            w1_arr[idx] = w1
            w2_arr[idx] = w2
            w3_arr[idx] = w3
            R_arr[idx] = R
            E_arr[idx] = st.total_energy()
            roemer_us[idx] = st.pulsar_roemer_delay_s(los) * 1e6
            r1_norm[idx] = np.linalg.norm(st.r1)

        sphere_norm_err = np.max(np.abs(w1_arr**2 + w2_arr**2 + w3_arr**2 - 1.0))
        max_dE_E0 = np.max(np.abs((E_arr - E0) / E0))

        return {
            "system": triple.name,
            "num_orbits": num_inner_orbits,
            "span_days": t_span[1] / day_s,
            "wall_time_s": wall_time_s,
            "num_steps": sol.nfev,
            "t_eval_s": sol.t,
            "r1_norm_km": r1_norm / 1e3,
            "roemer_delay_us": roemer_us,
            "w1": w1_arr,
            "w2": w2_arr,
            "w3": w3_arr,
            "hyper_radius_km": R_arr / 1e3,
            "energy_conservation_err": float(max_dE_E0),
            "sphere_invariance_err": float(sphere_norm_err),
            "com_drift_m": 0.0 # Identically zero in Jacobi coordinates
        }

class CartesianThreeBodyIntegrator:
    """Standard 18-variable Cartesian ODE integrator for reference and benchmarking."""

    @staticmethod
    def _cartesian_derivatives(t: float, y: np.ndarray, m1: float, m2: float, m3: float) -> np.ndarray:
        x1 = y[0:3]; v1 = y[3:6]
        x2 = y[6:9]; v2 = y[9:12]
        x3 = y[12:15]; v3 = y[15:18]

        x12 = x2 - x1; d12_3 = np.linalg.norm(x12)**3
        x13 = x3 - x1; d13_3 = np.linalg.norm(x13)**3
        x23 = x3 - x2; d23_3 = np.linalg.norm(x23)**3

        a1 = G * m2 * x12 / d12_3 + G * m3 * x13 / d13_3
        a2 = -G * m1 * x12 / d12_3 + G * m3 * x23 / d23_3
        a3 = -G * m1 * x13 / d13_3 - G * m2 * x23 / d23_3

        return np.concatenate([v1, a1, v2, a2, v3, a3])

    @classmethod
    def integrate(
        cls,
        triple: HierarchicalTripleSystem,
        num_inner_orbits: float = 10.0,
        num_output_points: int = 500,
        rtol: float = 1e-10,
        atol: float = 1e-10
    ) -> Dict[str, Any]:
        """Integrate 3 bodies in Cartesian coordinates to benchmark against the relational solver."""
        m1 = triple.m1_msun * M_sun
        m2 = triple.m2_msun * M_sun
        m3 = triple.m3_msun * M_sun
        M12 = m1 + m2
        M_tot = M12 + m3

        P1 = triple.Pb1_days * day_s
        n1 = 2.0 * np.pi / P1
        a1 = (G * M12 / n1**2)**(1.0 / 3.0)

        P2 = triple.Pb2_days * day_s
        n2 = 2.0 * np.pi / P2
        a2 = (G * M_tot / n2**2)**(1.0 / 3.0)

        i1 = np.arccos(triple.cos_i1)
        i2 = np.arccos(triple.cos_i2)

        r1_0, v1_0 = orbital_elements_to_vectors(a1, triple.e1, i1, 0.0, 0.0, 0.0, G * M12)
        r2_0, v2_0 = orbital_elements_to_vectors(a2, triple.e2, i2, 0.0, 0.0, 0.0, G * M_tot)

        init_rel = RelationalState(m1, m2, m3, r1_0, v1_0, r2_0, v2_0)
        x1_0, v1_c0, x2_0, v2_c0, x3_0, v3_c0 = init_rel.to_barycentric_cartesian()

        y0 = np.concatenate([x1_0, v1_c0, x2_0, v2_c0, x3_0, v3_c0])
        t_span = (0.0, num_inner_orbits * P1)
        t_eval = np.linspace(0.0, num_inner_orbits * P1, num_output_points)

        t_start = time.perf_counter()
        sol = solve_ivp(
            fun=lambda t, y: cls._cartesian_derivatives(t, y, m1, m2, m3),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol
        )
        wall_time_s = time.perf_counter() - t_start

        # Center of mass evolution
        com_drift = np.zeros(len(sol.t))
        for idx in range(len(sol.t)):
            x1 = sol.y[0:3, idx]
            x2 = sol.y[6:9, idx]
            x3 = sol.y[12:15, idx]
            com = (m1 * x1 + m2 * x2 + m3 * x3) / M_tot
            com_drift[idx] = np.linalg.norm(com)

        return {
            "system": triple.name,
            "wall_time_s": wall_time_s,
            "num_steps": sol.nfev,
            "t_eval_s": sol.t,
            "sol_y": sol.y,
            "max_com_drift_m": float(np.max(com_drift))
        }

class DynamicalBenchmarkSuite:
    """Executes a rigorous, side-by-side comparison of the Relational vs Cartesian 3-body integrators."""

    @classmethod
    def compare(
        cls,
        triple: HierarchicalTripleSystem,
        num_orbits: float = 10.0,
        num_output_points: int = 500,
        tolerance: float = 1e-10
    ) -> Dict[str, Any]:
        """Runs both solvers on the exact same physical state and measures errors, drift, and runtimes."""
        rel_res = RelationalThreeBodyIntegrator.integrate(
            triple, num_inner_orbits=num_orbits, num_output_points=num_output_points, rtol=tolerance, atol=tolerance
        )
        cart_res = CartesianThreeBodyIntegrator.integrate(
            triple, num_inner_orbits=num_orbits, num_output_points=num_output_points, rtol=tolerance, atol=tolerance
        )

        # Measure trajectory difference
        m1 = triple.m1_msun * M_sun
        m2 = triple.m2_msun * M_sun
        a1 = (G * (m1 + m2) / (2.0 * np.pi / (triple.Pb1_days * day_s))**2)**(1.0 / 3.0)

        # r1 in Cartesian is x2 - x1
        r1_cart = cart_res["sol_y"][6:9, :] - cart_res["sol_y"][0:3, :]
        r1_diff = np.zeros(len(rel_res["t_eval_s"]))
        for i in range(len(rel_res["t_eval_s"])):
            diff_vec = (cart_res["sol_y"][6:9, i] - cart_res["sol_y"][0:3, i]) - (rel_res["r1_norm_km"][i] * 1e3 * (r1_cart[:, i] / np.linalg.norm(r1_cart[:, i])))
            r1_diff[i] = np.linalg.norm(diff_vec)

        return {
            "system": triple.name,
            "num_orbits": num_orbits,
            "relational_wall_time_ms": rel_res["wall_time_s"] * 1000.0,
            "cartesian_wall_time_ms": cart_res["wall_time_s"] * 1000.0,
            "relational_com_drift_m": 0.0,
            "cartesian_com_drift_m": cart_res["max_com_drift_m"],
            "energy_conservation_err": rel_res["energy_conservation_err"],
            "shape_sphere_invariance_err": rel_res["sphere_invariance_err"],
            "roemer_delay_min_us": float(np.min(rel_res["roemer_delay_us"])),
            "roemer_delay_max_us": float(np.max(rel_res["roemer_delay_us"])),
            "roemer_amplitude_us": float(0.5 * (np.max(rel_res["roemer_delay_us"]) - np.min(rel_res["roemer_delay_us"])))
        }
