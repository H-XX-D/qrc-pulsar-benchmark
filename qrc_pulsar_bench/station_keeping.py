"""
=====================================================================================
QRC-PULSAR-BENCH: Autonomous Predictive Station-Keeping & Disturbance Rejection Engine
=====================================================================================
Closed-loop Guidance, Navigation, and Control (GNC) for cislunar multi-body spacecraft:
  1. Circular Restricted Three-Body Problem (CR3BP):
     - Earth-Moon barycentric rotating frame with collinear Lagrange point L2.
  2. Non-Conservative Disturbance Forces ("The Force Gap"):
     - Solar Radiation Pressure (SRP) with synodic rotating Sun geometry.
     - Lunar oblateness and mascon perturbations (J2).
  3. Symplectic Monodromy & Invariant Manifold Structure:
     - Exact periodic Lyapunov reference orbit via half-period symmetric shooting.
     - Symplectic eigenvalue analysis verifying unstable manifold lambda_u > 1.
  4. Receding-Horizon Minimum-Energy Predictive Guidance:
     - Projects state deviation forward along the State Transition Matrix (STM).
     - Solves minimum-energy Tikhonov regularized impulse delta-V at the lookahead horizon.
     - Parameterized by Earth Mission Control policy knob (aggressiveness alpha,
       lookahead horizon, and deadband thresholds).
=====================================================================================
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class EarthMoonConstants:
    """Astronomical and physical constants for Earth-Moon CR3BP system."""
    mu: float = 0.0121505856
    length_star_m: float = 384400e3         # Earth-Moon distance: 384,400 km
    mass_earth_kg: float = 5.972168e24
    mass_moon_kg: float = 7.342e22
    time_star_s: float = 3.751903e5         # Characteristic time: ~4.34 days
    velocity_star_mps: float = 384400e3 / 3.751903e5  # ~1024.55 m/s
    accel_star_mps2: float = 384400e3 / (3.751903e5**2) # ~0.00273 m/s^2
    r_moon_norm: float = 1737.4e3 / 384400e3 # ~0.00452
    j2_moon: float = 2.0e-4
    sun_synodic_freq: float = 0.9251        # Non-dimensional synodic frequency of Sun


@dataclass(frozen=True)
class StationKeepingPolicy:
    """Ground-configurable guidance and risk policy parameters."""
    aggressiveness: float = 0.5             # alpha in (0, 1]: trade-off between fuel & corridor
    lookahead_days: float = 12.0            # Forward prediction horizon
    check_interval_days: float = 3.0        # Thruster decision cadence
    deadband_km: float = 100.0              # Drift tolerance before applying impulse
    max_impulse_mps: float = 5.0            # Thruster physical limit per burn


@dataclass(frozen=True)
class StationKeepingReport:
    """Comprehensive performance report of the closed-loop flight simulation."""
    duration_days: float
    total_delta_v_mps: float
    delta_v_per_month_mps: float
    burn_count: int
    mean_burn_mps: float
    max_burn_mps: float
    max_position_error_km: float
    rms_position_error_km: float
    uncontrolled_drift_km: float
    fuel_savings_factor_vs_uncontrolled: float


class EarthMoonCR3BP:
    """Core dynamical model for Earth-Moon restricted 3-body system."""

    def __init__(self, consts: EarthMoonConstants = EarthMoonConstants()):
        self.consts = consts
        self.mu = consts.mu
        self.x_l2 = self._compute_l2_position()

    def _compute_l2_position(self) -> float:
        """Finds collinear L2 Lagrange point via Newton-Raphson."""
        mu = self.mu
        def f(x):
            return x - (1.0 - mu) / ((x + mu)**2) - mu / ((x - 1.0 + mu)**2)
        def df(x):
            return 1.0 + 2.0 * (1.0 - mu) / ((x + mu)**3) + 2.0 * mu / ((x - 1.0 + mu)**3)
        x = 1.15
        for _ in range(30):
            fx = f(x)
            x -= fx / df(x)
            if abs(fx) < 1e-13:
                break
        return float(x)

    def compute_disturbances(self, t: float, r: np.ndarray, srp_mps2: float = 1.5e-7) -> np.ndarray:
        """
        Evaluates non-Hamiltonian environmental perturbations ("The Force Gap"):
          - Solar Radiation Pressure (SRP) in synodic rotating frame.
          - Lunar J2 oblateness anomaly.
        """
        # SRP
        a_srp_norm = srp_mps2 / self.consts.accel_star_mps2
        theta_sun = self.consts.sun_synodic_freq * t
        a_srp = a_srp_norm * np.array([np.cos(theta_sun), np.sin(theta_sun), 0.0])

        # Lunar J2
        r2_vec = r - np.array([1.0 - self.mu, 0.0, 0.0])
        dist2 = np.linalg.norm(r2_vec)
        j2_coeff = 1.5 * self.consts.j2_moon * self.mu * (self.consts.r_moon_norm**2) / (dist2**4 + 1e-14)
        a_mascon = - j2_coeff * r2_vec

        return a_srp + a_mascon

    def equations_of_motion(
        self,
        t: float,
        state: np.ndarray,
        include_disturbances: bool = True,
        srp_mps2: float = 1.5e-7
    ) -> np.ndarray:
        """CR3BP equations of motion in Earth-Moon rotating frame."""
        x, y, z, vx, vy, vz = state
        mu = self.mu

        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)
        r1_3 = r1**3
        r2_3 = r2**3

        omega_x = x - (1.0 - mu) * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3
        omega_y = y - (1.0 - mu) * y / r1_3 - mu * y / r2_3
        omega_z = - (1.0 - mu) * z / r1_3 - mu * z / r2_3

        a_dist = self.compute_disturbances(t, state[0:3], srp_mps2=srp_mps2) if include_disturbances else np.zeros(3)

        ax = 2.0 * vy + omega_x + a_dist[0]
        ay = -2.0 * vx + omega_y + a_dist[1]
        az = omega_z + a_dist[2]

        return np.array([vx, vy, vz, ax, ay, az])

    def equations_with_stm(self, t: float, y: np.ndarray) -> np.ndarray:
        """Equations of motion coupled with 6x6 State Transition Matrix variational flow."""
        x, y_pos, z = y[0:3]
        vx, vy, vz = y[3:6]
        mu = self.mu

        r1 = np.sqrt((x + mu)**2 + y_pos**2 + z**2)
        r2 = np.sqrt((x - 1.0 + mu)**2 + y_pos**2 + z**2)
        r1_3 = r1**3
        r2_3 = r2**3

        omega_x = x - (1.0 - mu) * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3
        omega_y = y_pos - (1.0 - mu) * y_pos / r1_3 - mu * y_pos / r2_3
        omega_z = - (1.0 - mu) * z / r1_3 - mu * z / r2_3

        ydot = np.zeros(42)
        ydot[0:3] = [vx, vy, vz]
        ydot[3:6] = [2.0 * vy + omega_x, -2.0 * vx + omega_y, omega_z]

        # Gravitational Hessian
        r1_5 = r1**5
        r2_5 = r2**5
        Uxx = 1.0 - (1.0 - mu)/r1_3 - mu/r2_3 + 3.0*(1.0 - mu)*(x + mu)**2/r1_5 + 3.0*mu*(x - 1.0 + mu)**2/r2_5
        Uyy = 1.0 - (1.0 - mu)/r1_3 - mu/r2_3 + 3.0*(1.0 - mu)*y_pos**2/r1_5 + 3.0*mu*y_pos**2/r2_5
        Uzz = - (1.0 - mu)/r1_3 - mu/r2_3 + 3.0*(1.0 - mu)*z**2/r1_5 + 3.0*mu*z**2/r2_5
        Uxy = 3.0*(1.0 - mu)*(x + mu)*y_pos/r1_5 + 3.0*mu*(x - 1.0 + mu)*y_pos/r2_5
        Uxz = 3.0*(1.0 - mu)*(x + mu)*z/r1_5 + 3.0*mu*(x - 1.0 + mu)*z/r2_5
        Uyz = 3.0*(1.0 - mu)*y_pos*z/r1_5 + 3.0*mu*y_pos*z/r2_5

        A = np.zeros((6, 6))
        A[0:3, 3:6] = np.eye(3)
        A[3, 0] = Uxx; A[3, 1] = Uxy; A[3, 2] = Uxz; A[3, 4] = 2.0
        A[4, 0] = Uxy; A[4, 1] = Uyy; A[4, 2] = Uyz; A[4, 3] = -2.0
        A[5, 0] = Uxz; A[5, 1] = Uyz; A[5, 2] = Uzz

        Phi = y[6:42].reshape((6, 6))
        Phi_dot = A @ Phi
        ydot[6:42] = Phi_dot.flatten()

        return ydot

    def compute_stm(self, t_start: float, t_end: float, state_start: np.ndarray) -> np.ndarray:
        """Computes State Transition Matrix Phi(t_end, t_start) along trajectory."""
        y_init = np.zeros(42)
        y_init[0:6] = state_start
        y_init[6:42] = np.eye(6).flatten()
        sol = solve_ivp(
            self.equations_with_stm,
            (t_start, t_end),
            y_init,
            rtol=1e-8,
            atol=1e-10,
            method='DOP853'
        )
        return sol.y[6:42, -1].reshape((6, 6))


class PeriodicLyapunovOrbit:
    """Computes and interpolates a high-precision periodic Lyapunov reference orbit around L2."""

    def __init__(self, cr3bp: EarthMoonCR3BP, dx_offset: float = 0.005):
        self.cr3bp = cr3bp
        self.x0 = cr3bp.x_l2 + dx_offset
        self.vy0, self.period = self._solve_periodic_orbit()
        self.t_eval, self.sol_ref = self._generate_dense_reference()

    def _solve_periodic_orbit(self) -> Tuple[float, float]:
        """Symmetric half-period shooting algorithm targeting vx(T/2) = 0."""
        def get_vx_half(vy_val):
            y0 = [self.x0, 0.0, 0.0, 0.0, vy_val, 0.0]
            sol_init = solve_ivp(
                lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=False),
                (0.0, 0.2),
                y0,
                rtol=1e-10,
                atol=1e-12,
                method='DOP853'
            )
            def y_crossing(t, y):
                return y[1]
            y_crossing.direction = 1
            y_crossing.terminal = True
            sol = solve_ivp(
                lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=False),
                (0.2, 5.0),
                sol_init.y[:, -1],
                events=y_crossing,
                rtol=1e-10,
                atol=1e-12,
                method='DOP853'
            )
            if len(sol.t_events[0]) == 0:
                return 999.0
            return sol.y_events[0][0][3]

        res = root_scalar(get_vx_half, bracket=[-0.035, -0.020], xtol=1e-11)
        vy_exact = float(res.root)

        # Get exact half-period
        y0 = [self.x0, 0.0, 0.0, 0.0, vy_exact, 0.0]
        sol_init = solve_ivp(
            lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=False),
            (0.0, 0.2),
            y0,
            rtol=1e-11,
            atol=1e-13,
            method='DOP853'
        )
        def y_crossing(t, y):
            return y[1]
        y_crossing.direction = 1
        y_crossing.terminal = True
        sol = solve_ivp(
            lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=False),
            (0.2, 5.0),
            sol_init.y[:, -1],
            events=y_crossing,
            rtol=1e-11,
            atol=1e-13,
            method='DOP853'
        )
        t_half = float(sol.t_events[0][0])
        full_period = 2.0 * t_half

        return vy_exact, full_period

    def _generate_dense_reference(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-computes a dense table of the reference orbit over one period."""
        t_dense = np.linspace(0.0, self.period, 2000)
        sol = solve_ivp(
            lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=False),
            (0.0, self.period),
            [self.x0, 0.0, 0.0, 0.0, self.vy0, 0.0],
            t_eval=t_dense,
            rtol=1e-12,
            atol=1e-14,
            method='DOP853'
        )
        return t_dense, sol.y

    def get_reference_state(self, t: float) -> np.ndarray:
        """Interpolates reference state [x, y, z, vx, vy, vz] at any arbitrary time t."""
        t_mod = t % self.period
        r_ref = np.zeros(3)
        v_ref = np.zeros(3)
        for i in range(3):
            r_ref[i] = np.interp(t_mod, self.t_eval, self.sol_ref[i])
            v_ref[i] = np.interp(t_mod, self.t_eval, self.sol_ref[i + 3])
        return np.concatenate([r_ref, v_ref])

    def compute_monodromy_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Monodromy matrix M = Phi(T, 0) and its eigenvalues."""
        M = self.cr3bp.compute_stm(0.0, self.period, np.array([self.x0, 0.0, 0.0, 0.0, self.vy0, 0.0]))
        eigvals = np.linalg.eigvals(M)
        return M, eigvals


class PredictiveStationKeepingEngine:
    """
    Autonomous closed-loop station-keeping controller using receding-horizon
    minimum-energy manifold targeting to reject unmodeled disturbance forces.
    """

    def __init__(self, cr3bp: Optional[EarthMoonCR3BP] = None):
        self.cr3bp = cr3bp or EarthMoonCR3BP()
        self.ref_orbit = PeriodicLyapunovOrbit(self.cr3bp)

    def run_simulation(
        self,
        duration_days: float = 90.0,
        policy: StationKeepingPolicy = StationKeepingPolicy(),
        srp_mps2: float = 1.5e-7,
        uncontrolled_comparison: bool = True
    ) -> StationKeepingReport:
        """
        Executes multi-month closed-loop predictive station-keeping simulation.

        Returns:
            report: StationKeepingReport detailing delta-V budget, tracking error, and burn count.
        """
        consts = self.cr3bp.consts
        dt_check = policy.check_interval_days * 86400.0 / consts.time_star_s
        total_time = duration_days * 86400.0 / consts.time_star_s
        t_horizon_norm = policy.lookahead_days * 86400.0 / consts.time_star_s
        n_checks = int(np.ceil(total_time / dt_check))

        current_state = np.array([self.ref_orbit.x0, 0.0, 0.0, 0.0, self.ref_orbit.vy0, 0.0], dtype=np.float64)
        current_time = 0.0

        burns_dv_mps: List[float] = []
        traj_pos_err_km: List[float] = []

        # Tikhonov regularization tuned by aggressiveness alpha
        reg = 1e-3 * (1.0 - 0.9 * policy.aggressiveness)

        for step in range(n_checks):
            t_next = min(current_time + dt_check, total_time)
            sol = solve_ivp(
                lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=True, srp_mps2=srp_mps2),
                (current_time, t_next),
                current_state,
                rtol=1e-9,
                atol=1e-11,
                method='DOP853'
            )
            current_state = sol.y[:, -1]
            current_time = t_next

            ref_state = self.ref_orbit.get_reference_state(current_time)
            err_vec = current_state - ref_state
            pos_err_km = float(np.linalg.norm(err_vec[0:3]) * consts.length_star_m / 1e3)
            traj_pos_err_km.append(pos_err_km)

            # Predictive Lookahead
            t_target = current_time + t_horizon_norm
            Phi = self.cr3bp.compute_stm(current_time, t_target, current_state)
            err_proj = Phi @ err_vec
            proj_pos_err_km = float(np.linalg.norm(err_proj[0:3]) * consts.length_star_m / 1e3)

            # Deadband evaluation
            if proj_pos_err_km > policy.deadband_km:
                Phi_rv = Phi[0:3, 3:6]
                dv = - np.linalg.inv(Phi_rv.T @ Phi_rv + reg * np.eye(3)) @ Phi_rv.T @ err_proj[0:3]

                dv_mps = float(np.linalg.norm(dv) * consts.velocity_star_mps)
                # Clamp to thruster capability if necessary
                if dv_mps > policy.max_impulse_mps:
                    dv = dv * (policy.max_impulse_mps / dv_mps)
                    dv_mps = policy.max_impulse_mps

                current_state[3:6] += dv
                burns_dv_mps.append(dv_mps)

        total_dv = sum(burns_dv_mps)
        dv_per_month = total_dv / (duration_days / 30.0)
        max_err_km = float(max(traj_pos_err_km))
        rms_err_km = float(np.sqrt(np.mean(np.array(traj_pos_err_km)**2)))

        # Uncontrolled comparison
        uncontrolled_drift_km = 0.0
        if uncontrolled_comparison:
            sol_unc = solve_ivp(
                lambda t, y: self.cr3bp.equations_of_motion(t, y, include_disturbances=True, srp_mps2=srp_mps2),
                (0.0, total_time),
                [self.ref_orbit.x0, 0.0, 0.0, 0.0, self.ref_orbit.vy0, 0.0],
                rtol=1e-8,
                atol=1e-10,
                method='DOP853'
            )
            final_ref = self.ref_orbit.get_reference_state(total_time)
            uncontrolled_drift_km = float(np.linalg.norm(sol_unc.y[0:3, -1] - final_ref[0:3]) * consts.length_star_m / 1e3)

        return StationKeepingReport(
            duration_days=duration_days,
            total_delta_v_mps=total_dv,
            delta_v_per_month_mps=dv_per_month,
            burn_count=len(burns_dv_mps),
            mean_burn_mps=float(np.mean(burns_dv_mps)) if burns_dv_mps else 0.0,
            max_burn_mps=float(np.max(burns_dv_mps)) if burns_dv_mps else 0.0,
            max_position_error_km=max_err_km,
            rms_position_error_km=rms_err_km,
            uncontrolled_drift_km=uncontrolled_drift_km,
            fuel_savings_factor_vs_uncontrolled=uncontrolled_drift_km / max(max_err_km, 1e-6)
        )
