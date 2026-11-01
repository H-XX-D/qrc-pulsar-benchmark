"""Spacecraft 6-DOF thruster cluster geometry, control allocation, and attitude slewing.

Bridges translational orbital mechanics (delta-v) with physical spacecraft
actuation: individual thruster placement, line-of-action geometry, coupled
force-torque allocation, minimum impulse bit quantization, center-of-mass
travel, and quaternion-based attitude slew dynamics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import lsq_linear

G0 = 9.80665  # Standard sea-level gravitational acceleration (m/s^2)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / norm


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two unit quaternions q1 * q2 [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse) of a unit quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector v by unit quaternion q (body to inertial or vice-versa)."""
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    q_rot = quat_mult(quat_mult(q, q_v), quat_conjugate(q))
    return q_rot[1:]


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute unit quaternion rotating vector v_from into v_to."""
    u1 = _normalize(np.asarray(v_from, dtype=np.float64))
    u2 = _normalize(np.asarray(v_to, dtype=np.float64))
    dot = np.dot(u1, u2)
    if dot >= 1.0 - 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot <= -1.0 + 1e-10:
        # 180 degree rotation: pick an orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(u1, ortho))
        return np.array([0.0, axis[0], axis[1], axis[2]])
    cross = np.cross(u1, u2)
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]])
    return _normalize(q)


@dataclass(frozen=True)
class Thruster:
    """Physical thruster mount specification."""
    id: int
    name: str
    position_m: np.ndarray       # Position in body frame relative to geometric origin [x, y, z]
    direction_body: np.ndarray   # Unit thrust vector in body frame [nx, ny, nz]
    thrust_n: float              # Maximum steady-state thrust in Newtons
    isp_s: float                 # Specific impulse in seconds
    min_pulse_s: float = 0.01    # Minimum impulse bit (MIB) pulse duration (10 ms)
    healthy: bool = True         # Operational status
    stuck_open: bool = False     # Fault status: firing uncontrollably

    def __post_init__(self):
        pos = np.asarray(self.position_m, dtype=np.float64)
        dir_b = np.asarray(self.direction_body, dtype=np.float64)
        if pos.shape != (3,) or dir_b.shape != (3,):
            raise ValueError("Position and direction must be 3D vectors")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(dir_b)):
            raise ValueError("Thruster parameters must be finite")
        if self.thrust_n <= 0 or self.isp_s <= 0 or self.min_pulse_s < 0:
            raise ValueError("Thrust, Isp, and min_pulse must be positive")
        norm = np.linalg.norm(dir_b)
        if abs(norm - 1.0) > 1e-5:
            object.__setattr__(self, "direction_body", dir_b / norm)


@dataclass
class SpacecraftBus:
    """Physical mass, geometry, and moment of inertia characteristics."""
    dry_mass_kg: float = 400.0
    propellant_mass_kg: float = 100.0
    dimensions_m: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 1.5]))
    inertia_dry: np.ndarray = field(default_factory=lambda: np.diag([120.0, 130.0, 100.0]))
    com_dry_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    com_prop_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.2]))

    def total_mass(self) -> float:
        return self.dry_mass_kg + max(0.0, self.propellant_mass_kg)

    def center_of_mass(self) -> np.ndarray:
        """Mass-weighted center of mass in body coordinates."""
        m_tot = self.total_mass()
        m_prop = max(0.0, self.propellant_mass_kg)
        return (self.dry_mass_kg * self.com_dry_m + m_prop * self.com_prop_m) / m_tot

    def inertia_tensor(self) -> np.ndarray:
        """Effective inertia tensor around the current center of mass (Parallel Axis Theorem)."""
        r_cm = self.center_of_mass()
        d_dry = self.com_dry_m - r_cm
        d_prop = self.com_prop_m - r_cm
        
        # Parallel axis theorem: I_new = I_cm + m * (|d|^2 I - d (x) d)
        def shift_tensor(m, d):
            return m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        
        i_prop_tank = max(0.0, self.propellant_mass_kg) * np.diag([0.1, 0.1, 0.1])
        i_total = (self.inertia_dry + shift_tensor(self.dry_mass_kg, d_dry) +
                   i_prop_tank + shift_tensor(max(0.0, self.propellant_mass_kg), d_prop))
        return i_total


class RCSCluster:
    """Full 6-DOF reaction control cluster geometry and configuration."""
    def __init__(self, thrusters: List[Thruster]):
        if len(thrusters) < 6:
            raise ValueError("At least 6 thrusters required for 6-DOF control authority")
        self.thrusters = {t.id: t for t in thrusters}

    @classmethod
    def standard_12_thruster_cluster(cls, radius_m: float = 0.6, height_m: float = 0.75,
                                     thrust_n: float = 1.0, isp_s: float = 220.0) -> "RCSCluster":
        """Construct standard 12-thruster RCS layout: 4 pods with 3 thrusters each.
        
        Provides complete, decoupled 3-axis translation and 3-axis rotation.
        """
        thrusters = []
        tid = 0
        
        # 4 pods placed at +/- X, +/- Y at z = +/- height
        pod_configs = [
            # Pod 1: +X, +Y, +Z quadrant
            (np.array([radius_m, radius_m, height_m]), [
                ("P1_POS_X", np.array([1.0, 0.0, 0.0])),
                ("P1_POS_Y", np.array([0.0, 1.0, 0.0])),
                ("P1_POS_Z", np.array([0.0, 0.0, 1.0])),
            ]),
            # Pod 2: -X, +Y, -Z quadrant
            (np.array([-radius_m, radius_m, -height_m]), [
                ("P2_NEG_X", np.array([-1.0, 0.0, 0.0])),
                ("P2_POS_Y", np.array([0.0, 1.0, 0.0])),
                ("P2_NEG_Z", np.array([0.0, 0.0, -1.0])),
            ]),
            # Pod 3: -X, -Y, +Z quadrant
            (np.array([-radius_m, -radius_m, height_m]), [
                ("P3_NEG_X", np.array([-1.0, 0.0, 0.0])),
                ("P3_NEG_Y", np.array([0.0, -1.0, 0.0])),
                ("P3_POS_Z", np.array([0.0, 0.0, 1.0])),
            ]),
            # Pod 4: +X, -Y, -Z quadrant
            (np.array([radius_m, -radius_m, -height_m]), [
                ("P4_POS_X", np.array([1.0, 0.0, 0.0])),
                ("P4_NEG_Y", np.array([0.0, -1.0, 0.0])),
                ("P4_NEG_Z", np.array([0.0, 0.0, -1.0])),
            ]),
        ]
        
        for pos, directions in pod_configs:
            for name, d in directions:
                thrusters.append(Thruster(
                    id=tid,
                    name=name,
                    position_m=pos,
                    direction_body=d,
                    thrust_n=thrust_n,
                    isp_s=isp_s,
                    min_pulse_s=0.01
                ))
                tid += 1
                
        return cls(thrusters)

    def configuration_matrix(self, r_cm: np.ndarray) -> np.ndarray:
        """Compute the 6 x M Thruster Configuration Matrix B_T.
        
        First 3 rows: net force in body frame (F_x, F_y, F_z).
        Last 3 rows: net torque around r_cm (tau_x, tau_y, tau_z).
        """
        n_thrusters = len(self.thrusters)
        b_matrix = np.zeros((6, n_thrusters), dtype=np.float64)
        
        for i, (tid, thruster) in enumerate(sorted(self.thrusters.items())):
            n_hat = thruster.direction_body
            r_arm = thruster.position_m - r_cm
            torque_arm = np.cross(r_arm, n_hat)
            
            b_matrix[:3, i] = n_hat
            b_matrix[3:, i] = torque_arm
            
        return b_matrix

    def set_fault(self, thruster_id: int, fault_type: str):
        """Inject thruster fault: 'stuck_closed', 'stuck_open', or 'healthy'."""
        if thruster_id not in self.thrusters:
            raise KeyError(f"Thruster ID {thruster_id} not in cluster")
        t = self.thrusters[thruster_id]
        if fault_type == "stuck_closed":
            self.thrusters[thruster_id] = Thruster(
                id=t.id, name=t.name, position_m=t.position_m, direction_body=t.direction_body,
                thrust_n=t.thrust_n, isp_s=t.isp_s, min_pulse_s=t.min_pulse_s,
                healthy=False, stuck_open=False
            )
        elif fault_type == "stuck_open":
            self.thrusters[thruster_id] = Thruster(
                id=t.id, name=t.name, position_m=t.position_m, direction_body=t.direction_body,
                thrust_n=t.thrust_n, isp_s=t.isp_s, min_pulse_s=t.min_pulse_s,
                healthy=False, stuck_open=True
            )
        elif fault_type == "healthy":
            self.thrusters[thruster_id] = Thruster(
                id=t.id, name=t.name, position_m=t.position_m, direction_body=t.direction_body,
                thrust_n=t.thrust_n, isp_s=t.isp_s, min_pulse_s=t.min_pulse_s,
                healthy=True, stuck_open=False
            )
        else:
            raise ValueError(f"Unknown fault type: {fault_type}")


@dataclass(frozen=True)
class StructuralFrame:
    """Spacecraft bus structural and elastodynamic properties."""
    bus_length_m: float = 1.5                   # Distance between opposing thruster mounting decks
    strut_area_m2: float = 0.002                # Cross-sectional load-bearing area (20 cm^2)
    youngs_modulus_pa: float = 70.0e9           # Aluminum-lithium / Carbon composite modulus (70 GPa)
    fundamental_freq_hz: float = 2.5            # Fundamental flexible bending mode (e.g. solar array / bus)
    damping_ratio: float = 0.005                # Internal material & joint damping ratio in vacuum (0.5%)

    @property
    def stiffness_n_per_m(self) -> float:
        """Axial structural stiffness K = E * A / L."""
        return (self.youngs_modulus_pa * self.strut_area_m2) / self.bus_length_m

    @property
    def modal_decay_time_s(self) -> float:
        """1/e vibrational amplitude decay time tau = 1 / (zeta * omega_n)."""
        omega_n = 2.0 * np.pi * self.fundamental_freq_hz
        return 1.0 / (self.damping_ratio * omega_n)


@dataclass
class StructuralStressMetrics:
    """Stress, strain energy, and vibrational dissipation from thruster firings."""
    opposing_force_n: float                     # Antagonistic compressive force across the bus
    frame_stress_mpa: float                     # Internal compressive stress sigma = F / A
    strain_energy_j: float                      # Stored elastic strain energy U = F^2 L / (2 E A)
    vibrational_energy_j: float                 # Peak modal kinetic/potential vibration energy
    vibration_decay_time_s: float               # Time required for structural ringing to settle
    thermal_dissipation_rate_w: float           # Rate at which vibration converts to heat via joint friction


@dataclass
class AllocationResult:
    """Outcome of control allocation solve."""
    success: bool
    pulse_durations_s: np.ndarray               # Firing duration for each thruster in seconds
    thrust_levels_n: np.ndarray                 # Commanded thrust for each thruster in Newtons
    commanded_wrench: np.ndarray                # Requested [Fx, Fy, Fz, tau_x, tau_y, tau_z]
    achieved_wrench: np.ndarray                 # Realized [Fx, Fy, Fz, tau_x, tau_y, tau_z]
    wrench_residual: np.ndarray                 # Commanded - Achieved
    propellant_used_kg: float                   # Propellant consumed by this allocation
    active_thruster_count: int                  # Number of thrusters fired
    structural_metrics: StructuralStressMetrics # Internal frame stress and vibrational dissipation


class ThrusterAllocator:
    """Solves bounded, non-negative control allocation for coupled force and torque."""
    def __init__(self, cluster: RCSCluster, interval_s: float = 1.0,
                 force_weight: float = 1.0, torque_weight: float = 10.0,
                 fuel_penalty: float = 0.01,
                 frame: StructuralFrame = StructuralFrame()):
        self.cluster = cluster
        self.interval_s = float(interval_s)
        self.force_weight = float(force_weight)
        self.torque_weight = float(torque_weight)
        self.fuel_penalty = float(fuel_penalty)
        self.frame = frame

    def allocate(self, wrench_des: np.ndarray, r_cm: np.ndarray) -> AllocationResult:
        """Map commanded 6D wrench [F_body (N); tau_body (N*m)] to thruster pulses."""
        wrench_des = np.asarray(wrench_des, dtype=np.float64)
        if wrench_des.shape != (6,):
            raise ValueError("Wrench must be 6-element vector [F_x, F_y, F_z, tau_x, tau_y, tau_z]")

        thrusters = [self.cluster.thrusters[tid] for tid in sorted(self.cluster.thrusters.keys())]
        n = len(thrusters)
        b_matrix = self.cluster.configuration_matrix(r_cm)

        # Upper bounds for thrust
        ub = np.zeros(n, dtype=np.float64)
        lb = np.zeros(n, dtype=np.float64)
        stuck_open_wrench = np.zeros(6, dtype=np.float64)

        # Identify free vs fixed variables
        free_mask = np.zeros(n, dtype=bool)
        fixed_values = np.zeros(n, dtype=np.float64)

        for i, t in enumerate(thrusters):
            if t.stuck_open:
                fixed_values[i] = t.thrust_n
            elif t.healthy:
                free_mask[i] = True
                lb[i] = 0.0
                ub[i] = t.thrust_n
            else:
                # Stuck closed
                fixed_values[i] = 0.0

        # Weighted least-squares objective: || W_diag * (B * T - wrench_des) ||^2 + fuel_penalty * ||T||^2
        w_diag = np.array([self.force_weight]*3 + [self.torque_weight]*3)
        a_weighted = w_diag[:, None] * b_matrix
        b_weighted = w_diag * wrench_des

        # Regularization for fuel minimization
        reg_a = np.vstack([a_weighted, np.sqrt(self.fuel_penalty) * np.eye(n)])
        reg_b = np.concatenate([b_weighted, np.zeros(n)])

        t_optimal = np.zeros(n, dtype=np.float64)
        t_optimal[~free_mask] = fixed_values[~free_mask]

        if np.any(free_mask):
            # Subtract contribution of fixed thrusters from target vector
            fixed_contrib = reg_a[:, ~free_mask] @ fixed_values[~free_mask]
            reg_b_free = reg_b - fixed_contrib
            reg_a_free = reg_a[:, free_mask]
            lb_free = lb[free_mask]
            ub_free = ub[free_mask]

            res = lsq_linear(reg_a_free, reg_b_free, bounds=(lb_free, ub_free), method='bvls')
            t_optimal[free_mask] = np.maximum(0.0, res.x)
            success = bool(res.success)
        else:
            success = True

        # Compute pulse durations
        durations = np.zeros(n, dtype=np.float64)
        prop_mass = 0.0
        active_count = 0

        for i, t in enumerate(thrusters):
            if t_optimal[i] > 1e-6:
                # Pulse modulation over interval
                duty_fraction = t_optimal[i] / t.thrust_n
                dt_pulse = duty_fraction * self.interval_s
                
                # Check Minimum Impulse Bit (MIB)
                if dt_pulse >= t.min_pulse_s or t.stuck_open:
                    durations[i] = dt_pulse
                    m_dot = t.thrust_n / (t.isp_s * G0)
                    prop_mass += m_dot * dt_pulse
                    active_count += 1
                else:
                    # Below MIB: truncate to 0 unless stuck open
                    durations[i] = 0.0
                    t_optimal[i] = 0.0

        achieved_wrench = b_matrix @ t_optimal
        residual = wrench_des - achieved_wrench

        # Structural stress and vibrational dissipation mechanics
        frame = self.frame
        opposing_n = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if np.dot(thrusters[i].direction_body, thrusters[j].direction_body) < -0.9:
                    opposing_n += float(min(t_optimal[i], t_optimal[j]))

        stress_pa = opposing_n / frame.strut_area_m2
        stress_mpa = stress_pa / 1e6
        strain_energy_j = (opposing_n ** 2 * frame.bus_length_m) / (2.0 * frame.youngs_modulus_pa * frame.strut_area_m2) if opposing_n > 0 else 0.0
        vib_energy_j = (opposing_n ** 2) / (2.0 * frame.stiffness_n_per_m) if opposing_n > 0 else 0.0
        omega_n = 2.0 * np.pi * frame.fundamental_freq_hz
        dissipation_rate_w = 2.0 * frame.damping_ratio * omega_n * vib_energy_j if vib_energy_j > 0 else 0.0

        struct_metrics = StructuralStressMetrics(
            opposing_force_n=float(opposing_n),
            frame_stress_mpa=float(stress_mpa),
            strain_energy_j=float(strain_energy_j),
            vibrational_energy_j=float(vib_energy_j),
            vibration_decay_time_s=float(frame.modal_decay_time_s),
            thermal_dissipation_rate_w=float(dissipation_rate_w)
        )

        return AllocationResult(
            success=bool(success),
            pulse_durations_s=durations,
            thrust_levels_n=t_optimal,
            commanded_wrench=wrench_des,
            achieved_wrench=achieved_wrench,
            wrench_residual=residual,
            propellant_used_kg=float(prop_mass),
            active_thruster_count=active_count,
            structural_metrics=struct_metrics
        )


@dataclass
class SlewProfile:
    """Planned rotational slew maneuver profile."""
    initial_quat: np.ndarray
    target_quat: np.ndarray
    slew_angle_rad: float
    slew_axis_body: np.ndarray
    duration_s: float
    accel_time_s: float
    coast_time_s: float
    max_rate_rad_s: float
    max_accel_rad_s2: float
    propellant_used_kg: float


class AttitudeSlewPlanner:
    """Plans rotational reorientation maneuvers and calculates real-world slew latency."""
    def __init__(self, bus: SpacecraftBus, cluster: RCSCluster,
                 max_angular_rate_deg_s: float = 2.0,
                 max_angular_accel_deg_s2: float = 0.5):
        self.bus = bus
        self.cluster = cluster
        self.max_rate_rad_s = np.radians(max_angular_rate_deg_s)
        self.max_accel_rad_s2 = np.radians(max_angular_accel_deg_s2)

    def plan_slew_to_direction(self, q_current: np.ndarray,
                              desired_inertial_dir: np.ndarray,
                              body_thrust_axis: np.ndarray = np.array([0.0, 0.0, 1.0])) -> SlewProfile:
        """Plan slew maneuver to point spacecraft's body thrust axis along desired inertial vector."""
        current_thrust_dir_inertial = quat_rotate(q_current, body_thrust_axis)
        q_rot = quat_from_two_vectors(current_thrust_dir_inertial, desired_inertial_dir)
        q_target = quat_mult(q_rot, q_current)
        q_target = _normalize(q_target)
        return self.plan_slew(q_current, q_target)

    def plan_slew(self, q_start: np.ndarray, q_target: np.ndarray) -> SlewProfile:
        """Compute trapezoidal or triangular angular rate slew profile."""
        q_start = _normalize(np.asarray(q_start, dtype=np.float64))
        q_target = _normalize(np.asarray(q_target, dtype=np.float64))

        # Relative quaternion: delta_q = q_target * q_start^-1
        delta_q = quat_mult(q_target, quat_conjugate(q_start))
        if delta_q[0] < 0.0:
            delta_q = -delta_q  # Shortest path

        # Slew angle
        w_clamped = np.clip(delta_q[0], -1.0, 1.0)
        theta = 2.0 * np.arccos(w_clamped)

        if theta < 1e-6:
            # Already aligned
            return SlewProfile(
                initial_quat=q_start, target_quat=q_target, slew_angle_rad=0.0,
                slew_axis_body=np.array([1.0, 0.0, 0.0]), duration_s=0.0,
                accel_time_s=0.0, coast_time_s=0.0, max_rate_rad_s=0.0,
                max_accel_rad_s2=0.0, propellant_used_kg=0.0
            )

        axis_norm = np.linalg.norm(delta_q[1:])
        axis = delta_q[1:] / axis_norm if axis_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

        alpha = self.max_accel_rad_s2
        omega_max = self.max_rate_rad_s

        # Check if triangular profile suffices
        theta_crit = (omega_max ** 2) / alpha
        if theta <= theta_crit:
            # Triangular profile (accelerate then decelerate immediately)
            t_accel = np.sqrt(theta / alpha)
            t_coast = 0.0
            peak_omega = alpha * t_accel
            total_duration = 2.0 * t_accel
        else:
            # Trapezoidal profile (accelerate, coast at omega_max, decelerate)
            t_accel = omega_max / alpha
            theta_accel = 0.5 * alpha * (t_accel ** 2)
            theta_coast = theta - 2.0 * theta_accel
            t_coast = theta_coast / omega_max
            peak_omega = omega_max
            total_duration = 2.0 * t_accel + t_coast

        # Slew fuel estimate: firing torque pair to spin up and spin down
        i_tensor = self.bus.inertia_tensor()
        i_axis = float(axis.T @ i_tensor @ axis)
        required_torque = i_axis * alpha
        
        # Allocate required torque through RCS
        allocator = ThrusterAllocator(self.cluster, interval_s=t_accel)
        torque_cmd = axis * required_torque
        res_accel = allocator.allocate(np.concatenate([np.zeros(3), torque_cmd]), self.bus.center_of_mass())
        res_decel = allocator.allocate(np.concatenate([np.zeros(3), -torque_cmd]), self.bus.center_of_mass())
        
        fuel_total = res_accel.propellant_used_kg + res_decel.propellant_used_kg

        return SlewProfile(
            initial_quat=q_start,
            target_quat=q_target,
            slew_angle_rad=float(theta),
            slew_axis_body=axis,
            duration_s=float(total_duration),
            accel_time_s=float(t_accel),
            coast_time_s=float(t_coast),
            max_rate_rad_s=float(peak_omega),
            max_accel_rad_s2=float(alpha),
            propellant_used_kg=float(fuel_total)
        )
