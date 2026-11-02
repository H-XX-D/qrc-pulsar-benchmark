"""
Flight-Critical Spacecraft Dynamics & Environmental Perturbation Models.

Models the fundamental physical phenomena required for operational deep-space GN&C:
1. Solar Radiation Pressure (SRP) Force & Center-of-Pressure (CoP) Disturbance Torque.
2. Reaction Wheel Assembly (RWA) 3D Momentum Management, Saturation, & RCS Desaturation Dumping.
3. Propellant Slosh Dynamics (spring-mass-damper analog) & Zero-G Ullage Settling Checks.
4. Gravity Gradient Disturbance Torques in arbitrary central gravitational fields.
5. Star Tracker Optical Blinding (Sun/Moon/Earth/Plume exclusion cones) & Relativistic Stellar Aberration.
6. Eclipse Thermal Snap Mechanical Angular Momentum Shock Impulse.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .constants import AU, c, G
from .cislunar_forces import (
    SOLAR_PRESSURE_1AU,
    sunlight_fraction,
    _vector,
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    R_EARTH,
    R_MOON,
    R_SUN,
)
from .thruster_allocation import RCSCluster, ThrusterAllocator, AllocationResult, SpacecraftBus


# ====================================================================================================
# 1. SOLAR RADIATION PRESSURE & DISTURBANCE TORQUE
# ====================================================================================================

@dataclass(frozen=True)
class OpticalSurface:
    """An individual optical plate/facet on the spacecraft exterior."""
    name: str
    area_m2: float
    normal_body: Tuple[float, float, float]
    center_of_pressure_body_m: Tuple[float, float, float]
    specular: float = 0.0
    diffuse: float = 0.0

    def __post_init__(self):
        n = np.asarray(self.normal_body, dtype=float)
        cop = np.asarray(self.center_of_pressure_body_m, dtype=float)
        if self.area_m2 <= 0 or not np.all(np.isfinite([self.area_m2, self.specular, self.diffuse])):
            raise ValueError(f"Invalid area or optical properties for plate {self.name}")
        if self.specular < 0 or self.diffuse < 0 or (self.specular + self.diffuse) > 1.0:
            raise ValueError(f"Optical fractions must be non-negative and sum <= 1.0 for plate {self.name}")
        if n.shape != (3,) or not np.isclose(np.linalg.norm(n), 1.0, atol=1e-8):
            raise ValueError(f"Normal vector for plate {self.name} must be a unit 3-vector")
        if cop.shape != (3,) or not np.all(np.isfinite(cop)):
            raise ValueError(f"Center of pressure for plate {self.name} must be a finite 3-vector")
        object.__setattr__(self, "normal_body", tuple(n))
        object.__setattr__(self, "center_of_pressure_body_m", tuple(cop))


@dataclass
class SolarRadiationTorqueResult:
    """Result of full solar radiation force and torque evaluation."""
    force_icrf_n: np.ndarray             # Translational SRP force in inertial ICRF [N]
    force_body_n: np.ndarray             # Translational SRP force in body frame [N]
    torque_body_nm: np.ndarray           # Disturbance torque about Center of Mass in body frame [N*m]
    sunlight_fraction: float             # Eclipse factor (1.0 = full sun, 0.0 = total umbra)
    sun_direction_body: np.ndarray       # Unit vector toward the Sun in body frame


class SolarRadiationTorqueModel:
    """Calculates attitude-dependent SRP force and CoP-CoM lever-arm disturbance torque."""
    def __init__(self, surfaces: List[OpticalSurface]):
        if not surfaces:
            raise ValueError("SolarRadiationTorqueModel requires at least one optical surface")
        self.surfaces = tuple(surfaces)

    def evaluate(self, r_sc_icrf: np.ndarray, r_sun_icrf: np.ndarray,
                 dcm_body_to_icrf: np.ndarray, center_of_mass_body_m: np.ndarray,
                 r_earth_icrf: Optional[np.ndarray] = None) -> SolarRadiationTorqueResult:
        """
        Compute SRP force and torque.
        dcm_body_to_icrf: Direction Cosine Matrix mapping vectors from body frame to ICRF frame.
        """
        r_sc = _vector(r_sc_icrf, "r_sc_icrf")
        r_sun = _vector(r_sun_icrf, "r_sun_icrf")
        r_cm = _vector(center_of_mass_body_m, "center_of_mass_body_m")
        dcm = np.asarray(dcm_body_to_icrf, dtype=float)
        if dcm.shape != (3, 3) or not np.allclose(dcm.T @ dcm, np.eye(3), atol=1e-6):
            raise ValueError("dcm_body_to_icrf must be a valid 3x3 rotation matrix")

        # Sun vector in inertial and body frames
        to_sun_icrf = r_sun - r_sc
        dist_sun = float(np.linalg.norm(to_sun_icrf))
        if dist_sun < R_SUN:
            raise ValueError("Spacecraft inside Sun radius")
        dir_sun_icrf = to_sun_icrf / dist_sun

        # Transform Sun vector to body frame: v_body = DCM^T * v_icrf
        dir_sun_body = dcm.T @ dir_sun_icrf

        # Eclipse fraction
        if r_earth_icrf is not None:
            vis = sunlight_fraction(r_sc, r_earth_icrf, r_sun)
        else:
            vis = 1.0

        p_srp = SOLAR_PRESSURE_1AU * ((AU / dist_sun) ** 2) * vis

        total_force_body = np.zeros(3)
        total_torque_body = np.zeros(3)

        for s in self.surfaces:
            n_b = np.array(s.normal_body)
            cop_b = np.array(s.center_of_pressure_body_m)
            cos_inc = float(np.dot(n_b, dir_sun_body))
            if cos_inc <= 0.0:
                continue  # Back-side illumination (shadowed plate)

            # Flat-plate radiation pressure force in body frame
            # F = -P * A * cos(theta) * [(1 - rho_s) * s_dir + 2 * (rho_s * cos(theta) + rho_d / 3) * n]
            f_plate = -p_srp * s.area_m2 * cos_inc * (
                (1.0 - s.specular) * dir_sun_body
                + 2.0 * (s.specular * cos_inc + s.diffuse / 3.0) * n_b
            )
            total_force_body += f_plate

            # Torque = (r_cop - r_cm) x F_plate
            lever_arm = cop_b - r_cm
            total_torque_body += np.cross(lever_arm, f_plate)

        # Force in inertial ICRF frame
        total_force_icrf = dcm @ total_force_body

        return SolarRadiationTorqueResult(
            force_icrf_n=total_force_icrf,
            force_body_n=total_force_body,
            torque_body_nm=total_torque_body,
            sunlight_fraction=float(vis),
            sun_direction_body=dir_sun_body,
        )


# ====================================================================================================
# 2. REACTION WHEEL ASSEMBLY (RWA) & MOMENTUM MANAGEMENT
# ====================================================================================================

@dataclass
class ReactionWheel:
    """Single reaction wheel mechanical characteristics."""
    spin_axis_body: Tuple[float, float, float]
    inertia_kg_m2: float = 0.04          # Rotor spin inertia [kg*m^2]
    max_speed_rpm: float = 6000.0        # Maximum rotor angular velocity [RPM]
    max_torque_nm: float = 0.2           # Maximum motor torque [N*m]
    static_unbalance_g_mm: float = 2.0   # Static unbalance mass-distance [g*mm]
    dynamic_unbalance_g_mm2: float = 20.0 # Dynamic unbalance product of inertia [g*mm^2]

    def __post_init__(self):
        ax = np.asarray(self.spin_axis_body, dtype=float)
        if ax.shape != (3,) or not np.isclose(np.linalg.norm(ax), 1.0, atol=1e-8):
            raise ValueError("Wheel spin axis must be a unit 3-vector")
        object.__setattr__(self, "spin_axis_body", tuple(ax))

    @property
    def max_speed_rad_s(self) -> float:
        return self.max_speed_rpm * (2.0 * np.pi / 60.0)


@dataclass
class MomentumDumpPlan:
    """Plan for desaturating reaction wheels using external RCS thruster couples."""
    required_torque_nm: np.ndarray       # External torque required from thrusters [N*m]
    target_momentum_nm_s: np.ndarray     # Residual momentum target [N*m*s]
    current_momentum_nm_s: np.ndarray    # Current wheel array momentum in body frame [N*m*s]
    burn_duration_s: float               # Duration of RCS firing [s]
    allocation: AllocationResult         # RCS cluster allocation result
    propellant_used_kg: float            # Propellant consumed by the dump


class ReactionWheelArray:
    """Manages an array of reaction wheels: torque allocation, saturation, and desaturation."""
    def __init__(self, wheels: List[ReactionWheel]):
        if len(wheels) < 3:
            raise ValueError("Reaction wheel array requires at least 3 wheels for 3-axis control")
        self.wheels = tuple(wheels)
        self.num_wheels = len(wheels)
        # Mounting matrix A_w in R^(3 x M): columns are unit spin axes
        self.A_w = np.column_stack([w.spin_axis_body for w in self.wheels])
        if np.linalg.matrix_rank(self.A_w) < 3:
            raise ValueError("Reaction wheel mounting matrix must have full rank 3")
        # Current wheel angular speeds [rad/s]
        self.wheel_speeds_rad_s = np.zeros(self.num_wheels)

    @classmethod
    def standard_3_axis_orthogonal(cls, inertia_kg_m2: float = 0.04,
                                   max_speed_rpm: float = 6000.0,
                                   max_torque_nm: float = 0.2) -> "ReactionWheelArray":
        """Orthogonal 3-wheel configuration aligned with body X, Y, Z."""
        wheels = [
            ReactionWheel(spin_axis_body=(1.0, 0.0, 0.0), inertia_kg_m2=inertia_kg_m2,
                          max_speed_rpm=max_speed_rpm, max_torque_nm=max_torque_nm),
            ReactionWheel(spin_axis_body=(0.0, 1.0, 0.0), inertia_kg_m2=inertia_kg_m2,
                          max_speed_rpm=max_speed_rpm, max_torque_nm=max_torque_nm),
            ReactionWheel(spin_axis_body=(0.0, 0.0, 1.0), inertia_kg_m2=inertia_kg_m2,
                          max_speed_rpm=max_speed_rpm, max_torque_nm=max_torque_nm),
        ]
        return cls(wheels)

    @classmethod
    def standard_4_wheel_pyramid(cls, skew_angle_deg: float = 35.26438968,
                                 inertia_kg_m2: float = 0.04,
                                 max_speed_rpm: float = 6000.0,
                                 max_torque_nm: float = 0.2) -> "ReactionWheelArray":
        """
        Standard 4-wheel pyramid configuration offering isotropic control and 1-wheel redundancy.
        skew_angle is the elevation angle above the X-Y plane.
        beta = arcsin(1/sqrt(3)) ~ 35.264 deg yields equal 3-axis authority:
        A_w A_w^T = diag(2*cos^2(beta), 2*cos^2(beta), 4*sin^2(beta)) = diag(4/3, 4/3, 4/3).
        (Note: 54.74 deg is the zenith half-angle from the Z-axis, which would cause a 4:1 imbalance).
        """
        beta = np.radians(skew_angle_deg)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        # 4 wheels canted around Z axis at 0, 90, 180, 270 deg azimuth
        axes = [
            (cos_b, 0.0, sin_b),
            (0.0, cos_b, sin_b),
            (-cos_b, 0.0, sin_b),
            (0.0, -cos_b, sin_b),
        ]
        wheels = [
            ReactionWheel(spin_axis_body=ax, inertia_kg_m2=inertia_kg_m2,
                          max_speed_rpm=max_speed_rpm, max_torque_nm=max_torque_nm)
            for ax in axes
        ]
        return cls(wheels)

    @property
    def wheel_inertias(self) -> np.ndarray:
        return np.array([w.inertia_kg_m2 for w in self.wheels])

    @property
    def max_speeds_rad_s(self) -> np.ndarray:
        return np.array([w.max_speed_rad_s for w in self.wheels])

    @property
    def max_torques_nm(self) -> np.ndarray:
        return np.array([w.max_torque_nm for w in self.wheels])

    def wheel_angular_momentum_vector(self) -> np.ndarray:
        """Total reaction wheel angular momentum expressed in body frame [N*m*s]."""
        # h_i = I_w,i * omega_w,i
        h_wheels = self.wheel_inertias * self.wheel_speeds_rad_s
        return self.A_w @ h_wheels

    def allocate_torque(self, commanded_torque_body_nm: np.ndarray) -> np.ndarray:
        """
        Allocate commanded attitude control torque to wheel motor torques:
        tau_cmd = -A_w * u_motor  ==>  u_motor = -A_w^+ * tau_cmd.
        Clips to motor torque limits.
        """
        tau_cmd = _vector(commanded_torque_body_nm, "commanded_torque_body_nm")
        # Minimum-norm pseudo-inverse allocation: A_w^+ = A_w^T (A_w A_w^T)^-1
        pinv_a = np.linalg.pinv(self.A_w)
        u_raw = -pinv_a @ tau_cmd
        # Enforce motor torque saturation limits
        u_clipped = np.clip(u_raw, -self.max_torques_nm, self.max_torques_nm)
        return u_clipped

    def step(self, dt_s: float, motor_torques_nm: np.ndarray) -> np.ndarray:
        """
        Integrate wheel spin speeds: omega_w(t + dt) = omega_w(t) + (u_motor / I_w) * dt.
        Returns achieved reaction torque on the spacecraft bus: tau_bus = -A_w * u_motor.
        """
        u = np.asarray(motor_torques_nm, dtype=float)
        # Check motor limits
        u = np.clip(u, -self.max_torques_nm, self.max_torques_nm)
        d_omega = (u / self.wheel_inertias) * float(dt_s)
        self.wheel_speeds_rad_s += d_omega
        # Clip wheel speeds to physical breakdown limits
        self.wheel_speeds_rad_s = np.clip(self.wheel_speeds_rad_s, -self.max_speeds_rad_s, self.max_speeds_rad_s)
        # Reactive torque imparted to spacecraft bus
        return -self.A_w @ u

    def check_desaturation(self, threshold_fraction: float = 0.8) -> Tuple[bool, np.ndarray]:
        """
        Check if any wheel exceeds threshold_fraction of its maximum speed.
        Returns (desaturation_required, wheel_speed_ratios).
        """
        ratios = np.abs(self.wheel_speeds_rad_s) / self.max_speeds_rad_s
        needs_dump = bool(np.any(ratios >= float(threshold_fraction)))
        return needs_dump, ratios

    def plan_momentum_dump(self, allocator: ThrusterAllocator, r_cm_body_m: np.ndarray,
                           duration_s: float = 5.0,
                           target_momentum_nm_s: Optional[np.ndarray] = None) -> MomentumDumpPlan:
        """
        Plan an RCS momentum desaturation burn to dump wheel angular momentum without linear translation.
        Delta H = H_wheels - H_target. Required RCS torque: tau_rcs = Delta H / duration_s.
        """
        if duration_s <= 0:
            raise ValueError("Dump duration must be positive")
        current_h = self.wheel_angular_momentum_vector()
        if target_momentum_nm_s is None:
            target_h = np.zeros(3)
        else:
            target_h = _vector(target_momentum_nm_s, "target_momentum_nm_s")

        # External RCS torque to desaturate wheels:
        # To change wheel momentum from current_h to target_h over duration_s,
        # the external torque imparted to the spacecraft is tau_rcs = (target_h - current_h) / duration_s
        tau_rcs_des = (target_h - current_h) / float(duration_s)

        # Commanded wrench: 0 force, tau_rcs torque
        cmd_wrench = np.r_[np.zeros(3), tau_rcs_des]
        alloc_res = allocator.allocate(cmd_wrench, r_cm_body_m)

        return MomentumDumpPlan(
            required_torque_nm=tau_rcs_des,
            target_momentum_nm_s=target_h,
            current_momentum_nm_s=current_h,
            burn_duration_s=float(duration_s),
            allocation=alloc_res,
            propellant_used_kg=alloc_res.propellant_used_kg,
        )

    def micro_vibration_jitter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate high-frequency micro-vibration disturbances caused by wheel imbalances:
        F_static = sum( U_s,i * omega_i^2 ),  Tau_dynamic = sum( U_d,i * omega_i^2 ).
        Returns (jitter_force_n, jitter_torque_nm).
        """
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for i, w in enumerate(self.wheels):
            omega = self.wheel_speeds_rad_s[i]
            # Static unbalance: F_s = (m * r) * omega^2 = U_s [kg*m] * omega^2
            u_s_kg_m = w.static_unbalance_g_mm * 1e-6
            f_mag = u_s_kg_m * (omega ** 2)
            # Dynamic unbalance: Tau_d = I_xz * omega^2 = U_d [kg*m^2] * omega^2
            # 1 g*mm^2 = 1e-3 kg * (1e-3 m)^2 = 1e-9 kg*m^2
            u_d_kg_m2 = w.dynamic_unbalance_g_mm2 * 1e-9
            tau_mag = u_d_kg_m2 * (omega ** 2)

            # Jitter acts radially perpendicular to the spin axis
            ax = np.array(w.spin_axis_body)
            # Find an arbitrary perpendicular unit vector
            perp = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = np.cross(ax, perp)
            perp /= np.linalg.norm(perp)

            total_force += f_mag * perp
            total_torque += tau_mag * np.cross(ax, perp)

        return total_force, total_torque


# ====================================================================================================
# 3. PROPELLANT TANK SLOSH DYNAMICS & ZERO-G ULLAGE SETTLING
# ====================================================================================================

@dataclass
class SloshState:
    """Dynamic state of the liquid slosh oscillator."""
    displacement_m: np.ndarray          # 2D lateral displacement in tank plane [m]
    velocity_m_s: np.ndarray            # 2D lateral velocity [m/s]
    reaction_force_body_n: np.ndarray   # 3D force exerted on the bus [N]
    reaction_torque_body_nm: np.ndarray # 3D torque exerted on the bus [N*m]


class PropellantTankSlosh:
    """
    Mechanical spring-mass-damper analog representing low-gravity liquid slosh.
    Models fluid-structure interaction during burns and checks zero-g ullage settling.
    """
    def __init__(self, tank_location_body_m: Tuple[float, float, float],
                 propellant_mass_kg: float = 100.0,
                 tank_radius_m: float = 0.4,
                 slosh_fraction: float = 0.25,
                 damping_ratio: float = 0.02,
                 feed_axis_body: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        self.tank_location = np.asarray(tank_location_body_m, dtype=float)
        self.propellant_mass_kg = float(propellant_mass_kg)
        self.tank_radius_m = float(tank_radius_m)
        self.slosh_fraction = float(slosh_fraction)
        self.damping_ratio = float(damping_ratio)
        self.feed_axis = np.asarray(feed_axis_body, dtype=float)
        self.feed_axis /= np.linalg.norm(self.feed_axis)

        # 2D lateral state in plane orthogonal to feed axis
        self.displacement_2d = np.zeros(2)
        self.velocity_2d = np.zeros(2)

        # Construct orthonormal basis [e1, e2, feed_axis]
        ref = np.array([1.0, 0.0, 0.0]) if abs(self.feed_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(self.feed_axis, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.feed_axis, e1)
        self.plane_basis = np.column_stack([e1, e2])  # 3x2 matrix

    @property
    def slosh_mass_kg(self) -> float:
        return self.propellant_mass_kg * self.slosh_fraction

    def natural_frequency_rad_s(self, axial_accel_m_s2: float) -> float:
        """
        Slosh natural frequency under axial settling acceleration:
        omega_n = sqrt( a_axial / L_pendulum ), where L_pend ~ 0.5 * R_tank.
        In freefall (a_axial -> 0), surface-tension / capillary baseline frequency applies.
        """
        a_settling = max(0.001, float(axial_accel_m_s2))
        pendulum_arm = 0.5 * self.tank_radius_m
        return float(np.sqrt(a_settling / pendulum_arm))

    def step(self, dt_s: float, bus_accel_body_m_s2: np.ndarray) -> SloshState:
        """
        Integrate slosh oscillator state under bus acceleration:
        q_ddot + 2 * zeta * omega_n * q_dot + omega_n^2 * q = -a_lateral.
        Computes reactive force and torque imparted to the bus.
        """
        a_bus = _vector(bus_accel_body_m_s2, "bus_accel_body_m_s2")
        # Axial acceleration along tank feed axis
        a_axial = float(np.dot(a_bus, self.feed_axis))
        # Lateral acceleration projected onto slosh plane
        a_lat_3d = a_bus - a_axial * self.feed_axis
        a_lat_2d = self.plane_basis.T @ a_lat_3d

        omega_n = self.natural_frequency_rad_s(a_axial)
        m_s = self.slosh_mass_kg
        k_s = m_s * (omega_n ** 2)
        c_s = 2.0 * self.damping_ratio * m_s * omega_n

        # Slosh relative equation of motion:
        # q_ddot = -a_lat_2d - (c_s / m_s) * q_dot - (k_s / m_s) * q
        q_ddot = -a_lat_2d - (2.0 * self.damping_ratio * omega_n) * self.velocity_2d - (omega_n ** 2) * self.displacement_2d

        # Forward Euler integration
        self.velocity_2d += q_ddot * float(dt_s)
        self.displacement_2d += self.velocity_2d * float(dt_s)

        # Bounded within tank radius
        r_disp = float(np.linalg.norm(self.displacement_2d))
        max_r = 0.8 * self.tank_radius_m
        if r_disp > max_r:
            self.displacement_2d = (self.displacement_2d / r_disp) * max_r
            self.velocity_2d *= 0.5  # Inelastic wall boundary collision

        # Reaction force exerted on spacecraft bus:
        # F_slosh = k_s * q + c_s * q_dot (in 2D plane)
        f_slosh_2d = k_s * self.displacement_2d + c_s * self.velocity_2d
        f_slosh_3d = self.plane_basis @ f_slosh_2d

        # Torque about spacecraft origin: tau = r_tank x F_slosh
        tau_slosh_3d = np.cross(self.tank_location, f_slosh_3d)

        return SloshState(
            displacement_m=self.displacement_2d.copy(),
            velocity_m_s=self.velocity_2d.copy(),
            reaction_force_body_n=f_slosh_3d,
            reaction_torque_body_nm=tau_slosh_3d,
        )

    def check_ullage_settling(self, net_accel_body_m_s2: np.ndarray,
                              threshold_m_s2: float = 0.02) -> Tuple[bool, bool, float]:
        """
        Check if axial acceleration along feed axis is sufficient to settle liquid over the drain.
        Returns: (is_settled, gas_bubble_ingestion_hazard, axial_accel_m_s2).
        """
        a_net = _vector(net_accel_body_m_s2, "net_accel_body_m_s2")
        a_axial = float(np.dot(a_net, self.feed_axis))
        is_settled = a_axial >= float(threshold_m_s2)
        bubble_hazard = not is_settled
        return is_settled, bubble_hazard, a_axial

    def plan_ullage_settling_burn(self, total_spacecraft_mass_kg: float,
                                  rcs_settling_thrust_n: float = 4.0,
                                  required_delta_v_m_s: float = 0.1) -> Dict[str, float]:
        """
        Compute duration and propellant required for an RCS ullage settling burn
        before main engine valve ignition.
        """
        if total_spacecraft_mass_kg <= 0 or rcs_settling_thrust_n <= 0:
            raise ValueError("Mass and thrust must be positive")
        accel = rcs_settling_thrust_n / total_spacecraft_mass_kg
        duration_s = required_delta_v_m_s / accel
        isp_s = 220.0
        g0 = 9.80665
        mass_flow = rcs_settling_thrust_n / (isp_s * g0)
        prop_used_kg = mass_flow * duration_s

        return {
            "duration_s": float(duration_s),
            "settling_acceleration_m_s2": float(accel),
            "propellant_used_kg": float(prop_used_kg),
            "delta_v_achieved_m_s": float(required_delta_v_m_s),
        }


# ====================================================================================================
# 4. GRAVITY GRADIENT DISTURBANCE TORQUE
# ====================================================================================================

class GravityGradientTorque:
    """
    Computes continuum gravity gradient disturbance torque:
    tau_gg = (3 * mu / r^3) * [ r_hat x (I_bus * r_hat) ].
    Applies to arbitrary non-spherical spacecraft inertia in planetary/lunar/asteroid orbits.
    """
    @staticmethod
    def compute(r_body_to_center_m: np.ndarray, inertia_tensor_kg_m2: np.ndarray,
                gm_central_body: float) -> np.ndarray:
        """
        r_body_to_center_m: Vector from spacecraft center of mass toward central body, in body frame.
        inertia_tensor_kg_m2: 3x3 spacecraft inertia tensor in body frame.
        gm_central_body: Gravitational parameter of central body [m^3/s^2].
        """
        r_vec = _vector(r_body_to_center_m, "r_body_to_center_m")
        r_dist = float(np.linalg.norm(r_vec))
        if r_dist == 0:
            raise ValueError("Distance to central body cannot be zero")
        r_hat = r_vec / r_dist

        inertia = np.asarray(inertia_tensor_kg_m2, dtype=float)
        if inertia.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3x3")

        # tau_gg = 3 * mu / r^3 * (r_hat x (I * r_hat))
        i_r = inertia @ r_hat
        torque_gg = (3.0 * float(gm_central_body) / (r_dist ** 3)) * np.cross(r_hat, i_r)
        return torque_gg


# ====================================================================================================
# 5. STAR TRACKER BLINDING & RELATIVISTIC STELLAR ABERRATION
# ====================================================================================================

@dataclass
class StarTrackerStatus:
    """Diagnostic outcome of star tracker optical field of view check."""
    is_blinded: bool
    blinded_by: Optional[str]            # 'sun', 'moon', 'earth', 'plume', or None
    sun_angle_deg: float
    moon_angle_deg: float
    earth_angle_deg: float


class StarTrackerModel:
    """
    Models star tracker optical exclusion zones and relativistic stellar aberration.
    Detects when Sun, Moon, or bright planetary limb blinds the sensor.
    """
    def __init__(self, boresight_body: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 sun_exclusion_deg: float = 35.0,
                 moon_exclusion_deg: float = 15.0,
                 earth_exclusion_deg: float = 20.0,
                 fov_deg: float = 20.0):
        b = np.asarray(boresight_body, dtype=float)
        if b.shape != (3,) or not np.isclose(np.linalg.norm(b), 1.0, atol=1e-8):
            raise ValueError("Boresight must be a unit 3-vector")
        self.boresight_body = b
        self.sun_exclusion_deg = float(sun_exclusion_deg)
        self.moon_exclusion_deg = float(moon_exclusion_deg)
        self.earth_exclusion_deg = float(earth_exclusion_deg)
        self.fov_deg = float(fov_deg)

    def check_blinding(self, r_sc_icrf: np.ndarray, r_sun_icrf: np.ndarray,
                       dcm_body_to_icrf: np.ndarray,
                       r_moon_icrf: Optional[np.ndarray] = None,
                       r_earth_icrf: Optional[np.ndarray] = None,
                       thruster_plume_active: bool = False) -> StarTrackerStatus:
        """
        Check if star tracker optical boresight enters exclusion cones of bright bodies
        or is obscured by glowing RCS plume ejecta.
        """
        r_sc = _vector(r_sc_icrf, "r_sc_icrf")
        r_sun = _vector(r_sun_icrf, "r_sun_icrf")
        dcm = np.asarray(dcm_body_to_icrf, dtype=float)

        # Boresight in ICRF: b_icrf = DCM * b_body
        b_icrf = dcm @ self.boresight_body

        # Sun angle
        to_sun = r_sun - r_sc
        dir_sun = to_sun / np.linalg.norm(to_sun)
        cos_sun = np.clip(np.dot(b_icrf, dir_sun), -1.0, 1.0)
        sun_ang = float(np.degrees(np.arccos(cos_sun)))

        # Moon angle
        moon_ang = 180.0
        if r_moon_icrf is not None:
            to_moon = np.asarray(r_moon_icrf) - r_sc
            dir_moon = to_moon / np.linalg.norm(to_moon)
            cos_moon = np.clip(np.dot(b_icrf, dir_moon), -1.0, 1.0)
            moon_ang = float(np.degrees(np.arccos(cos_moon)))

        # Earth angle
        earth_ang = 180.0
        if r_earth_icrf is not None:
            to_earth = np.asarray(r_earth_icrf) - r_sc
            dir_earth = to_earth / np.linalg.norm(to_earth)
            cos_earth = np.clip(np.dot(b_icrf, dir_earth), -1.0, 1.0)
            earth_ang = float(np.degrees(np.arccos(cos_earth)))

        blinded_by = None
        if thruster_plume_active:
            blinded_by = "plume"
        elif sun_ang < self.sun_exclusion_deg:
            blinded_by = "sun"
        elif moon_ang < self.moon_exclusion_deg:
            blinded_by = "moon"
        elif earth_ang < self.earth_exclusion_deg:
            blinded_by = "earth"

        return StarTrackerStatus(
            is_blinded=blinded_by is not None,
            blinded_by=blinded_by,
            sun_angle_deg=sun_ang,
            moon_angle_deg=moon_ang,
            earth_angle_deg=earth_ang,
        )

    @staticmethod
    def relativistic_stellar_aberration(catalog_star_unit_icrf: np.ndarray,
                                        sc_velocity_icrf_m_s: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute relativistic stellar aberration:
        u_apparent = [ u_0 + beta + (gamma - 1)/(beta^2) * (beta . u_0) * beta - gamma * beta ] / [ 1 + beta . u_0 ]
        To leading order: delta_u = beta - (beta . u_0) * u_0.
        Returns (apparent_unit_vector, deflection_arcseconds).
        """
        u0 = _vector(catalog_star_unit_icrf, "catalog_star_unit_icrf")
        u0 /= np.linalg.norm(u0)
        v = _vector(sc_velocity_icrf_m_s, "sc_velocity_icrf_m_s")

        beta = v / c
        beta_mag = np.linalg.norm(beta)
        if beta_mag >= 1.0:
            raise ValueError("Spacecraft velocity cannot exceed or equal c")

        # Leading-order deflection vector
        delta_u = beta - np.dot(beta, u0) * u0
        u_apparent = u0 + delta_u
        u_apparent /= np.linalg.norm(u_apparent)

        cos_deflection = np.clip(np.dot(u0, u_apparent), -1.0, 1.0)
        deflection_rad = float(np.arccos(cos_deflection))
        deflection_arcsec = float(np.degrees(deflection_rad) * 3600.0)

        return u_apparent, deflection_arcsec


# ====================================================================================================
# 6. ECLIPSE THERMAL SNAP (BOOM CONTRACTION MOMENTUM IMPULSE)
# ====================================================================================================

class EclipseThermalSnapModel:
    """
    Models thermal snap shock impulse induced when crossing the penumbra/umbra boundary.
    Based on structural thermo-elastic beam bending (Thornton & Foster formulation):
    Asymmetric cooling across the boom cross-section (diameter 2*R_b) creates a through-depth
    thermal gradient Delta T_grad = eta * Delta T, inducing a thermal bending moment M_T and
    rapid tip transverse velocity v_trans = (alpha * L^2 * Delta T_grad) / (6 * R_b * Delta t).
    The transverse velocity of the tip mass m_p creates the angular momentum impulse:
    Delta H_snap = r_boom x (m_p * v_trans).
    """
    def __init__(self, boom_length_m: float = 5.0,
                 boom_radius_m: float = 0.05,
                 panel_mass_kg: float = 25.0,
                 thermal_expansion_coeff: float = 23.0e-6,  # Aluminum / Carbon composite CTE [1/K]
                 delta_temperature_k: float = 200.0,        # Temperature drop across eclipse [K]
                 transverse_gradient_ratio: float = 0.15,   # Cross-sectional Delta T ratio (15%)
                 snap_time_scale_s: float = 0.5):
        self.boom_length_m = float(boom_length_m)
        self.boom_radius_m = float(boom_radius_m)
        self.panel_mass_kg = float(panel_mass_kg)
        self.alpha_cte = float(thermal_expansion_coeff)
        self.delta_t_k = float(delta_temperature_k)
        self.grad_ratio = float(transverse_gradient_ratio)
        self.snap_time_scale_s = float(snap_time_scale_s)

    def compute_shock_impulse(self, prev_sunlight_fraction: float,
                              curr_sunlight_fraction: float,
                              dt_s: float,
                              boom_axis_body: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                              sun_incidence_body: Optional[Tuple[float, float, float]] = None) -> Optional[np.ndarray]:
        """
        Check if a rapid eclipse transition occurred (|dPhi/dt| > threshold).
        If so, returns angular momentum impulse Delta H_snap in body frame [N*m*s].
        """
        if dt_s <= 0:
            return None
        dphi_dt = (curr_sunlight_fraction - prev_sunlight_fraction) / float(dt_s)
        # Transitioning into shadow (dphi_dt < -0.1) or emerging into sunlight (dphi_dt > 0.1)
        if abs(dphi_dt) < 0.1:
            return None  # Steady lighting, zero thermal shock

        # Through-depth cross-sectional gradient: Delta T_grad = eta * Delta T
        delta_t_grad = self.grad_ratio * self.delta_t_k

        # Thermo-elastic tip transverse bending displacement:
        # delta_tip = (alpha * L^2 * Delta T_grad) / (6 * R_b)
        delta_tip = (self.alpha_cte * (self.boom_length_m ** 2) * delta_t_grad) / (6.0 * self.boom_radius_m)
        v_trans_mag = delta_tip / self.snap_time_scale_s

        boom_dir = np.asarray(boom_axis_body, dtype=float)
        boom_dir /= np.linalg.norm(boom_dir)
        r_boom = boom_dir * self.boom_length_m

        # Transverse bending direction: orthogonal to boom axis
        if sun_incidence_body is not None:
            s_dir = np.asarray(sun_incidence_body, dtype=float)
            s_dir /= np.linalg.norm(s_dir)
            trans_dir = s_dir - np.dot(s_dir, boom_dir) * boom_dir
            if np.linalg.norm(trans_dir) > 1e-4:
                trans_dir /= np.linalg.norm(trans_dir)
            else:
                trans_dir = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([0.0, 0.0, 1.0]) if abs(boom_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            trans_dir = np.cross(boom_dir, ref)
            trans_dir /= np.linalg.norm(trans_dir)

        # Transverse tip velocity
        v_trans = v_trans_mag * trans_dir

        # Angular momentum impulse: Delta H = r_boom x (m_panel * v_trans)
        delta_h_snap = np.cross(r_boom, self.panel_mass_kg * v_trans)
        # Sign flips depending on cooling (contraction/bending inward) vs heating (expansion)
        sign = -1.0 if dphi_dt < 0 else 1.0
        return delta_h_snap * sign
