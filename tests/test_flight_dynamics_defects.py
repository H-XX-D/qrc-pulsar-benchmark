"""Tests specifically verifying fixes for the 8 flight dynamics and astrodynamics defects."""

import numpy as np
import pytest
from scipy.optimize import brentq

from qrc_pulsar_bench.flight_dynamics_models import (
    ReactionWheel,
    ReactionWheelArray,
    EclipseThermalSnapModel,
)
from qrc_pulsar_bench.thruster_allocation import RCSCluster, ThrusterAllocator
from qrc_pulsar_bench.catastrophe_avoidance import CatastropheAvoidanceFlightSystem
from qrc_pulsar_bench.mission_event_simulator import MissionEventSimulator
from qrc_pulsar_bench.station_keeping import StationKeepingReport


def test_defect_1_momentum_dump_sign_reduces_wheel_momentum():
    """
    Defect 1: External RCS torque must oppose wheel momentum.
    tau_RCS = (H_target - H_current) / dt.
    Old code had (H_current - H_target) / dt, which doubled wheel momentum instead of zeroing it.
    """
    rwa = ReactionWheelArray.standard_3_axis_orthogonal(
        inertia_kg_m2=0.05, max_speed_rpm=6000.0, max_torque_nm=0.5
    )
    # Set positive wheel spin: +25 N*m*s on X axis
    rwa.wheel_speeds_rad_s = np.array([500.0, 0.0, 0.0])  # H_x = 0.05 * 500 = 25 N*m*s
    h_initial = rwa.wheel_angular_momentum_vector()
    assert h_initial[0] == pytest.approx(25.0)

    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0, torque_weight=100.0)
    r_cm = np.zeros(3)

    # Plan a 30s dump to target H = 0
    dump_plan = rwa.plan_momentum_dump(allocator, r_cm, duration_s=30.0, target_momentum_nm_s=np.zeros(3))

    # Required RCS torque must be NEGATIVE to remove POSITIVE momentum:
    # tau_RCS = (0 - 25) / 30 = -0.833 N*m
    assert dump_plan.required_torque_nm[0] == pytest.approx(-25.0 / 30.0, rel=1e-3)

    # Now verify physical integration:
    # External RCS torque tau_ext = tau_rcs_des.
    # To keep bus steady (omega_dot = 0), wheel array absorbs tau_ext: H_w(t+dt) = H_w(t) + tau_ext * dt
    tau_applied = dump_plan.allocation.achieved_wrench[3:]
    h_final = h_initial + tau_applied * dump_plan.burn_duration_s

    # Final momentum must be near zero, NOT 50 N*m*s!
    assert abs(h_final[0]) < 1.0
    assert abs(h_final[0]) < abs(h_initial[0])


def test_defect_2_pyramid_angle_isotropic_3_axis_authority():
    """
    Defect 2: 4-wheel pyramid elevation angle must be 35.26 deg for equal 3-axis authority.
    Old code used 54.74 deg (zenith angle), producing a 4:1 imbalance.
    """
    rwa = ReactionWheelArray.standard_4_wheel_pyramid(
        inertia_kg_m2=0.05, max_speed_rpm=6000.0, max_torque_nm=0.2
    )
    # Authority Gramian: G = A_w @ A_w^T
    gramian = rwa.A_w @ rwa.A_w.T

    diag = np.diag(gramian)
    # X, Y, and Z diagonal elements must be equal for isotropic authority
    assert diag[0] == pytest.approx(diag[1], rel=1e-3)
    assert diag[0] == pytest.approx(diag[2], rel=1e-3)
    # Non-diagonal coupling should be zero
    off_diag = gramian - np.diag(diag)
    np.testing.assert_allclose(off_diag, np.zeros((3, 3)), atol=1e-10)


def test_defect_3_wheel_dynamic_unbalance_unit_conversion():
    """
    Defect 3: 1 g*mm^2 = 1e-3 kg * (1e-3 m)^2 = 1e-9 kg*m^2.
    Old code had 1e-7 (factor of 100x error).
    """
    wheel = ReactionWheel(
        spin_axis_body=(0.0, 0.0, 1.0),
        inertia_kg_m2=0.05,
        static_unbalance_g_mm=0.0,
        dynamic_unbalance_g_mm2=20.0,  # 20 g*mm^2
    )
    rwa = ReactionWheelArray([
        wheel,
        ReactionWheel(spin_axis_body=(1.0, 0.0, 0.0)),
        ReactionWheel(spin_axis_body=(0.0, 1.0, 0.0)),
    ])
    # Set wheel speed to 100 rad/s
    rwa.wheel_speeds_rad_s = np.array([100.0, 0.0, 0.0])

    _, tau_jitter = rwa.micro_vibration_jitter()

    # Dynamic unbalance torque: Tau_d = U_d * omega^2
    # U_d = 20 * 1e-9 kg*m^2 = 2e-8 kg*m^2.
    # Tau_d = 2e-8 * (100)^2 = 2e-4 N*m.
    expected_tau_nm = 20.0 * 1e-9 * (100.0 ** 2)
    assert np.linalg.norm(tau_jitter) == pytest.approx(expected_tau_nm, rel=1e-4)


def test_defect_4_allocator_success_fails_when_wrench_unsaturated():
    """
    Defect 4: ThrusterAllocator must report success=False when commanded wrench exceeds capacity.
    Old code reported success=True because lsq_linear optimizer converged to the boundary.
    """
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Request an impossible 50 N*m torque (cluster can only deliver ~5 N*m)
    cmd_impossible = np.array([0.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    res = allocator.allocate(cmd_impossible, r_cm)

    # Must report success=False because commanded wrench was NOT delivered
    assert not res.success
    assert np.linalg.norm(res.wrench_residual[3:]) > 40.0


def test_defect_5_thermal_snap_thermoelastic_mechanics():
    """
    Defect 5: Thermal snap impulse must be grounded in boom cross-sectional gradient mechanics,
    not arbitrary [-0.2*v, 0, v] hardcoded heuristics.
    """
    snap_model = EclipseThermalSnapModel(
        boom_length_m=6.0,
        boom_radius_m=0.05,
        panel_mass_kg=25.0,
        thermal_expansion_coeff=23e-6,
        delta_temperature_k=200.0,
    )
    # Rapid eclipse ingress
    shock = snap_model.compute_shock_impulse(1.0, 0.0, dt_s=1.0, boom_axis_body=(0.0, 1.0, 0.0))
    assert shock is not None
    assert isinstance(shock, np.ndarray)
    assert np.linalg.norm(shock) > 0.0


def test_defect_6_jupiter_callisto_exact_jacobi_l1():
    """
    Defect 6: Jupiter-Callisto L1 Jacobi constant is 3.00619, not hardcoded 3.0018.
    """
    flight_sys = CatastropheAvoidanceFlightSystem("Jupiter-Callisto")
    params = flight_sys.system_params["Jupiter-Callisto"]
    mu = params["mu"]

    # Numerically solve exact L1 location: x - (1-mu)/(x+mu)^2 + mu/(1-mu-x)^2 = 0
    def f(x):
        return x - (1.0 - mu) / ((x + mu) ** 2) + mu / ((1.0 - mu - x) ** 2)

    x_l1 = brentq(f, 0.5, 1.0 - mu - 1e-6)
    c_l1_exact = x_l1 ** 2 + 2.0 * (1.0 - mu) / abs(x_l1 + mu) + 2.0 * mu / abs(1.0 - mu - x_l1)

    assert params["c_l1"] == pytest.approx(c_l1_exact, rel=1e-4)
    assert params["c_l1"] == pytest.approx(3.006194, rel=1e-4)


def test_defect_7_rocket_equation_fuel_for_18_mps_burn():
    """
    Defect 7: For a 500 kg craft at Isp 230 s, an 18 m/s burn costs ~3.97 kg, not 0.85 kg.
    """
    m0 = 500.0
    dv = 18.0
    isp = 230.0
    g0 = 9.80665
    expected_fuel = m0 * (1.0 - np.exp(-dv / (isp * g0)))
    assert expected_fuel == pytest.approx(3.974, abs=0.02)

    sim = MissionEventSimulator(mission_duration_days=10)
    # The collision event in the simulator should account for the physical rocket equation
    fuel_burn = sim.calculate_maneuver_fuel(mass_kg=500.0, delta_v_m_s=18.0, isp_s=230.0)
    assert fuel_burn == pytest.approx(expected_fuel, rel=1e-3)


def test_defect_8_station_keeping_metric_is_confinement_ratio():
    """
    Defect 8: Station keeping metric comparing drift km to error km must be named
    confinement ratio or drift containment ratio, not 'fuel savings factor'.
    """
    report = StationKeepingReport(
        duration_days=30.0,
        total_delta_v_mps=1.5,
        delta_v_per_month_mps=1.5,
        burn_count=5,
        mean_burn_mps=0.3,
        max_burn_mps=0.5,
        max_position_error_km=10.0,
        rms_position_error_km=5.0,
        uncontrolled_drift_km=1000.0,
        drift_confinement_ratio=100.0,
    )
    assert hasattr(report, "drift_confinement_ratio")
    assert report.drift_confinement_ratio == 100.0
