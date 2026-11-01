"""Tests for spacecraft 6-DOF thruster cluster, control allocation, and attitude slew dynamics."""

import numpy as np
import pytest

from qrc_pulsar_bench.thruster_allocation import (
    SpacecraftBus,
    Thruster,
    RCSCluster,
    ThrusterAllocator,
    StructuralFrame,
    StructuralStressMetrics,
    AttitudeSlewPlanner,
    quat_mult,
    quat_conjugate,
    quat_rotate,
    quat_from_two_vectors,
)


def test_quaternion_operations():
    # Test identity rotation
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat_rotate(q_id, v), v, atol=1e-12)

    # Test 90 degree rotation around Z axis
    angle = np.pi / 2.0
    q_z90 = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    v_x = np.array([1.0, 0.0, 0.0])
    v_rot = quat_rotate(q_z90, v_x)
    np.testing.assert_allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-12)

    # Test conjugate is inverse
    q_inv = quat_conjugate(q_z90)
    q_res = quat_mult(q_z90, q_inv)
    np.testing.assert_allclose(q_res, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    # Test quat_from_two_vectors
    q_align = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(quat_rotate(q_align, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_spacecraft_bus_com_and_inertia_shift():
    bus = SpacecraftBus(
        dry_mass_kg=400.0,
        propellant_mass_kg=100.0,
        com_dry_m=np.array([0.0, 0.0, 0.0]),
        com_prop_m=np.array([0.0, 0.0, -0.5]),
    )
    assert bus.total_mass() == 500.0
    # Center of mass: (400*0 + 100*(-0.5)) / 500 = -0.050 / 5 = -0.1 m along Z
    np.testing.assert_allclose(bus.center_of_mass(), [0.0, 0.0, -0.1], atol=1e-12)

    # When propellant burns down to 0
    bus.propellant_mass_kg = 0.0
    assert bus.total_mass() == 400.0
    np.testing.assert_allclose(bus.center_of_mass(), [0.0, 0.0, 0.0], atol=1e-12)

    # Inertia tensor changes with mass shift
    i_empty = bus.inertia_tensor()
    assert np.all(np.diag(i_empty) > 0.0)


def test_rcs_cluster_configuration_matrix():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.5, height_m=0.75, thrust_n=1.0)
    assert len(cluster.thrusters) == 12

    b_matrix = cluster.configuration_matrix(np.zeros(3))
    assert b_matrix.shape == (6, 12)

    # Full 6-DOF rank: can actuate all 3 forces and 3 torques
    rank = np.linalg.matrix_rank(b_matrix)
    assert rank == 6


def test_pure_translation_allocation():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Request pure +Z force of 2.0 N, zero torque
    cmd_wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    res = allocator.allocate(cmd_wrench, r_cm)

    assert res.success
    # Net force should match +Z
    np.testing.assert_allclose(res.achieved_wrench[:3], [0.0, 0.0, 2.0], atol=1e-2)
    # Net torque should be zero
    np.testing.assert_allclose(res.achieved_wrench[3:], [0.0, 0.0, 0.0], atol=1e-2)
    assert res.propellant_used_kg > 0.0
    assert res.active_thruster_count >= 1


def test_pure_torque_allocation():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Request pure Roll torque (tau_x = 0.5 N*m), zero net force
    cmd_wrench = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    res = allocator.allocate(cmd_wrench, r_cm)

    assert res.success
    # Achieved torque should match tau_x
    assert res.achieved_wrench[3] == pytest.approx(0.5, abs=0.05)
    # Net translational force should be near zero
    np.testing.assert_allclose(res.achieved_wrench[:3], [0.0, 0.0, 0.0], atol=0.05)


def test_center_of_mass_shift_parasitic_torque_compensation():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0, torque_weight=50.0)

    # Non-zero CoM (e.g. after asymmetrical propellant depletion)
    r_cm = np.array([0.05, -0.05, 0.1])

    # Request pure translational burn along +Z
    cmd_wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    res = allocator.allocate(cmd_wrench, r_cm)

    assert res.success
    # Force should be delivered along Z
    assert res.achieved_wrench[2] == pytest.approx(2.0, abs=0.1)
    # Allocator should have suppressed parasitic torque around the shifted CoM
    np.testing.assert_allclose(res.achieved_wrench[3:], [0.0, 0.0, 0.0], atol=0.05)


def test_minimum_impulse_bit_quantization():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    # Thruster has min_pulse_s = 0.01 s
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Extremely tiny wrench command below MIB threshold
    cmd_wrench = np.array([0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])
    res = allocator.allocate(cmd_wrench, r_cm)

    # Truncated to zero due to MIB
    assert res.active_thruster_count == 0
    assert res.propellant_used_kg == 0.0
    np.testing.assert_allclose(res.achieved_wrench, np.zeros(6), atol=1e-6)


def test_attitude_slew_planner():
    bus = SpacecraftBus()
    cluster = RCSCluster.standard_12_thruster_cluster()
    planner = AttitudeSlewPlanner(bus, cluster, max_angular_rate_deg_s=2.0, max_angular_accel_deg_s2=0.5)

    # Slew from identity to 90 degrees around Y axis
    q_start = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(90.0)
    q_target = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0])

    slew = planner.plan_slew(q_start, q_target)

    assert slew.slew_angle_rad == pytest.approx(angle, abs=1e-5)
    assert slew.duration_s > 0.0
    # 90 deg at 2 deg/s max with accel takes > 45 seconds
    assert slew.duration_s > 45.0
    assert slew.propellant_used_kg > 0.0
    assert slew.accel_time_s > 0.0

    # Test pointing slew to arbitrary inertial direction
    desired_dir = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    slew_pt = planner.plan_slew_to_direction(q_start, desired_dir)
    assert slew_pt.duration_s > 0.0
    assert slew_pt.slew_angle_rad > 0.0


def test_thruster_fault_reconfiguration_stuck_closed():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Normal allocation for +X force
    cmd = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_normal = allocator.allocate(cmd, r_cm)
    assert res_normal.success

    # Inject failure: Thruster 0 (P1_POS_X) fails stuck-closed
    cluster.set_fault(0, "stuck_closed")
    assert not cluster.thrusters[0].healthy

    # Allocator must reconfigure to use other thrusters (e.g. Pod 4 POS_X)
    res_fault = allocator.allocate(cmd, r_cm)
    assert res_fault.success
    # Thruster 0 should have zero pulse
    assert res_fault.pulse_durations_s[0] == 0.0
    # Net force should still be delivered
    assert res_fault.achieved_wrench[0] == pytest.approx(1.0, abs=0.1)


def test_thruster_fault_compensation_stuck_open():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0, torque_weight=100.0)
    r_cm = np.zeros(3)

    # Thruster 0 stuck open: constantly firing at 2.0 N!
    cluster.set_fault(0, "stuck_open")
    assert cluster.thrusters[0].stuck_open

    # Request zero net force and zero net torque (holding station/attitude)
    cmd = np.zeros(6)
    res = allocator.allocate(cmd, r_cm)

    # Thruster 0 must be reported firing
    assert res.pulse_durations_s[0] == 1.0
    # Opposing thrusters must fire to null the stuck-open torque
    assert res.active_thruster_count > 1
    # Residual torque should be tightly bounded despite the stuck thruster
    np.testing.assert_allclose(res.achieved_wrench[3:], [0.0, 0.0, 0.0], atol=0.2)


def test_opposing_thruster_structural_stress_and_vibration():
    frame = StructuralFrame(
        bus_length_m=1.5,
        strut_area_m2=0.002,
        youngs_modulus_pa=70.0e9,
        fundamental_freq_hz=2.5,
        damping_ratio=0.005,
    )
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0, torque_weight=100.0, frame=frame)
    r_cm = np.zeros(3)

    # Force opposing firings: Thruster 0 is stuck open firing +X at 2 N
    cluster.set_fault(0, "stuck_open")
    cmd = np.zeros(6)
    res = allocator.allocate(cmd, r_cm)

    assert res.success
    metrics = res.structural_metrics
    assert isinstance(metrics, StructuralStressMetrics)

    # Opposing force must capture antagonistic compressive load across bus
    assert metrics.opposing_force_n >= 1.8
    # Stress sigma = F / A (e.g. 2 N / 0.002 m^2 = 1000 Pa = 1e-3 MPa)
    assert metrics.frame_stress_mpa > 0.0
    assert metrics.strain_energy_j > 0.0
    assert metrics.vibrational_energy_j > 0.0
    assert metrics.thermal_dissipation_rate_w > 0.0

    # Modal decay time: 1 / (zeta * 2 * pi * f_1) = 1 / (0.005 * 2 * pi * 2.5) ~ 12.73 s
    expected_decay_s = 1.0 / (0.005 * 2.0 * np.pi * 2.5)
    assert metrics.vibration_decay_time_s == pytest.approx(expected_decay_s, rel=1e-3)


def test_nominal_burn_zero_opposing_stress():
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0)
    r_cm = np.zeros(3)

    # Clean +Z translation burn
    cmd = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    res = allocator.allocate(cmd, r_cm)

    assert res.success
    metrics = res.structural_metrics
    # Nominal unidirectional burn should induce zero opposing thruster stress
    assert metrics.opposing_force_n == pytest.approx(0.0, abs=1e-6)
    assert metrics.frame_stress_mpa == pytest.approx(0.0, abs=1e-6)
    assert metrics.strain_energy_j == pytest.approx(0.0, abs=1e-6)
    assert metrics.vibrational_energy_j == pytest.approx(0.0, abs=1e-6)
    assert metrics.thermal_dissipation_rate_w == pytest.approx(0.0, abs=1e-6)

