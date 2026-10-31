"""Tests for Autonomous Inter-Body Transfer & In-Flight Mission Re-Targeting Engine."""

import pytest
import numpy as np
from qrc_pulsar_bench.interbody_transfer import (
    MultiBodySystemConstants,
    PeriodicOrbit,
    TransferPlanReport,
    InterBodyDynamics,
    AutonomousInterBodyPlanner,
)


def test_multibody_constants_and_lagrange_points():
    """Verify multi-body parameters, collinear equilibrium points, and Jacobi integral barriers."""
    # 1. Earth-Moon System
    em_consts = MultiBodySystemConstants.earth_moon()
    assert em_consts.mu == pytest.approx(0.0121505856, rel=1e-5)
    dynamics_em = InterBodyDynamics(em_consts)

    l_pts = dynamics_em.lagrange_points
    assert 0.80 < l_pts["L1"] < 0.85     # L1 between Earth and Moon
    assert 1.14 < l_pts["L2"] < 1.18     # L2 far side of Moon
    assert -1.02 < l_pts["L3"] < -0.99   # L3 opposite Earth

    c_l1 = dynamics_em.compute_jacobi_constant(np.array([l_pts["L1"], 0, 0, 0, 0, 0]))
    c_l2 = dynamics_em.compute_jacobi_constant(np.array([l_pts["L2"], 0, 0, 0, 0, 0]))
    c_l3 = dynamics_em.compute_jacobi_constant(np.array([l_pts["L3"], 0, 0, 0, 0, 0]))

    assert c_l1 > c_l2 > c_l3             # Energy hierarchy of collinear equilibrium points
    assert pytest.approx(3.188, abs=0.01) == c_l1
    assert pytest.approx(3.172, abs=0.01) == c_l2

    # Neck opening verification
    assert dynamics_em.is_neck_open(c_jacobi=3.10, neck="L1") is True   # Open
    assert dynamics_em.is_neck_open(c_jacobi=3.25, neck="L1") is False  # Closed by forbidden region

    # 2. Jupiter-Europa System
    je_consts = MultiBodySystemConstants.jupiter_europa()
    dynamics_je = InterBodyDynamics(je_consts)
    l_pts_je = dynamics_je.lagrange_points
    assert 0.95 < l_pts_je["L1"] < 0.99
    assert 1.01 < l_pts_je["L2"] < 1.05


def test_periodic_orbit_monodromy_and_manifolds():
    """Verify periodic Lyapunov orbits and invariant manifold tube generation."""
    planner = AutonomousInterBodyPlanner()

    # Compute L1 Lyapunov orbit
    orbit_l1 = planner.compute_periodic_orbit("L1", dx_offset=0.005)
    assert isinstance(orbit_l1, PeriodicOrbit)
    assert orbit_l1.libration_point == "L1"
    assert 10.0 <= orbit_l1.period_days <= 14.0
    assert orbit_l1.unstable_eigenvalue > 1.0
    assert orbit_l1.stable_eigenvalue < 1.0
    # Symplectic reciprocity: lambda_u * lambda_s ~ 1
    assert orbit_l1.unstable_eigenvalue * orbit_l1.stable_eigenvalue == pytest.approx(1.0, rel=0.05)

    # Compute L2 Lyapunov orbit
    orbit_l2 = planner.compute_periodic_orbit("L2", dx_offset=0.005)
    assert orbit_l2.libration_point == "L2"
    assert 13.0 <= orbit_l2.period_days <= 16.0
    assert orbit_l2.unstable_eigenvalue > 1.0

    # Generate manifold tubes
    t_u, traj_u = planner.generate_manifold_tube(orbit_l1, branch="unstable", t_max_days=10.0)
    assert len(t_u) > 10
    assert traj_u.shape[0] == 6

    t_s, traj_s = planner.generate_manifold_tube(orbit_l2, branch="stable", t_max_days=10.0)
    assert len(t_s) > 10
    assert traj_s.shape[0] == 6


def test_heteroclinic_manifold_transfer_earth_moon():
    """Verify low-energy heteroclinic transfer between Earth-Moon L1 and L2."""
    planner = AutonomousInterBodyPlanner()
    orbit_l1 = planner.compute_periodic_orbit("L1", dx_offset=0.005)
    orbit_l2 = planner.compute_periodic_orbit("L2", dx_offset=0.005)

    report = planner.plan_heteroclinic_transfer(
        origin_orbit=orbit_l1,
        destination_orbit=orbit_l2,
        max_tof_days=25.0
    )

    assert isinstance(report, TransferPlanReport)
    assert report.transfer_type == "heteroclinic_manifold_transfer"
    assert report.safety_certified is True
    # Low-energy multi-body transfer requires ~163 m/s (< 180 m/s)
    assert report.total_dv_mps < 180.0
    assert report.time_of_flight_days > 10.0
    # Enormous propellant savings vs 2-body Keplerian Hohmann baseline (~820 m/s)
    assert report.fuel_savings_percentage > 75.0
    # Planetary surface clearance verified
    assert report.min_secondary_alt_km > 500.0  # > 500 km above lunar surface
    assert report.onboard_replanning_time_us < 500.0  # Microsecond FPGA latency


def test_autonomous_in_flight_retargeting():
    """Verify autonomous in-flight re-targeting to a new body/orbit upon mission update."""
    planner = AutonomousInterBodyPlanner()
    orbit_l2 = planner.compute_periodic_orbit("L2", dx_offset=0.005)

    # Spacecraft in cislunar space (e.g. midway between Earth and Moon)
    current_state = np.array([0.92, 0.05, 0.0, -0.05, 0.15, 0.0])

    report = planner.retarget_mission_in_flight(
        current_state_norm=current_state,
        target_orbit=orbit_l2,
        dv_budget_mps=450.0,
        max_lookahead_days=20.0
    )

    assert isinstance(report, TransferPlanReport)
    assert report.transfer_type == "in_flight_retarget"
    assert report.destination_name == "Earth-Moon L2"
    assert report.total_dv_mps <= 450.0
    assert report.safety_certified is True
    assert report.min_secondary_alt_km >= 250.0  # Clear of lunar surface (> 250 km)
    assert report.fuel_savings_percentage > 35.0  # Fuel savings vs direct Hohmann
    assert report.onboard_replanning_time_us < 500.0


def test_collision_watchdog_rejection():
    """Verify that dangerous trajectories impacting planetary surface are rejected."""
    planner = AutonomousInterBodyPlanner()
    dynamics = planner.dynamics

    # Spacecraft trajectory that impacts the Moon (surface radius = 1,737 km)
    # Moon position is at x = 1 - mu = 0.98785
    impact_state = np.array([1.0 - dynamics.mu + 0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    is_safe, alt1_km, alt2_km = dynamics.check_collision(impact_state, floor_secondary_m=100e3)

    # Altitude is ~384 km minus 1,737 km = negative (below surface)
    assert is_safe is False
    assert alt2_km < 0.0
