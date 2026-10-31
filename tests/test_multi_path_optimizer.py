"""Tests for Multi-Path Trajectory Optimizer & Mission-Critical Safety Watchdog."""

import pytest
import numpy as np
from qrc_pulsar_bench.multi_path_optimizer import (
    FlightSafetyState,
    SafetyThresholds,
    MultiPathTrajectoryOptimizer,
    R_MOON,
)


def test_candidate_path_evaluation_and_safety_floor():
    """Verify symplectic candidate rollout and automatic detection of perilune floor violation."""
    optimizer = MultiPathTrajectoryOptimizer()

    # Initial state near apoapse (70,000 km)
    r0 = np.array([0.0, 0.0, 70000.0e3])
    v0 = np.array([200.0, 0.0, 0.0])

    # Safe candidate
    target_pos = np.array([0.0, 0.0, 70000.0e3])
    drift_m, min_r, is_safe = optimizer.evaluate_candidate_path(
        r0, v0, candidate_dv=np.zeros(3), target_pos_at_horizon=target_pos, horizon_s=3600.0, dt_sub=60.0
    )
    assert is_safe is True
    assert min_r > 60000.0e3

    # Unsafe candidate starting low (altitude 800 km) with steep downward velocity
    r_low = np.array([0.0, 0.0, R_MOON + 800.0e3])
    v_low = np.array([0.0, 0.0, -500.0]) # plunging downward at 500 m/s
    drift_m_unsafe, min_r_unsafe, is_safe_unsafe = optimizer.evaluate_candidate_path(
        r_low, v_low, candidate_dv=np.zeros(3), target_pos_at_horizon=r_low, horizon_s=3600.0, dt_sub=60.0
    )
    assert is_safe_unsafe is False


def test_opportunistic_hold_fire_near_perilune():
    """Verify optimizer holds fire when spacecraft is inside the perilune high-gradient zone."""
    optimizer = MultiPathTrajectoryOptimizer()

    # Spacecraft near perilune: r = 5,000 km (< 40,000 km threshold)
    r_peri = np.array([0.0, 0.0, 5000.0e3])
    v_peri = np.array([2000.0, 0.0, 0.0])
    target_r = np.array([0.0, 0.0, 5050.0e3]) # 50 km tracking offset
    target_v = v_peri.copy()

    dv, state = optimizer.find_optimal_impulse(r_peri, v_peri, target_r, target_v)

    # Must hold fire: condition is not optimal for fuel usage
    assert np.allclose(dv, 0.0)
    assert state == FlightSafetyState.NOMINAL_OPTIMIZING


def test_opportunistic_burn_near_apoapse():
    """Verify optimizer computes an optimal micro-burn when near apoapse."""
    optimizer = MultiPathTrajectoryOptimizer()

    # Spacecraft near apoapse: r = 70,000 km
    r_apo = np.array([0.0, 0.0, 70000.0e3])
    v_apo = np.array([200.0, 0.0, 0.0])
    target_r = np.array([0.0, 0.0, 70040.0e3]) # 40 km tracking error (> 25 km deadband)
    target_v = v_apo.copy()

    dv, state = optimizer.find_optimal_impulse(r_apo, v_apo, target_r, target_v)

    # Should trigger opportunistic fire with gentle micro-burn
    assert state == FlightSafetyState.OPPORTUNISTIC_FIRE
    assert np.linalg.norm(dv) > 0.0
    # Must be a micro-burn (< 200 mm/s)
    assert np.linalg.norm(dv) < 0.20


def test_emergency_safe_hold_abort():
    """Verify watchdog triggers emergency safe-hold abort if altitude breaches safety floor."""
    optimizer = MultiPathTrajectoryOptimizer()

    # Spacecraft dangerously low: altitude = 300 km (< 500 km floor)
    r_danger = np.array([0.0, 0.0, R_MOON + 300.0e3])
    v_danger = np.array([1600.0, 0.0, 0.0])

    dv, state = optimizer.find_optimal_impulse(r_danger, v_danger, r_danger, v_danger)

    # Must trigger EMERGENCY_SAFE_HOLD with positive prograde periapse-raising impulse
    assert state == FlightSafetyState.EMERGENCY_SAFE_HOLD
    assert np.linalg.norm(dv) == pytest.approx(0.5, rel=1e-2)
    # Prograde direction (aligned with velocity)
    assert np.dot(dv, v_danger) > 0.0
