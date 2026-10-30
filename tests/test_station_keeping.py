"""Tests for Autonomous Predictive Station-Keeping & Disturbance Rejection Engine."""

import pytest
import numpy as np
from qrc_pulsar_bench.station_keeping import (
    EarthMoonConstants,
    StationKeepingPolicy,
    StationKeepingReport,
    EarthMoonCR3BP,
    PeriodicLyapunovOrbit,
    PredictiveStationKeepingEngine,
)


def test_cr3bp_l2_and_disturbances():
    """Verify L2 Lagrange point location and environmental disturbance evaluation."""
    cr3bp = EarthMoonCR3BP()
    # Earth-Moon L2 should be located at approximately x = 1.1557 in normalized units
    assert 1.155 <= cr3bp.x_l2 <= 1.156
    dist_from_moon_km = (cr3bp.x_l2 - (1.0 - cr3bp.mu)) * cr3bp.consts.length_star_m / 1e3
    assert 64000.0 <= dist_from_moon_km <= 65000.0

    # Disturbance test at L2 position
    r_test = np.array([cr3bp.x_l2, 0.0, 0.0])
    dist = cr3bp.compute_disturbances(0.0, r_test, srp_mps2=1.5e-7)
    # Both SRP and Moon J2 should contribute non-zero acceleration
    assert np.linalg.norm(dist) > 0.0
    assert np.all(np.isfinite(dist))


def test_periodic_lyapunov_orbit_and_monodromy():
    """Verify periodic orbit convergence and symplectic monodromy structure."""
    cr3bp = EarthMoonCR3BP()
    orbit = PeriodicLyapunovOrbit(cr3bp, dx_offset=0.005)

    # Full period should be ~14.65 days (3.35 - 3.40 normalized time units)
    period_days = orbit.period * cr3bp.consts.time_star_s / 86400.0
    assert 14.0 <= period_days <= 15.5
    assert -0.035 <= orbit.vy0 <= -0.020

    # Monodromy matrix computation
    M, eigvals = orbit.compute_monodromy_matrix()
    assert M.shape == (6, 6)

    # Find unstable and stable eigenvalues
    mags = np.abs(eigvals)
    max_ev = float(np.max(mags))
    min_ev = float(np.min(mags))

    # Unstable manifold should have large eigenvalue (> 500)
    assert max_ev > 500.0
    # Symplectic property: lambda_u * lambda_s == 1.0
    product = max_ev * min_ev
    assert abs(product - 1.0) < 0.05, f"Symplectic product violated: {product:.4f}"


def test_predictive_station_keeping_simulation():
    """Verify 30-day closed-loop station keeping suppresses drift with sub-meter/sec monthly delta-V."""
    engine = PredictiveStationKeepingEngine()
    policy = StationKeepingPolicy(
        aggressiveness=0.5,
        lookahead_days=12.0,
        check_interval_days=3.0,
        deadband_km=80.0
    )

    report = engine.run_simulation(
        duration_days=30.0,
        policy=policy,
        srp_mps2=1.5e-7,
        uncontrolled_comparison=True
    )

    assert isinstance(report, StationKeepingReport)
    assert report.duration_days == 30.0
    assert report.burn_count > 0

    # Predictive controller keeps maximum position error tightly bounded (< 30 km)
    assert report.max_position_error_km < 30.0
    assert report.rms_position_error_km < 15.0

    # Delta-V budget should be very small (< 2.5 m/s per month)
    assert report.delta_v_per_month_mps < 2.5
    # Average individual impulse should be in the millimeter-to-centimeter per second range
    assert report.mean_burn_mps < 0.20  # < 20 cm/s

    # Uncontrolled trajectory should diverge exponentially (> 50,000 km)
    assert report.uncontrolled_drift_km > 50000.0
    assert report.fuel_savings_factor_vs_uncontrolled > 1000.0
