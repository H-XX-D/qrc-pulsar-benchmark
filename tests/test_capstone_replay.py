"""Tests for NASA CAPSTONE Flight Telemetry Replay & Autonomous GNC Benchmark."""

import pytest
import numpy as np
from qrc_pulsar_bench.capstone_replay import CapstoneFlightReplay, CapstoneReplayReport


def test_capstone_dataset_loading_and_disturbances():
    """Verify loading of real NASA CAPSTONE ephemeris and disturbance field extraction."""
    replay = CapstoneFlightReplay()

    assert len(replay.times_jd) == 361
    assert replay.total_duration_days == pytest.approx(30.0, rel=1e-3)
    assert replay.pos_km.shape == (361, 3)
    assert replay.vel_km_s.shape == (361, 3)

    # Orbital geometry checks for lunar NRHO
    radii = np.linalg.norm(replay.pos_km, axis=1)
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))

    assert 3000.0 <= min_r <= 4500.0   # Perilune radius: ~3,756 km
    assert 65000.0 <= max_r <= 75000.0 # Apoapse radius: ~71,745 km

    # Disturbance field
    dist_mags = np.linalg.norm(replay.a_dist_flight, axis=1)
    mean_dist = float(np.mean(dist_mags))
    assert 1e-4 <= mean_dist <= 1e-2


def test_capstone_autonomous_replay_short_horizon():
    """Verify autonomous predictive station-keeping against real CAPSTONE flight disturbances."""
    replay = CapstoneFlightReplay()

    # Autonomous test run: check every 6 hours with 36h lookahead over 5-day NRHO apolune arc
    report = replay.run_replay(
        autonomous=True,
        check_interval_h=6.0,
        lookahead_h=36.0,
        deadband_km=30.0,
        duration_days=5.0
    )

    assert isinstance(report, CapstoneReplayReport)
    assert report.control_mode == "onboard_autonomous"
    assert report.ground_connection_required is False
    assert report.burn_count > 0

    # Max tracking error should be bounded within ~55 km despite real-world perturbations
    assert report.max_tracking_error_km < 60.0
    # Monthly delta-V should be very small (< 15.0 m/s per month in realistic NRHO)
    assert report.delta_v_per_month_mps < 15.0
    # Mean burn impulse should be in the millimeter to centimeter range
    assert report.mean_burn_mps < 0.20 # < 20 cm/s
