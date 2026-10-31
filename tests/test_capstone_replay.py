"""Tests for Multi-Dataset Flight Telemetry Replay & Causal Integrity Engine."""

import pytest
import numpy as np
from qrc_pulsar_bench.capstone_replay import (
    MultiDatasetFlightReplay,
    FlightReplayReport,
    CapstoneFlightReplay,
    CapstoneReplayReport,
)


def test_multi_dataset_loading_and_disturbance_extraction():
    """Verify loading across multiple distinct real NASA datasets and 3-body tidal extraction."""
    # 1. Baseline CAPSTONE Nov 2022 (NRHO)
    replay_nov = MultiDatasetFlightReplay("capstone_nov2022")
    assert len(replay_nov.times_jd) == 361
    assert replay_nov.total_duration_days == pytest.approx(30.0, rel=1e-3)
    assert replay_nov.mean_disturbance > 0.0
    # True physical SRP disturbance at apoapse is in the micro-m/s^2 regime
    assert 1e-6 <= replay_nov.median_disturbance <= 1e-4

    # 2. Out-of-Sample CAPSTONE Feb 2023 (NRHO Window 2)
    replay_feb = MultiDatasetFlightReplay("capstone_feb2023")
    assert len(replay_feb.times_jd) == 337
    assert replay_feb.total_duration_days == pytest.approx(28.0, rel=1e-3)
    assert np.min(replay_feb.pos_km) < 5000.0  # Perilune dip verified (~3,302 km)
    assert 1e-6 <= replay_feb.median_disturbance <= 1e-4

    # 3. Artemis 1 Orion Nov-Dec 2022 (Distant Retrograde Orbit)
    replay_artemis = MultiDatasetFlightReplay("artemis1_nov2022")
    assert len(replay_artemis.times_jd) == 121
    assert replay_artemis.total_duration_days == pytest.approx(10.0, rel=1e-3)
    # DRO radii range between 38,000 km and 95,000 km
    dro_radii = np.linalg.norm(replay_artemis.pos_km, axis=1)
    assert np.min(dro_radii) > 35000.0
    assert np.max(dro_radii) < 100000.0
    assert 1e-6 <= replay_artemis.median_disturbance <= 1e-4

    # 4. LRO Low Lunar Orbit (Extreme Mascon Gravity Field)
    replay_lro = MultiDatasetFlightReplay("lro_jan2023")
    assert len(replay_lro.times_jd) == 2161
    assert replay_lro.total_duration_days == pytest.approx(3.0, rel=1e-3)
    # Altitude is strictly in low lunar orbit: 80 - 105 km
    lro_altitudes = np.linalg.norm(replay_lro.pos_km, axis=1) - 1737.4
    assert 70.0 <= np.min(lro_altitudes) <= 90.0
    assert 95.0 <= np.max(lro_altitudes) <= 120.0
    # In low orbit, lunar mascons dominate: median disturbance in mm/s^2 regime
    assert 1e-3 <= replay_lro.median_disturbance <= 1e-2


def test_causal_replay_capstone_baseline():
    """Verify causal replay on baseline CAPSTONE Nov 2022 dataset."""
    replay = MultiDatasetFlightReplay("capstone_nov2022")

    report = replay.run_causal_replay(
        check_interval_h=6.0,
        deadband_km=25.0,
        duration_days=5.0
    )

    assert isinstance(report, FlightReplayReport)
    assert report.causal_integrity_verified is True
    assert report.burn_count > 0
    assert report.max_tracking_error_km < 40.0
    assert report.delta_v_per_month_mps < 10.0
    assert report.mean_burn_mps < 0.20


def test_causal_replay_out_of_sample_capstone():
    """Verify causal replay on completely disjoint out-of-sample Feb 2023 CAPSTONE segment."""
    replay_feb = MultiDatasetFlightReplay("capstone_feb2023")

    report = replay_feb.run_causal_replay(
        check_interval_h=6.0,
        deadband_km=25.0,
        duration_days=5.0
    )

    assert isinstance(report, FlightReplayReport)
    assert report.causal_integrity_verified is True
    assert report.burn_count > 0
    # Tracking error clamped without future leakage on out-of-sample data
    assert report.max_tracking_error_km < 60.0
    assert report.delta_v_per_month_mps < 15.0
    assert report.mean_burn_mps < 0.20


def test_causal_replay_artemis1_orion_dro():
    """Verify causal replay on NASA Artemis 1 Orion Distant Retrograde Orbit."""
    replay_artemis = MultiDatasetFlightReplay("artemis1_nov2022")

    report = replay_artemis.run_causal_replay(
        check_interval_h=6.0,
        deadband_km=50.0,
        duration_days=5.0
    )

    assert isinstance(report, FlightReplayReport)
    assert report.causal_integrity_verified is True
    assert report.dataset_name == "artemis1_nov2022"
    assert report.burn_count > 0
    # Maintains stable DRO despite real flight disturbance
    assert report.max_tracking_error_km < 400.0  # < 0.5% of 92,000 km orbit
    assert report.delta_v_per_month_mps < 30.0


def test_backward_compatibility_capstone_flight_replay():
    """Verify legacy CapstoneFlightReplay API functions identically."""
    legacy = CapstoneFlightReplay()
    report = legacy.run_replay(autonomous=True, duration_days=5.0)

    assert isinstance(report, CapstoneReplayReport)
    assert report.control_mode == "onboard_autonomous"
    assert report.ground_connection_required is False
    assert report.burn_count > 0
    assert report.max_tracking_error_km < 40.0
