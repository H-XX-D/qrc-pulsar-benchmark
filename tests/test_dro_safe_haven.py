"""Unit tests for Distant Retrograde Orbit (DRO) Safe Haven and Ground Reinsertion Architecture."""

import pytest
import numpy as np

from qrc_pulsar_bench.dro_safe_haven import (
    DROSafeHavenArchitect,
    OrbitStabilityComparison,
    DROReinsertionPlan,
)


def test_dro_vs_halo_stability():
    """Verifies that DROs have unitary Floquet multipliers (|lambda|=1) and zero station-keeping."""
    architect = DROSafeHavenArchitect()

    # Callisto L2 Halo is dynamically unstable
    callisto_halo = architect.evaluate_stability("Callisto_L2_Halo")
    assert callisto_halo.max_floquet_multiplier > 100.0
    assert callisto_halo.lyapunov_doubling_time_days < 5.0
    assert callisto_halo.annual_station_keeping_dv_mps > 20.0
    assert callisto_halo.max_safe_mode_wait_time_days < 20.0

    # Callisto DRO is passively stable forever
    callisto_dro = architect.evaluate_stability("Callisto_DRO")
    assert callisto_dro.max_floquet_multiplier == 1.000
    assert np.isinf(callisto_dro.lyapunov_doubling_time_days)
    assert callisto_dro.annual_station_keeping_dv_mps == 0.0
    assert callisto_dro.max_safe_mode_wait_time_days >= 3650.0  # >10 years
    assert callisto_dro.passive_stability_duration_years >= 100.0


def test_ground_reinsertion_planning():
    """Verifies that ground control has an unconstrained window to plan low-delta-V reinsertion."""
    architect = DROSafeHavenArchitect()
    reinsertion_callisto = architect.calculate_reinsertion_trajectory(
        haven_name="Callisto",
        target_orbit_type="Europa_Plume_Skimmer",
    )

    # Ground has full year window to plan with zero fuel penalty
    assert reinsertion_callisto.ground_planning_window_available_days >= 365.0
    # Total reinsertion delta-V is modest (<100 m/s)
    assert reinsertion_callisto.total_reinsertion_dv_mps < 100.0
    assert reinsertion_callisto.escape_burn_dv_mps < 50.0
    assert reinsertion_callisto.insertion_burn_dv_mps < 50.0

    # Titan system reinsertion
    reinsertion_titan = architect.calculate_reinsertion_trajectory(
        haven_name="Titan",
        target_orbit_type="Enceladus_Plume_Skimmer",
    )
    assert reinsertion_titan.total_reinsertion_dv_mps < 100.0
    assert reinsertion_titan.transfer_time_days > 10.0


def test_full_stability_ledger():
    """Verifies stability audit across all planetary bodies (Callisto, Titan, Moon)."""
    architect = DROSafeHavenArchitect()
    ledger = architect.generate_full_stability_ledger()
    assert len(ledger) == 6

    # Every DRO in the catalog must have zero station-keeping fuel requirement
    for item in ledger:
        if "DRO" in item.orbit_type:
            assert item.annual_station_keeping_dv_mps == 0.0
            assert item.max_floquet_multiplier == 1.000
            assert item.passive_stability_duration_years >= 100.0
        else:
            # Non-DRO orbits have positive station-keeping requirements
            assert item.annual_station_keeping_dv_mps > 0.0
            assert item.max_floquet_multiplier > 1.0
