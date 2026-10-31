"""
=====================================================================================
Tests for ARGONAUT-Pathfinder: Asteroid Belt Precursor & TRL Retirement Engine
=====================================================================================
Verifies:
  1. Astronomical parameters and astrodynamics of Dwarf Planet Ceres (2.77 AU).
  2. Round-trip light time causality delay (29.4 to 62.7 min) validating autonomy.
  3. Ceres proximity operations delta-v budget (Touch-And-Go + Escape < 500 m/s).
  4. Technology Readiness Level (TRL 4/5 -> TRL 9) retirement ledger.
  5. Discovery-class cost and accelerated schedule comparison (Sample in GEO by 2034).
  6. Executive mission synthesis report formatting.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.asteroid_pathfinder import (
    AsteroidPathfinderConstants,
    AsteroidBeltPathfinder,
    PathfinderMissionReport,
    plan_argonaut_pathfinder_mission,
)


def test_asteroid_pathfinder_constants_and_astrodynamics():
    """Verify physical and astrodynamic parameters of Ceres precursor."""
    c = AsteroidPathfinderConstants

    assert abs(c.SEMI_MAJOR_AXIS_AU - 2.769) < 0.01
    assert abs(c.SURFACE_GRAVITY_MPS2 - 0.284) < 0.01
    assert abs(c.V_ESC_CERES_MPS - 516.4) < 1.0  # Escape speed is only 516 m/s!

    # Light time delay ensures autonomy is genuinely tested
    assert c.RTLT_MIN_MINUTES > 25.0
    assert c.RTLT_MAX_MINUTES > 60.0
    assert abs(c.RTLT_NOMINAL_MINUTES - 46.0) < 1.0

    # Hohmann transit time ~1.28 years (469 days)
    assert abs(c.TOF_HOHMANN_YEARS - 1.28) < 0.05


def test_ceres_pathfinder_mission_evaluation():
    """Verify Ceres end-to-end mission profile, delta-v ledger, and Earth capture."""
    mission = AsteroidBeltPathfinder.evaluate_ceres_mission(
        canister_gross_mass_kg=12.0,
        sample_payload_kg=0.75,
    )

    assert "Ceres" in mission["target"]
    assert "Occator" in mission["sampling_site"]
    assert mission["sample_mass_harvested_kg"] == 0.75
    assert mission["canister_gross_mass_kg"] == 12.0

    # Proximity delta-v is modest (under 500 m/s total)
    assert abs(mission["total_proximity_dv_mps"] - 485.0) < 10.0

    # Transit times
    assert abs(mission["outbound_transit_tof_years"] - 1.28) < 0.05
    assert mission["return_transit_tof_years"] == 1.45

    # Earth arrival and energy reduction
    assert mission["earth_arrival_v_inf_mps"] == 2200.0
    assert mission["kinetic_energy_reduction_percent"] > 20.0
    assert mission["peak_heat_flux_reduction_percent"] > 30.0
    assert mission["aerogeo_velocity_shed_mps"] > 900.0
    assert abs(mission["earth_docking_dv_mps"] - 3.569) < 0.05
    assert "FEASIBLE" in mission["status"]


def test_trl_retirement_and_cost_comparison():
    """Verify TRL advancement ladder and Discovery-class budget savings."""
    mission = AsteroidBeltPathfinder.evaluate_ceres_mission()

    ladder = mission["trl_retirement_ladder"]
    assert len(ladder) == 4

    subsystems = [item["subsystem"] for item in ladder]
    assert any("FPGA/QRC XNAV" in s for s in subsystems)
    assert any("Dual-Mode Micro-Propulsion" in s for s in subsystems)
    assert any("Cryogenic Volatile Encapsulation" in s for s in subsystems)
    assert any("Earth Aero-GEO" in s for s in subsystems)

    # All subsystems advance to TRL 9 after flight
    for item in ladder:
        assert item["pre_mission_trl"] in [4, 5]
        assert item["post_pathfinder_trl"] == 9

    # Cost comparison
    cs = mission["cost_and_schedule"]
    assert cs["pathfinder_estimated_cost_usd"] < 500e6  # Under $500M (Discovery class)
    assert cs["cost_reduction_factor"] > 8.0            # Over 8x cheaper than Flagship
    assert cs["pathfinder_total_duration_years"] < 8.0  # Returns sample in 2034!


def test_pathfinder_mission_report_summary():
    """Verify executive summary formatting and programmatic narrative."""
    report = plan_argonaut_pathfinder_mission()
    summary = report.summary()

    assert "ARGONAUT-PATHFINDER: ASTEROID BELT PRECURSOR" in summary
    assert "Dwarf Planet 1 Ceres" in summary
    assert "Occator Crater" in summary
    assert "WHY CERES IS THE ESSENTIAL TECHNOLOGY STEPPING STONE" in summary
    assert "ACCELERATED FLIGHT CHRONOLOGY" in summary
    assert "November 2034" in summary
    assert "TECHNOLOGY READINESS LEVEL (TRL) RETIREMENT LADDER" in summary
    assert "COST & RISK COMPARISON" in summary
    assert "8.7x cheaper" in summary
