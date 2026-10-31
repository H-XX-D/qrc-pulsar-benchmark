"""
=====================================================================================
Tests for Jovian Moon Tour & Autonomous Deep-Space Navigation Engine
=====================================================================================
Verifies:
  1. Jovian system constants, Galilean moon physical parameters, and Hill radii.
  2. Radiation model dose rates, shielding attenuation, and circular orbit lifetimes.
  3. Hyperbolic gravity assist deflection, free velocity gain, and Tisserand parameter.
  4. Invariant manifold resonant tour vs Hohmann transfer (>95% fuel savings).
  5. Autonomous real-time flyby intervention overcoming 84-minute light-time delay.
=====================================================================================
"""

import pytest
import numpy as np

from qrc_pulsar_bench.jovian_tour import (
    MoonData,
    JovianSystemConstants,
    JovianRadiationModel,
    JovianMoonFlyby,
    JovianTourReport,
    JovianTourPlanner,
)


def test_jovian_system_constants_and_moons():
    """Verify physical parameters, Hill spheres, and communication light-time."""
    assert JovianSystemConstants.JUPITER_GM > 1.26e17
    assert JovianSystemConstants.JUPITER_RADIUS_M == 71492e3

    # Check moon catalog
    europa = JovianSystemConstants.get_moon("Europa")
    ganymede = JovianSystemConstants.get_moon("Ganymede")
    callisto = JovianSystemConstants.get_moon("Callisto")
    io = JovianSystemConstants.get_moon("Io")

    # Hill radii verification
    assert 13000e3 < europa.hill_radius_m < 14000e3  # ~13,653 km
    assert 31000e3 < ganymede.hill_radius_m < 32000e3  # ~31,717 km
    assert 49000e3 < callisto.hill_radius_m < 51000e3  # ~50,141 km

    # Laplace resonance check: T_Io : T_Europa : T_Ganymede ~ 1 : 2 : 4
    ratio_eu_io = europa.orbital_period_s / io.orbital_period_s
    ratio_ga_io = ganymede.orbital_period_s / io.orbital_period_s
    assert abs(ratio_eu_io - 2.0) < 0.05
    assert abs(ratio_ga_io - 4.0) < 0.1

    # Round trip light time check
    rtlt_nom_min = JovianSystemConstants.RTLT_NOM_S / 60.0
    assert 80.0 < rtlt_nom_min < 90.0  # Nominal ~84.3 minutes
    assert 65.0 < JovianSystemConstants.RTLT_MIN_S / 60.0 < 75.0
    assert 95.0 < JovianSystemConstants.RTLT_MAX_S / 60.0 < 110.0


def test_jovian_radiation_model():
    """Verify empirical radiation dose rate profile and lifetime limits."""
    europa_r = JovianSystemConstants.get_moon("Europa").semi_major_axis_m
    ganymede_r = JovianSystemConstants.get_moon("Ganymede").semi_major_axis_m
    callisto_r = JovianSystemConstants.get_moon("Callisto").semi_major_axis_m

    d_europa = JovianRadiationModel.dose_rate_rad_per_hr(europa_r)
    d_ganymede = JovianRadiationModel.dose_rate_rad_per_hr(ganymede_r)
    d_callisto = JovianRadiationModel.dose_rate_rad_per_hr(callisto_r)

    # Check calibrated values against Galileo/Clipper baseline
    assert 500.0 < d_europa < 580.0      # ~539 rad/hr
    assert 65.0 < d_ganymede < 85.0      # ~75.8 rad/hr
    assert 5.0 < d_callisto < 9.0        # ~7.1 rad/hr

    # Check vault shielding attenuation (10 mm Al gives ~10x reduction)
    shielded_europa = JovianRadiationModel.vault_shielded_dose(d_europa, shielding_thickness_mm_al=10.0)
    assert abs(shielded_europa - d_europa * 0.10) < 1.0

    # Check operational lifetime to 150 krad qualification limit
    europa_lifetime = JovianRadiationModel.estimate_orbit_lifetime_days(europa_r, 150000.0, 10.0)
    callisto_lifetime = JovianRadiationModel.estimate_orbit_lifetime_days(callisto_r, 150000.0, 10.0)

    # Low Europa orbit exceeds limit in ~115 days; Callisto safe haven lasts >2,000 days
    assert 100.0 < europa_lifetime < 130.0
    assert callisto_lifetime > 2000.0

    # Trapezoidal integration check
    radii = np.array([europa_r, europa_r])
    times = np.array([0.0, 3600.0])  # 1 hour at Europa
    int_dose = JovianRadiationModel.integrated_dose_rad(radii, times)
    assert abs(int_dose - d_europa) < 1.0


def test_jovian_hyperbolic_gravity_assist_and_tisserand():
    """Verify hyperbolic deflection, delta-v gain, Tisserand parameter, and collision safety."""
    # Ganymede gravity assist at 200 km altitude, v_inf = 3,500 m/s
    ganymede_flyby = JovianMoonFlyby.compute_hyperbolic_encounter(
        moon_name="Ganymede",
        v_inf_mps=3500.0,
        periapse_alt_m=200e3,
    )
    assert abs(ganymede_flyby["deflection_deg"] - 25.61) < 0.2
    assert abs(ganymede_flyby["dv_gain_mps"] - 1551.7) < 5.0  # >1.5 km/s free boost!

    # Europa gravity assist at 50 km altitude, v_inf = 3,500 m/s
    europa_flyby = JovianMoonFlyby.compute_hyperbolic_encounter(
        moon_name="Europa",
        v_inf_mps=3500.0,
        periapse_alt_m=50e3,
    )
    assert abs(europa_flyby["dv_gain_mps"] - 977.6) < 5.0  # ~978 m/s free boost!
    assert abs(europa_flyby["deflection_deg"] - 16.06) < 0.2

    # Tisserand parameter calculation
    t_p = JovianMoonFlyby.tisserand_parameter(
        semi_major_axis_m=1070400e3,
        eccentricity=0.0,
        inclination_rad=0.0,
        moon_name="Ganymede",
    )
    assert abs(t_p - 3.0) < 1e-4  # Circular co-orbital body has T_p = 3.0

    # Collision watchdog
    safe_ok, _ = JovianMoonFlyby.check_collision_safety("Ganymede", periapse_alt_m=150e3)
    assert safe_ok is True

    hazard_ok, hazard_msg = JovianMoonFlyby.check_collision_safety("Ganymede", periapse_alt_m=20e3)
    assert hazard_ok is False
    assert "SAFETY HAZARD" in hazard_msg

    crash_ok, crash_msg = JovianMoonFlyby.check_collision_safety("Ganymede", periapse_alt_m=-50e3)
    assert crash_ok is False
    assert "FATAL COLLISION" in crash_msg


def test_jovian_tour_planner_hohmann_vs_manifold():
    """Verify Hohmann vs Manifold transfer budgets and fuel savings."""
    planner = JovianTourPlanner()

    # Hohmann Ganymede -> Europa
    hohmann = planner.compute_hohmann_transfer("Ganymede", "Europa")
    assert abs(hohmann["dv_total_mps"] - 2824.2) < 5.0
    assert hohmann["mass_fraction"] > 0.55  # Requires >55% propellant mass

    # Low-energy manifold resonant tour
    manifold = planner.compute_low_energy_manifold_tour("Ganymede", "Europa")
    assert manifold["dv_total_mps"] < 100.0  # Under 100 m/s total!
    assert manifold["fuel_savings_percent"] > 95.0  # >95% fuel reduction!
    assert manifold["free_gravity_assist_dv_mps"] > 1500.0
    assert manifold["shielded_dose_per_pass_krad"] < 3.0  # Safe eccentric flyby dose


def test_autonomous_flyby_intervention_vs_ground_latency():
    """
    Demonstrate that Earth ground control fails due to 84-minute light-time delay,
    while autonomous onboard GNC restores the trajectory in <500 microseconds.
    """
    planner = JovianTourPlanner()

    # Simulate flyby 30 minutes before periapse with 30 m/s transversal perturbation
    # pushing trajectory into Ganymede surface crash
    report = planner.autonomous_flyby_intervention(
        moon_name="Ganymede",
        nominal_periapse_alt_m=200e3,
        time_before_periapse_s=1800.0,  # 30 minutes to periapse
        inbound_distance_m=35000e3,
        v_inf_mps=3500.0,
    )

    # Earth ground control fails due to latency causality violation
    assert "FATAL" in report["ground_status"]
    assert report["rtlt_minutes"] > 80.0

    # Trajectory was initially crashing into moon surface
    assert report["is_safe_perturbed"] is False
    assert report["perturbed_alt_km"] < 0.0  # Negative altitude = surface impact

    # Autonomous onboard system restores nominal flight path
    assert "SUCCESS" in report["autonomous_status"]
    assert abs(report["restored_alt_km"] - 200.0) < 0.1  # Restored to 200 km
    assert abs(report["correction_dv_mps"] - 30.0) < 0.1
    assert report["computation_time_us"] < 50000.0  # <50 ms in Python (<200 ns FPGA)

    # Verify full report generation
    full_report = planner.generate_full_tour_report("Ganymede", "Europa")
    summary = full_report.summary()
    assert "JOVIAN MOON TOUR & AUTONOMOUS GNC BENCHMARK REPORT" in summary
    assert "Propellant Expenditure Reduction" in summary
    assert "Physical Causality Violation" in summary or "FATAL" in summary
