"""
=====================================================================================
Tests for Jovian Sample Return & Earth Orbital Station Capture Engine
=====================================================================================
Verifies:
  1. Heliocentric transfer geometry from Jupiter to Earth (2.73 years, v_inf values).
  2. Stationary collector surface launch options and electrospray propulsion mass budgets.
  3. Interplanetary ballistic dispersion vs autonomous closed-loop XNAV midcourse guidance.
  4. Space station capture mechanisms (LEO Aerocapture vs Lunar Gateway NRHO vs Direct Burn).
  5. Mission report synthesis and executive summary formatting.
=====================================================================================
"""

import pytest
import numpy as np

from qrc_pulsar_bench.sample_return import (
    SampleReturnConstants,
    StationaryCollectorLauncher,
    DualModePropulsionSystem,
    AutonomousInterplanetaryGNC,
    EarthOrbitalStationCapture,
    SampleReturnMissionReport,
    plan_jovian_sample_return_mission,
)


def test_sample_return_constants_and_transfer_geometry():
    """Verify physical parameters and Jupiter-to-Earth heliocentric transfer ellipse."""
    # Transfer semi-major axis: (5.2044 + 1.0) / 2 ~ 3.1022 AU
    a_au = SampleReturnConstants.A_TRANS_M / SampleReturnConstants.AU_M
    assert abs(a_au - 3.1022) < 0.01

    # Time of flight: ~2.73 years (998 days)
    assert 2.65 < SampleReturnConstants.TOF_YEARS < 2.80
    assert 970.0 < SampleReturnConstants.TOF_DAYS < 1020.0

    # Hyperbolic excess velocities
    assert 5500.0 < SampleReturnConstants.V_INF_JUPITER_OUT_MPS < 5800.0  # ~5.64 km/s
    assert 8600.0 < SampleReturnConstants.V_INF_EARTH_IN_MPS < 9000.0    # ~8.79 km/s

    # Corridor tolerances
    assert SampleReturnConstants.AERO_PERIAPSIS_TARGET_ALT_M == 75e3
    assert SampleReturnConstants.AERO_CORRIDOR_TOLERANCE_M == 5e3


def test_stationary_collector_launch_and_propulsion_trades():
    """Verify surface vacuum launcher and high-Isp electrospray mass allocations."""
    # Surface launch from Callisto
    callisto_launch = StationaryCollectorLauncher.evaluate_surface_launch(
        moon_name="Callisto",
        payload_mass_kg=10.0,
        sample_mass_kg=1.5,
        launcher_type="electromagnetic_rail",
    )
    assert abs(callisto_launch["surface_escape_speed_mps"] - 2440.0) < 50.0  # ~2.44 km/s
    assert callisto_launch["launch_propellant_kg"] == 0.0  # Vacuum railgun consumes 0 onboard fuel
    assert callisto_launch["nominal_radiation_rad_hr"] < 10.0  # Safe haven (<10 rad/hr)

    # Chemical launch comparison
    chem_launch = StationaryCollectorLauncher.evaluate_surface_launch(
        moon_name="Callisto",
        payload_mass_kg=10.0,
        sample_mass_kg=1.5,
        launcher_type="chemical_rocket",
    )
    assert chem_launch["launch_propellant_kg"] > 5.0  # Consumes >5 kg of 10 kg mass

    # Electrospray interplanetary injection (4.7 km/s delta-v)
    electro = StationaryCollectorLauncher.evaluate_interplanetary_injection(
        canister_mass_kg=10.0,
        propulsion_type="electrospray_ion",
    )
    assert electro["propellant_mass_fraction"] < 0.20  # Under 20% propellant mass
    assert abs(electro["propellant_mass_kg"] - 1.48) < 0.1
    assert electro["dry_mass_kg"] > 8.0  # Ample margin for 1.5 kg sample + avionics

    # Bipropellant comparison
    biprop = StationaryCollectorLauncher.evaluate_interplanetary_injection(
        canister_mass_kg=10.0,
        propulsion_type="bipropellant",
    )
    assert biprop["propellant_mass_fraction"] > 0.75  # Over 75% mass is fuel!


def test_interplanetary_dispersion_and_autonomous_gnc():
    """Verify trajectory drift without guidance vs autonomous closed-loop precision."""
    # Without autonomous guidance: 0.5 m/s cutoff error causes >40,000 km miss
    drift_no_gnc = AutonomousInterplanetaryGNC.simulate_dispersion_and_corrections(
        initial_velocity_cutoff_error_mps=0.5,
        enable_autonomous_tcm=False,
    )
    assert drift_no_gnc["autonomous_guidance_enabled"] is False
    assert drift_no_gnc["uncorrected_miss_km"] > 40000.0  # >40,000 km miss
    assert drift_no_gnc["aerocapture_corridor_hit"] is False
    assert "MISSION LOST" in drift_no_gnc["status"]

    # With autonomous onboard GNC: 3 TCMs achieve <3 km targeting precision
    guided = AutonomousInterplanetaryGNC.simulate_dispersion_and_corrections(
        initial_velocity_cutoff_error_mps=0.5,
        enable_autonomous_tcm=True,
    )
    assert guided["autonomous_guidance_enabled"] is True
    assert guided["total_tcm_dv_mps"] < 10.0  # Consumes <10 m/s propellant
    assert guided["terminal_miss_km"] < 5.0    # Within +/- 5 km corridor!
    assert guided["aerocapture_corridor_hit"] is True
    assert guided["fpga_eval_latency_us"] < 250.0  # FPGA cycle latency <250 us
    assert "SUCCESS" in guided["status"]


def test_earth_orbital_station_capture_modes():
    """Compare LEO Aerocapture, Lunar Gateway NRHO capture, and direct burn."""
    capture_modes = EarthOrbitalStationCapture.evaluate_all_capture_modes()

    mode_a = capture_modes["mode_a_leo_aerocapture"]
    mode_b = capture_modes["mode_b_lunar_gateway_nrho"]
    mode_c = capture_modes["mode_c_direct_propulsive"]

    # Mode A (LEO Aerocapture at 400 km)
    assert abs(mode_a["atmospheric_entry_speed_mps"] - 14170.0) < 100.0  # ~14.17 km/s
    assert abs(mode_a["aerodynamic_delta_v_shed_mps"] - 6220.0) < 100.0  # ~6.22 km/s shed for free
    assert mode_a["onboard_burn_delta_v_mps"] < 120.0  # Only ~95 m/s circularization burn
    assert "HIGHLY OPTIMAL" in mode_a["feasibility"]

    # Mode B (Lunar Gateway NRHO multi-body capture)
    assert abs(mode_b["aerodynamic_delta_v_shed_mps"] - 1601.0) < 50.0   # ~1.6 km/s lunar assist
    assert mode_b["onboard_burn_delta_v_mps"] < 160.0  # ~142 m/s insertion
    assert mode_b["thermal_protection_required"] == "None (Pure vacuum multi-body capture, zero atmospheric heat)"
    assert "CRYOGENIC SAFE" in mode_b["feasibility"]

    # Mode C (Direct chemical burn)
    assert mode_c["onboard_burn_delta_v_mps"] > 6000.0  # Massive >6.3 km/s burn
    assert mode_c["propellant_mass_fraction_burn"] > 0.85  # Over 85% fuel required
    assert "INFEASIBLE" in mode_c["feasibility"]

    # Mode D (Geosynchronous Orbiting Station Capture & Orbiting Rover)
    mode_d = capture_modes["mode_d_geo_station"]
    assert abs(mode_d["geo_altitude_km"] - 35785.8) < 10.0  # GEO altitude ~35,786 km
    assert abs(mode_d["geo_radius_km"] - 42164.0) < 10.0    # GEO radius ~42,164 km
    assert abs(mode_d["aerodynamic_delta_v_shed_mps"] - 3806.5) < 50.0  # ~3.81 km/s shed aerodynamically
    assert abs(mode_d["onboard_burn_delta_v_mps"] - 1488.4) < 10.0      # ~1.49 km/s circularization at GEO
    assert "Orbiting Rover" in mode_d["orbiting_rover_intercept"]
    assert "QUARANTINE SAFE" in mode_d["planetary_protection"]
    assert "HIGHLY OPTIMAL FOR GEO" in mode_d["feasibility"]


def test_mission_synthesis_report():
    """Verify end-to-end mission synthesis report and executive formatting."""
    report = plan_jovian_sample_return_mission(
        collector_moon="Callisto",
        canister_gross_mass_kg=10.0,
        sample_payload_mass_kg=1.5,
    )

    assert report.collector_moon == "Callisto"
    assert report.canister_gross_mass_kg == 10.0
    assert report.sample_payload_mass_kg == 1.5
    assert report.propellant_mass_kg < 2.0
    assert report.dry_bus_mass_kg > 8.0
    assert report.terminal_targeting_accuracy_km < 5.0
    assert "geo_altitude_km" in report.capture_mode_d_geo_station

    summary = report.summary()
    assert "JOVIAN SAMPLE RETURN & ORBITAL STATION CAPTURE REPORT" in summary
    assert "Electrospray Propellant Mass" in summary
    assert "LEO Space Station Aerocapture" in summary
    assert "Lunar Gateway NRHO Invariant Manifold Capture" in summary
    assert "Geosynchronous Orbiting Station" in summary
    assert "Orbiting Rover Tug Intercept" in summary


def test_dual_mode_propulsion_system():
    """Verify hybrid Hydrazine impulsive + micro-ion continuous propulsion ledger."""
    prop = DualModePropulsionSystem(
        gross_mass_kg=15.0,
        sample_payload_kg=1.0,
        avionics_bus_dry_kg=5.0,
        propulsion_hardware_dry_kg=3.0,
        hydrazine_propellant_kg=3.8,
        ion_propellant_kg=2.2,
    )

    assert prop.dry_mass_kg == 9.0
    assert prop.current_mass_kg == 15.0

    # Test single burns
    burn_res = prop.burn_chemical(100.0)
    assert burn_res["burn_type"] == "chemical_hydrazine"
    assert burn_res["delta_v_mps"] == 100.0
    assert burn_res["fuel_consumed_kg"] > 0.0

    ion_res = prop.burn_ion(500.0)
    assert ion_res["burn_type"] == "micro_ion"
    assert ion_res["delta_v_mps"] == 500.0
    assert ion_res["fuel_consumed_kg"] > 0.0

    # Exhaustion check
    with pytest.raises(ValueError, match="Hydrazine fuel exhausted"):
        prop.burn_chemical(5000.0)

    # Staged mission profile simulation
    fresh_prop = DualModePropulsionSystem()
    ledger = fresh_prop.simulate_staged_mission_profile()

    assert ledger["initial_gross_mass_kg"] == 15.0
    assert ledger["dry_mass_kg"] == 9.0
    assert ledger["phase1_departure_chem_dv_mps"] > 100.0  # ~123.6 m/s
    assert ledger["phase2_cruise_ion_dv_mps"] > 4800.0      # ~4,952.4 m/s
    assert ledger["phase3_tcm_chem_dv_mps"] > 70.0         # ~76.5 m/s
    assert ledger["phase4_aerogeo_attitude_chem_dv_mps"] > 15.0  # ~19.5 m/s
    assert ledger["phase5_terminal_geo_chem_dv_mps"] > 400.0     # ~454.8 m/s
    assert ledger["total_mission_dv_mps"] > 5600.0         # Combined >5.6 km/s capability
    assert ledger["final_mass_after_docking_kg"] >= 9.0    # Preserves sample payload

