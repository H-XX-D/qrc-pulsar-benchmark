"""
=====================================================================================
Tests for Unified Outer-Planet Sample Return & Fleet Exploration Architecture
=====================================================================================
Verifies:
  1. Astronomical parameters and physical constants across Jupiter, Saturn, Uranus, Neptune.
  2. Round-trip light time (RTLT) causality limits (84 min to 500 min).
  3. Saturn Enceladus plume capture, Titan/Rhea sanctuary staging, and 75 K cryo-storage.
  4. Uranus Miranda Verona Rupes capture and 55 K ammonia-hydrate preservation.
  5. Neptune Triton active nitrogen geyser capture and 38 K deep-cryo preservation.
  6. Cross-planetary fleet synthesis report formatting and GEO capture consistency.
=====================================================================================
"""

import pytest
import numpy as np

from qrc_pulsar_bench.outer_planets import (
    OuterPlanetSystem,
    OUTER_PLANET_SYSTEMS,
    OuterPlanetMissionArchitect,
    OuterPlanetFleetReport,
    plan_outer_planet_fleet_mission,
)


def test_outer_planet_systems_catalog_and_causality():
    """Verify astronomical parameters and causality limits across outer solar system."""
    assert len(OUTER_PLANET_SYSTEMS) == 4
    for key in ["jupiter", "saturn", "uranus", "neptune"]:
        assert key in OUTER_PLANET_SYSTEMS
        sys = OUTER_PLANET_SYSTEMS[key]
        assert sys.semi_major_axis_au > 5.0
        assert sys.rtlt_nominal_min > 60.0  # All outer planets exceed 1 hour RTLT
        assert sys.rtlt_nominal_hours > 1.0
        assert sys.station_lifetime_years > 20.0  # All staging stations in low-radiation havens

    # Specific light time assertions
    assert abs(OUTER_PLANET_SYSTEMS["jupiter"].rtlt_nominal_min - 84.3) < 1.0
    assert abs(OUTER_PLANET_SYSTEMS["saturn"].rtlt_nominal_min - 159.0) < 1.0
    assert abs(OUTER_PLANET_SYSTEMS["uranus"].rtlt_nominal_min - 319.4) < 1.0
    assert abs(OUTER_PLANET_SYSTEMS["neptune"].rtlt_nominal_min - 500.2) < 1.0

    # Neptune RTLT exceeds 8 hours
    assert OUTER_PLANET_SYSTEMS["neptune"].rtlt_nominal_hours > 8.0


def test_saturn_enceladus_mission_architecture():
    """Verify Saturn Enceladus plume capture and Titan/Rhea staging architecture."""
    saturn = OuterPlanetMissionArchitect.evaluate_system("saturn", num_rover_sorties=6)

    assert saturn["system_name"] == "Saturn"
    assert abs(saturn["semi_major_axis_au"] - 9.58) < 0.1
    assert saturn["nominal_rtlt_hours"] > 2.5
    assert "GROUND-IN-THE-LOOP IMPOSSIBLE" in saturn["causality_verdict"]

    # Staging depot & cryo storage
    assert "Titan-Rhea" in saturn["staging_station"]
    assert saturn["staging_radiation_rad_hr"] < 2.0  # Safe radiation haven
    assert saturn["cleanroom_temperature_kelvin"] == 75.0  # Methane/water ice preserve

    # Plume sampling
    assert saturn["target_moon"] == "Enceladus"
    assert saturn["flyby_altitude_km"] == 35.0
    assert saturn["flyby_speed_mps"] == 1250.0  # Gentle flyby
    assert saturn["total_sorties_executed"] == 6
    assert saturn["total_sample_harvested_kg"] == 7.5  # 6 * 1.25 kg
    assert saturn["remaining_station_depot_fuel_kg"] > 200.0

    # Heliocentric cruise to Earth GEO
    assert saturn["resonant_transit_tof_years"] == 6.80
    assert saturn["arrival_v_inf_resonant_mps"] == 3600.0
    assert saturn["kinetic_energy_reduction_percent"] > 30.0
    assert saturn["peak_heat_flux_reduction_percent"] > 45.0
    assert "Geosynchronous" in saturn["earth_quarantine_orbit"]


def test_uranus_miranda_mission_architecture():
    """Verify Uranus Miranda Verona Rupes plume/vent capture and Titania staging."""
    uranus = OuterPlanetMissionArchitect.evaluate_system("uranus", num_rover_sorties=6)

    assert uranus["system_name"] == "Uranus"
    assert abs(uranus["semi_major_axis_au"] - 19.22) < 0.1
    assert uranus["nominal_rtlt_hours"] > 5.0

    # Staging & cryo conditions
    assert "Titania" in uranus["staging_station"]
    assert uranus["staging_radiation_rad_hr"] < 1.0  # Near-zero radiation
    assert uranus["cleanroom_temperature_kelvin"] == 55.0  # Ammonia-hydrate preserve

    # Plume sampling
    assert uranus["target_moon"] == "Miranda"
    assert uranus["flyby_altitude_km"] == 30.0
    assert uranus["flyby_speed_mps"] == 1850.0
    assert uranus["single_sortie_delta_v_mps"] == 280.0
    assert uranus["total_sample_harvested_kg"] == 7.5

    # Cruise
    assert uranus["resonant_transit_tof_years"] == 14.20
    assert uranus["arrival_v_inf_resonant_mps"] == 4100.0
    assert uranus["kinetic_energy_reduction_percent"] > 30.0


def test_neptune_triton_mission_architecture():
    """Verify Neptune Triton active nitrogen geyser capture and 38 K deep-cryo preservation."""
    neptune = OuterPlanetMissionArchitect.evaluate_system("neptune", num_rover_sorties=6)

    assert neptune["system_name"] == "Neptune"
    assert abs(neptune["semi_major_axis_au"] - 30.07) < 0.1
    assert neptune["nominal_rtlt_hours"] > 8.0  # Over 8 hours RTLT!

    # Staging & ultra-cold nitrogen freezing point
    assert "Triton" in neptune["staging_station"]
    assert neptune["cleanroom_temperature_kelvin"] == 38.0  # Solid nitrogen frost!
    assert neptune["station_operational_lifetime_years"] > 100.0  # Zero radiation aging

    # Plume sampling
    assert neptune["target_moon"] == "Triton"
    assert neptune["flyby_altitude_km"] == 25.0
    assert neptune["flyby_speed_mps"] == 2400.0
    assert neptune["single_sortie_delta_v_mps"] == 315.0

    # Cruise
    assert neptune["resonant_transit_tof_years"] == 22.50
    assert neptune["arrival_v_inf_resonant_mps"] == 4500.0
    assert neptune["kinetic_energy_reduction_percent"] > 30.0


def test_outer_planet_fleet_report():
    """Verify cross-planetary fleet synthesis report and executive formatting."""
    fleet = plan_outer_planet_fleet_mission(num_sorties_per_system=6)

    assert fleet.jupiter_mission["system_name"] == "Jupiter"
    assert fleet.saturn_mission["system_name"] == "Saturn"
    assert fleet.uranus_mission["system_name"] == "Uranus"
    assert fleet.neptune_mission["system_name"] == "Neptune"

    summary = fleet.summary()
    assert "OUTER PLANET SAMPLE RETURN FLEET SYNTHESIS" in summary
    assert "LIGHT-TIME CAUSALITY & AUTONOMOUS GNC MANDATE" in summary
    assert "STAGING DEPOTS & CRYOGENIC VOLATILE PRESERVATION" in summary
    assert "REUSABLE ATMOSPHERIC & EJECTION SKIMMER ROVER SORTIES" in summary
    assert "MULTI-YEAR RESONANT HELIOCENTRIC CRUISE" in summary
    assert "EARTH GEOSYNCHRONOUS ORBITAL STATION QUARANTINE" in summary
    assert "Triton Deep-Chill (38 K)" in summary
    assert "Enceladus" in summary
    assert "Miranda" in summary
