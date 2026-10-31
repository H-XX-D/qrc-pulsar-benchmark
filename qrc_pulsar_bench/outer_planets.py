"""
=====================================================================================
QRC-PULSAR-BENCH: Unified Outer-Planet Sample Return & Fleet Exploration Architecture
=====================================================================================
Extends the autonomous staging, dynamic plume capture, cryogenic preparation,
and Earth Geosynchronous capture architecture to the entire outer Solar System:
  1. Jupiter System:
     - Staging Station: Callisto Halo (7.1 rad/hr haven, >24 yr life).
     - Target: Europa cryovolcanic ocean brine plumes (50 km alt, 2,100 m/s).
     - RTLT: 66 - 106 minutes (Nominal: 84.3 min).
  2. Saturn System:
     - Staging Station: Titan Orbit / Rhea Halo (Radiation-quiet: <2.5 rad/hr).
     - Target: Enceladus South Polar geysers (35 km alt, 1,250 m/s) & Titan organic haze.
     - RTLT: 132 - 176 minutes (Nominal: 159.0 min / 2.65 hours).
  3. Uranus System:
     - Staging Station: Titania / Oberon Halo (Zero radiation hazard).
     - Target: Miranda Verona Rupes vents (30 km alt, 1,850 m/s) & Ariel cryo-rift canyons.
     - RTLT: 295 - 352 minutes (Nominal: 319.4 min / 5.32 hours).
  4. Neptune System:
     - Staging Station: Triton Outer Orbital Depot / Proteus vicinity (Cryogenic haven).
     - Target: Triton retrograde nitrogen/methane geysers (25 km alt, 2,400 m/s).
     - RTLT: 478 - 532 minutes (Nominal: 500.2 min / 8.34 hours).

Common Architecture Principles:
  - Reusable Atmospheric & Ejection Skimmer Rover (No dangerous surface landing).
  - Outermost Sanctuary Moon Staging Station with Robotic Cleanroom Glovebox & Depot Refueling.
  - Periodic Earth Cargo Tanker Restock (propellant, fresh cassettes, return canisters).
  - Autonomous Synthesizable FPGA/QRC XNAV Guidance (ZERO DSN ground tracking).
  - Multi-Year Resonant Low-Thrust Heliocentric Cruise (halves Earth entry heat flux).
  - Geosynchronous Orbiting Station Capture & Orbiting Rover Intercept (Infinite Quarantine).
=====================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from .sample_return import (
    SampleReturnConstants,
    DualModePropulsionSystem,
    ResonantLowThrustTransfer,
    ReceiverSideDockingComputer,
    EarthOrbitalStationCapture,
)


@dataclass(frozen=True)
class OuterPlanetSystem:
    """Astronomical, physical, and mission parameters for an outer planetary system."""
    name: str
    semi_major_axis_au: float
    orbital_radius_m: float
    gm_planet: float
    rtlt_min_min: float
    rtlt_max_min: float
    rtlt_nominal_min: float
    staging_station_name: str
    staging_body_name: str
    staging_radiation_rad_hr: float
    staging_cleanroom_temp_k: float
    target_moon_name: str
    target_sampling_feature: str
    target_flyby_alt_km: float
    target_flyby_speed_mps: float
    plume_capture_matrix: str
    sortie_delta_v_mps: float
    fast_transit_tof_years: float
    resonant_transit_tof_years: float
    fast_arrival_v_inf_mps: float
    resonant_arrival_v_inf_mps: float

    @property
    def rtlt_nominal_hours(self) -> float:
        """Nominal round-trip light time in hours."""
        return self.rtlt_nominal_min / 60.0

    @property
    def station_lifetime_years(self) -> float:
        """Estimated electronics operational lifetime in station radiation environment."""
        shielded_dose = self.staging_radiation_rad_hr * 0.10  # 10 mm Al vault
        # Time to reach 150 krad qualification threshold
        hours_to_limit = 150000.0 / max(shielded_dose, 1e-4)
        return float(hours_to_limit / 8766.0)


# Curated catalog of outer planetary systems
OUTER_PLANET_SYSTEMS: Dict[str, OuterPlanetSystem] = {
    "jupiter": OuterPlanetSystem(
        name="Jupiter",
        semi_major_axis_au=5.2044,
        orbital_radius_m=5.2044 * SampleReturnConstants.AU_M,
        gm_planet=SampleReturnConstants.GM_JUPITER,
        rtlt_min_min=66.0,
        rtlt_max_min=106.0,
        rtlt_nominal_min=84.3,
        staging_station_name="Callisto Halo Station",
        staging_body_name="Callisto",
        staging_radiation_rad_hr=7.08,
        staging_cleanroom_temp_k=85.0,  # Cryogenic ocean brine & water ice
        target_moon_name="Europa",
        target_sampling_feature="Cryovolcanic Water Plumes & Sputter Halo",
        target_flyby_alt_km=50.0,
        target_flyby_speed_mps=2100.0,
        plume_capture_matrix="Graded Silica Aerogel (20 mg/cm3)",
        sortie_delta_v_mps=167.7,
        fast_transit_tof_years=2.73,
        resonant_transit_tof_years=4.50,
        fast_arrival_v_inf_mps=8790.0,
        resonant_arrival_v_inf_mps=3200.0,
    ),
    "saturn": OuterPlanetSystem(
        name="Saturn",
        semi_major_axis_au=9.5826,
        orbital_radius_m=9.5826 * SampleReturnConstants.AU_M,
        gm_planet=3.79312077e16,
        rtlt_min_min=132.0,
        rtlt_max_min=176.0,
        rtlt_nominal_min=159.0,
        staging_station_name="Titan-Rhea Logistics Station",
        staging_body_name="Titan / Rhea",
        staging_radiation_rad_hr=1.85,  # Outside inner belts, low radiation
        staging_cleanroom_temp_k=75.0,  # Liquid methane/ethane & water-ammonia ice
        target_moon_name="Enceladus",
        target_sampling_feature="South Polar Tiger Stripe Geysers (E-Ring Venting)",
        target_flyby_alt_km=35.0,
        target_flyby_speed_mps=1250.0,  # Very gentle plume transit
        plume_capture_matrix="Graded Aerogel & Cryogenic Impact Funnel",
        sortie_delta_v_mps=210.0,
        fast_transit_tof_years=3.95,
        resonant_transit_tof_years=6.80,
        fast_arrival_v_inf_mps=10450.0,
        resonant_arrival_v_inf_mps=3600.0,
    ),
    "uranus": OuterPlanetSystem(
        name="Uranus",
        semi_major_axis_au=19.2184,
        orbital_radius_m=19.2184 * SampleReturnConstants.AU_M,
        gm_planet=5.793939e15,
        rtlt_min_min=295.0,
        rtlt_max_min=352.0,
        rtlt_nominal_min=319.4,
        staging_station_name="Titania Sanctuary Depot",
        staging_body_name="Titania",
        staging_radiation_rad_hr=0.45,  # Negligible radiation environment
        staging_cleanroom_temp_k=55.0,  # Ammonia hydrates & solid CO2 clathrates
        target_moon_name="Miranda",
        target_sampling_feature="Verona Rupes Cryotectonic Fault Vents & Coronae",
        target_flyby_alt_km=30.0,
        target_flyby_speed_mps=1850.0,
        plume_capture_matrix="Porous Silica-Chalcogenide Aerogel Matrix",
        sortie_delta_v_mps=280.0,
        fast_transit_tof_years=8.50,
        resonant_transit_tof_years=14.20,
        fast_arrival_v_inf_mps=12100.0,
        resonant_arrival_v_inf_mps=4100.0,
    ),
    "neptune": OuterPlanetSystem(
        name="Neptune",
        semi_major_axis_au=30.0700,
        orbital_radius_m=30.0700 * SampleReturnConstants.AU_M,
        gm_planet=6.836529e15,
        rtlt_min_min=478.0,
        rtlt_max_min=532.0,
        rtlt_nominal_min=500.2,
        staging_station_name="Triton Outer Orbital Platform",
        staging_body_name="Triton Halo / Proteus",
        staging_radiation_rad_hr=0.20,  # Deep space background
        staging_cleanroom_temp_k=38.0,  # Solid cryogenic nitrogen & methane frost
        target_moon_name="Triton",
        target_sampling_feature="Active Nitrogen/Methane Cryogeysers & Tholin Plumes",
        target_flyby_alt_km=25.0,
        target_flyby_speed_mps=2400.0,
        plume_capture_matrix="Deep Cryo-Gold Aerogel & Electrostatic Dust Trap",
        sortie_delta_v_mps=315.0,
        fast_transit_tof_years=14.50,
        resonant_transit_tof_years=22.50,
        fast_arrival_v_inf_mps=13800.0,
        resonant_arrival_v_inf_mps=4500.0,
    ),
}


class OuterPlanetMissionArchitect:
    """
    Evaluates end-to-end sample return architectures across the outer solar system.
    """

    @classmethod
    def evaluate_system(
        cls,
        system_key: str,
        num_rover_sorties: int = 6,
        sample_mass_per_sortie_kg: float = 1.25,
        station_fuel_reserve_kg: float = 250.0,
    ) -> Dict[str, Any]:
        """
        Evaluate full mission architecture for a specific outer planetary system.
        """
        key = system_key.lower().strip()
        if key not in OUTER_PLANET_SYSTEMS:
            raise ValueError(f"Unsupported system '{key}'. Supported: {list(OUTER_PLANET_SYSTEMS.keys())}")

        sys = OUTER_PLANET_SYSTEMS[key]

        # Rover sortie fuel consumption per sortie
        rover_isp_s = 320.0
        g0 = 9.80665
        ve = rover_isp_s * g0
        rover_m0 = 50.0  # 35 kg dry + 15 kg fuel
        mf = rover_m0 * np.exp(-sys.sortie_delta_v_mps / ve)
        fuel_per_sortie = rover_m0 - mf
        total_fuel_consumed = fuel_per_sortie * num_rover_sorties

        remaining_station_depot_fuel = station_fuel_reserve_kg - total_fuel_consumed
        total_sample_harvested_kg = num_rover_sorties * sample_mass_per_sortie_kg
        canisters_prepared = min(num_rover_sorties, 10)

        # Resonant cruise physics to Earth GEO
        transfer = ResonantLowThrustTransfer(
            fast_tof_years=sys.fast_transit_tof_years,
            resonant_tof_years=sys.resonant_transit_tof_years,
            fast_v_inf_mps=sys.fast_arrival_v_inf_mps,
            resonant_v_inf_mps=sys.resonant_arrival_v_inf_mps,
        )
        savings = transfer.evaluate_energy_and_thermal_savings()

        # Terminal Earth GEO docking via Clohessy-Wiltshire solver
        docking = ReceiverSideDockingComputer.execute_handover_and_docking(
            handover_range_m=5000.0,
            initial_closing_speed_mps=2.0,
        )

        return {
            "system_name": sys.name,
            "semi_major_axis_au": sys.semi_major_axis_au,
            "nominal_rtlt_hours": sys.rtlt_nominal_hours,
            "nominal_rtlt_minutes": sys.rtlt_nominal_min,
            "causality_verdict": f"GROUND-IN-THE-LOOP IMPOSSIBLE: RTLT is {sys.rtlt_nominal_hours:.2f} hours; autonomous FPGA/QRC GNC strictly mandatory.",
            "staging_station": sys.staging_station_name,
            "staging_radiation_rad_hr": sys.staging_radiation_rad_hr,
            "station_operational_lifetime_years": sys.station_lifetime_years,
            "cleanroom_temperature_kelvin": sys.staging_cleanroom_temp_k,
            "target_moon": sys.target_moon_name,
            "sampling_feature": sys.target_sampling_feature,
            "flyby_altitude_km": sys.target_flyby_alt_km,
            "flyby_speed_mps": sys.target_flyby_speed_mps,
            "plume_capture_matrix": sys.plume_capture_matrix,
            "single_sortie_delta_v_mps": sys.sortie_delta_v_mps,
            "fuel_per_sortie_kg": float(fuel_per_sortie),
            "total_sorties_executed": num_rover_sorties,
            "total_sample_harvested_kg": float(total_sample_harvested_kg),
            "canisters_prepared_for_earth": canisters_prepared,
            "remaining_station_depot_fuel_kg": float(remaining_station_depot_fuel),
            "fast_transit_tof_years": sys.fast_transit_tof_years,
            "resonant_transit_tof_years": sys.resonant_transit_tof_years,
            "arrival_v_inf_fast_mps": sys.fast_arrival_v_inf_mps,
            "arrival_v_inf_resonant_mps": sys.resonant_arrival_v_inf_mps,
            "kinetic_energy_reduction_percent": savings["kinetic_energy_reduction_percent"],
            "peak_heat_flux_reduction_percent": savings["peak_heat_flux_reduction_percent"],
            "aerodynamic_strain_reduction_percent": savings["aerodynamic_strain_reduction_percent"],
            "earth_terminal_docking_dv_mps": docking["total_docking_dv_mps"],
            "earth_quarantine_orbit": "Geosynchronous Orbit (GEO, 35,786 km alt, >1M yr lifetime)",
        }


@dataclass
class OuterPlanetFleetReport:
    """Cross-planetary fleet synthesis comparing Jupiter, Saturn, Uranus, and Neptune architectures."""
    jupiter_mission: Dict[str, Any]
    saturn_mission: Dict[str, Any]
    uranus_mission: Dict[str, Any]
    neptune_mission: Dict[str, Any]

    def summary(self) -> str:
        """Format an executive technical comparison across all outer planetary systems."""
        j = self.jupiter_mission
        s = self.saturn_mission
        u = self.uranus_mission
        n = self.neptune_mission

        lines = [
            "==================================================================================================",
            "OUTER PLANET SAMPLE RETURN FLEET SYNTHESIS: JUPITER -> SATURN -> URANUS -> NEPTUNE",
            "==================================================================================================",
            "1. LIGHT-TIME CAUSALITY & AUTONOMOUS GNC MANDATE:",
            f"   - Jupiter (Callisto/Europa) : {j['nominal_rtlt_minutes']:5.1f} min RTLT ({j['nominal_rtlt_hours']:.2f} hrs) | {j['causality_verdict'][:45]}...",
            f"   - Saturn  (Titan/Enceladus) : {s['nominal_rtlt_minutes']:5.1f} min RTLT ({s['nominal_rtlt_hours']:.2f} hrs) | Ground control latency > 2.6 hours!",
            f"   - Uranus  (Titania/Miranda) : {u['nominal_rtlt_minutes']:5.1f} min RTLT ({u['nominal_rtlt_hours']:.2f} hrs) | Ground control latency > 5.3 hours!",
            f"   - Neptune (Triton Platform) : {n['nominal_rtlt_minutes']:5.1f} min RTLT ({n['nominal_rtlt_hours']:.2f} hrs) | Ground control latency > 8.3 hours!",
            "   -> Physical Causality: Zero ground-in-the-loop possible anywhere in outer solar system.",
            "",
            "2. STAGING DEPOTS & CRYOGENIC VOLATILE PRESERVATION:",
            f"   - Jupiter : {j['staging_station']:<28} | Temp: {j['cleanroom_temperature_kelvin']:4.1f} K | Life: {j['station_operational_lifetime_years']:5.1f} yrs | Haven: {j['staging_radiation_rad_hr']:.2f} rad/hr",
            f"   - Saturn  : {s['staging_station']:<28} | Temp: {s['cleanroom_temperature_kelvin']:4.1f} K | Life: {s['station_operational_lifetime_years']:5.1f} yrs | Haven: {s['staging_radiation_rad_hr']:.2f} rad/hr",
            f"   - Uranus  : {u['staging_station']:<28} | Temp: {u['cleanroom_temperature_kelvin']:4.1f} K | Life: {u['station_operational_lifetime_years']:5.1f} yrs | Haven: {u['staging_radiation_rad_hr']:.2f} rad/hr",
            f"   - Neptune : {n['staging_station']:<28} | Temp: {n['cleanroom_temperature_kelvin']:4.1f} K | Life: {n['station_operational_lifetime_years']:5.1f} yrs | Haven: {n['staging_radiation_rad_hr']:.2f} rad/hr",
            "   -> Triton Deep-Chill (38 K) prevents sublimation of solid nitrogen and methane clathrates.",
            "",
            "3. REUSABLE ATMOSPHERIC & EJECTION SKIMMER ROVER SORTIES:",
            f"   - Jupiter : Skims {j['target_moon']:<10} @ {j['flyby_altitude_km']:4.1f} km alt ({j['flyby_speed_mps']:5.0f} m/s) | dV: {j['single_sortie_delta_v_mps']:5.1f} m/s | Fuel: {j['fuel_per_sortie_kg']:.2f} kg",
            f"   - Saturn  : Skims {s['target_moon']:<10} @ {s['flyby_altitude_km']:4.1f} km alt ({s['flyby_speed_mps']:5.0f} m/s) | dV: {s['single_sortie_delta_v_mps']:5.1f} m/s | Fuel: {s['fuel_per_sortie_kg']:.2f} kg",
            f"   - Uranus  : Skims {u['target_moon']:<10} @ {u['flyby_altitude_km']:4.1f} km alt ({u['flyby_speed_mps']:5.0f} m/s) | dV: {u['single_sortie_delta_v_mps']:5.1f} m/s | Fuel: {u['fuel_per_sortie_kg']:.2f} kg",
            f"   - Neptune : Skims {n['target_moon']:<10} @ {n['flyby_altitude_km']:4.1f} km alt ({n['flyby_speed_mps']:5.0f} m/s) | dV: {n['single_sortie_delta_v_mps']:5.1f} m/s | Fuel: {n['fuel_per_sortie_kg']:.2f} kg",
            "   -> Zero surface landings; zero knife-edge penitentes or micro-gravity anchoring failures.",
            "",
            "4. MULTI-YEAR RESONANT HELIOCENTRIC CRUISE & EARTH ARRIVAL SAVINGS:",
            f"   - Jupiter -> Earth : {j['resonant_transit_tof_years']:5.1f} yrs | v_inf: {j['arrival_v_inf_fast_mps']/1e3:4.1f} -> {j['arrival_v_inf_resonant_mps']/1e3:4.1f} km/s | Energy: -{j['kinetic_energy_reduction_percent']:4.1f}% | Heat: -{j['peak_heat_flux_reduction_percent']:4.1f}%",
            f"   - Saturn  -> Earth : {s['resonant_transit_tof_years']:5.1f} yrs | v_inf: {s['arrival_v_inf_fast_mps']/1e3:4.1f} -> {s['arrival_v_inf_resonant_mps']/1e3:4.1f} km/s | Energy: -{s['kinetic_energy_reduction_percent']:4.1f}% | Heat: -{s['peak_heat_flux_reduction_percent']:4.1f}%",
            f"   - Uranus  -> Earth : {u['resonant_transit_tof_years']:5.1f} yrs | v_inf: {u['arrival_v_inf_fast_mps']/1e3:4.1f} -> {u['arrival_v_inf_resonant_mps']/1e3:4.1f} km/s | Energy: -{u['kinetic_energy_reduction_percent']:4.1f}% | Heat: -{u['peak_heat_flux_reduction_percent']:4.1f}%",
            f"   - Neptune -> Earth : {n['resonant_transit_tof_years']:5.1f} yrs | v_inf: {n['arrival_v_inf_fast_mps']/1e3:4.1f} -> {n['arrival_v_inf_resonant_mps']/1e3:4.1f} km/s | Energy: -{n['kinetic_energy_reduction_percent']:4.1f}% | Heat: -{n['peak_heat_flux_reduction_percent']:4.1f}%",
            "",
            "5. EARTH GEOSYNCHRONOUS ORBITAL STATION QUARANTINE:",
            f"   - Target Orbit                      : 35,786 km Geosynchronous Orbit (GEO)",
            f"   - Relative Docking Delta-V          : ~{j['earth_terminal_docking_dv_mps']:.3f} m/s (Two-impulse CW solution in <50 us)",
            f"   - Terminal Capture Contact Speed    : <= 0.050 m/s (Ultra-gentle robotic snare into Orbiting Rover tug)",
            f"   - Planetary Protection Status       : INFINITE QUARANTINE (>1,000,000 year non-decaying orbital lifetime)",
            f"   - Science Communication Downlink    : Permanent 24/7/365 stationary optical line-of-sight to Earth",
            "==================================================================================================",
        ]
        return "\n".join(lines)


def plan_outer_planet_fleet_mission(
    num_sorties_per_system: int = 6,
) -> OuterPlanetFleetReport:
    """
    Synthesize complete fleet sample return architectures across Jupiter, Saturn, Uranus, and Neptune.
    """
    j = OuterPlanetMissionArchitect.evaluate_system("jupiter", num_rover_sorties=num_sorties_per_system)
    s = OuterPlanetMissionArchitect.evaluate_system("saturn", num_rover_sorties=num_sorties_per_system)
    u = OuterPlanetMissionArchitect.evaluate_system("uranus", num_rover_sorties=num_sorties_per_system)
    n = OuterPlanetMissionArchitect.evaluate_system("neptune", num_rover_sorties=num_sorties_per_system)

    return OuterPlanetFleetReport(
        jupiter_mission=j,
        saturn_mission=s,
        uranus_mission=u,
        neptune_mission=n,
    )
