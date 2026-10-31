"""
=====================================================================================
Planet-Specific Rover Restocking & Permanent Surface Relay Network Architecture
=====================================================================================
Models the operational cycle where each interplanetary restock mission:
  1. Delivers a planet-specific atmospheric/ejection capture rover tailored to the
     target celestial body's unique atmospheric, gravitational, and thermal environment.
  2. Upon mission completion, transitions the preceding rover into a permanent
     soft-landed surface observation post and telecommunications relay station.
  3. Links landed surface observatories to the persistent orbital tanker/depot nodes
     in halo orbits, building an expanding interplanetary science & relay mesh.

Planetary Targets & Rover Adaptations:
  - Ceres: Low-gravity micro-thruster hopper (0.284 m/s^2, 150 K).
  - Europa / Callisto: High-speed rad-hard aerogel plume skimmer (85 K) -> Callisto seismic post.
  - Titan / Enceladus: Titan dense-atmosphere rotorcraft/glider (1.5 bar, 94 K) + Enceladus plume probe.
  - Miranda / Titania: Chalcogenide deep-cryo vacuum skimmer (55 K) -> Titania surface station.
  - Triton: Ultra-deep-freeze cryo-gold nitrogen geyser probe (38 K) -> Triton cantaloupe terrain post.
=====================================================================================
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PlanetSpecificRover:
    """Design specifications for a planet-tailored atmospheric/ejection capture rover."""
    target_planet: str
    target_sampling_body: str
    rover_type: str
    operating_environment: str  # Atmospheric vs hard vacuum
    atmospheric_pressure_bar: float
    ambient_temperature_k: float
    surface_gravity_mps2: float
    primary_locomotion_mode: str
    sampling_mechanism: str
    dry_mass_kg: float
    propellant_mass_kg: float


@dataclass
class SurfaceObservationPost:
    """Specification of the permanent surface post left after sampling completion."""
    landing_body: str
    landing_site: str
    surface_landing_dv_mps: float
    braking_mechanism: str
    power_source: str
    operational_lifespan_years: float
    science_payload: List[str]
    relay_telecom_link: str


class SurfaceRelayNetworkArchitect:
    """
    Architects planet-specific rover delivery, end-of-mission landing profiles,
    and long-term surface observatory network operations.
    """

    @classmethod
    def get_planet_specific_rovers(cls) -> List[PlanetSpecificRover]:
        return [
            PlanetSpecificRover(
                target_planet="Asteroid Belt (Ceres)",
                target_sampling_body="Dwarf Planet 1 Ceres",
                rover_type="Micro-Gravity Cold-Gas Thruster Hopper",
                operating_environment="Hard vacuum (<1e-12 bar)",
                atmospheric_pressure_bar=0.0,
                ambient_temperature_k=150.0,
                surface_gravity_mps2=0.284,
                primary_locomotion_mode="Pulsed cold-gas hopping and micro-spikes",
                sampling_mechanism="Touch-and-Go rotary brush and electrostatic brine funnel",
                dry_mass_kg=45.0,
                propellant_mass_kg=15.0,
            ),
            PlanetSpecificRover(
                target_planet="Jupiter (Jovian System)",
                target_sampling_body="Europa Plumes & Callisto Haven",
                rover_type="High-Speed Rad-Hard Vacuum Plume Skimmer",
                operating_environment="Ultra-high radiation vacuum exosphere",
                atmospheric_pressure_bar=1e-11,
                ambient_temperature_k=85.0,
                surface_gravity_mps2=1.315,  # Europa
                primary_locomotion_mode="Hyperbolic resonance hopping via Jovian invariant manifolds",
                sampling_mechanism="Graded silica aerogel (20 mg/cm^3) cassette",
                dry_mass_kg=85.0,
                propellant_mass_kg=25.0,
            ),
            PlanetSpecificRover(
                target_planet="Saturn (Saturnian System)",
                target_sampling_body="Titan Atmosphere & Enceladus Plumes",
                rover_type="Dense-Atmosphere Coaxial Rotorcraft & Cryo-Glider",
                operating_environment="Dense nitrogen-methane atmosphere (1.5 bar) & cryogenic vacuum",
                atmospheric_pressure_bar=1.45,
                ambient_temperature_k=94.0,
                surface_gravity_mps2=1.352,  # Titan
                primary_locomotion_mode="Atmospheric flight (coaxial counter-rotating rotors) on Titan",
                sampling_mechanism="Atmospheric cryo-gas intake & aerogel geyser trap",
                dry_mass_kg=110.0,
                propellant_mass_kg=20.0,
            ),
            PlanetSpecificRover(
                target_planet="Uranus (Uranian System)",
                target_sampling_body="Miranda Verona Rupes & Titania",
                rover_type="Deep-Cryogenic Vacuum Resonance Skimmer",
                operating_environment="Pristine deep-space ice giant vacuum",
                atmospheric_pressure_bar=0.0,
                ambient_temperature_k=55.0,
                surface_gravity_mps2=0.079,  # Miranda
                primary_locomotion_mode="Coplanar resonance transfer thrusters",
                sampling_mechanism="Chalcogenide low-temperature aerogel array",
                dry_mass_kg=65.0,
                propellant_mass_kg=18.0,
            ),
            PlanetSpecificRover(
                target_planet="Neptune (Neptunian System)",
                target_sampling_body="Triton Cryogenic Nitrogen Geysers",
                rover_type="Ultra-Deep-Freeze Cryo-Gold Nitrogen Probe",
                operating_environment="Rarefied nitrogen atmosphere (1.4 Pa) at 38 Kelvin",
                atmospheric_pressure_bar=1.4e-5,
                ambient_temperature_k=38.0,
                surface_gravity_mps2=0.779,  # Triton
                primary_locomotion_mode="Retrograde hyperbolic geyser intercept",
                sampling_mechanism="Cryo-gold coated aerogel trap at 38 K",
                dry_mass_kg=75.0,
                propellant_mass_kg=22.0,
            ),
        ]

    @classmethod
    def get_surface_observation_posts(cls) -> List[SurfaceObservationPost]:
        return [
            SurfaceObservationPost(
                landing_body="Dwarf Planet 1 Ceres",
                landing_site="Occator Crater (Cerealia Faculae brine floor)",
                surface_landing_dv_mps=285.0,
                braking_mechanism="Low-gravity pulsed monopropellant thruster descent",
                power_source="Small Radioisotope Thermoelectric Generator (RTG: 40 W)",
                operational_lifespan_years=15.0,
                science_payload=[
                    "Broadband 3-axis micro-seismometer (relict ocean brine movement)",
                    "Sublimation exosphere quadrupole mass spectrometer",
                    "Surface thermal gradient flux probe",
                    "Optical camera for seasonal faculae brightening",
                ],
                relay_telecom_link="UHF link (128 kbps) to Ceres Orbital Tanker Node -> Earth DSN/GEO",
            ),
            SurfaceObservationPost(
                landing_body="Callisto (Jupiter)",
                landing_site="Valhalla Basin Ringed Impact Plains",
                surface_landing_dv_mps=1420.0,
                braking_mechanism="Autonomous closed-loop throttleable bipropellant landing burn",
                power_source="Multi-Mission Radioisotope Thermoelectric Generator (MMRTG: 110 W)",
                operational_lifespan_years=25.0,
                science_payload=[
                    "Deep Jovian tidal strain seismometer (detects internal ocean tidal response)",
                    "Surface sputtering mass spectrometer (Jovian plasma interaction)",
                    "Long-baseline cosmic ray and Jovian magnetospheric detector",
                    "Broadband optical/laser ground station beacon",
                ],
                relay_telecom_link="Direct X/Ka-band link + UHF cross-link to Callisto Halo Logistics Depot",
            ),
            SurfaceObservationPost(
                landing_body="Titan (Saturn)",
                landing_site="Kraken Mare Organic Shorelines / Shangri-La Dunes",
                surface_landing_dv_mps=25.0,  # Atmosphere does 99% of deceleration!
                braking_mechanism="Dense-atmosphere supersonic drogue chute + parafoil gliding descent",
                power_source="Advanced Stirling Radioisotope Generator (ASRG: 140 W)",
                operational_lifespan_years=20.0,
                science_payload=[
                    "Meteorological station (methane rain, wind vectors, pressure cycles)",
                    "Liquid methane/ethane shoreline sonar and chemical sensor",
                    "Tholin organic sediment core spectrometer",
                    "Subsurface seismic array for internal water-ammonia ocean",
                ],
                relay_telecom_link="High-power S-band transceiver relay link to Titan Orbital Logistics Tanker",
            ),
            SurfaceObservationPost(
                landing_body="Titania (Uranus)",
                landing_site="Messina Chasma Fault Rift Basin",
                surface_landing_dv_mps=410.0,
                braking_mechanism="Terminal pulsed thruster descent into fault floor",
                power_source="Next-Gen Radioisotope Thermoelectric Generator (75 W)",
                operational_lifespan_years=20.0,
                science_payload=[
                    "Cryotectonic rift fracture seismometer",
                    "Ammonia-water cryomagmatic vent monitor",
                    "Deep-space interplanetary magnetic field magnetometer",
                ],
                relay_telecom_link="UHF cross-link to Titania Halo Orbital Depot Node",
            ),
            SurfaceObservationPost(
                landing_body="Triton (Neptune)",
                landing_site="Cantaloupe Terrain (South Polar Nitrogen Geyser Field)",
                surface_landing_dv_mps=860.0,
                braking_mechanism="Autonomous retrograde trajectory de-orbit burn and soft touchdown",
                power_source="Long-Life Radioisotope Thermoelectric Generator (90 W)",
                operational_lifespan_years=30.0,
                science_payload=[
                    "Nitrogen cryogeyser active eruption monitor (plume height and cadence)",
                    "Sub-surface ocean tidal dissipation seismometer",
                    "Solid nitrogen/methane ice phase sublimation monitor (38 K)",
                    "Interstellar boundary plasma wave instrument",
                ],
                relay_telecom_link="Optical laser transmitter relay link to Triton Halo Depot -> Earth GEO Hub",
            ),
        ]

    @classmethod
    def evaluate_interplanetary_network(cls) -> Dict[str, Any]:
        """
        Evaluate full architecture of planet-tailored rovers, surface posts, and relay mesh.
        """
        rovers = cls.get_planet_specific_rovers()
        posts = cls.get_surface_observation_posts()

        total_surface_posts = len(posts)
        cumulative_network_span_au = 30.07 - 2.77  # Distance span across Solar System
        total_surface_sensor_types = sum(len(p.science_payload) for p in posts)
        avg_surface_lifespan_years = sum(p.operational_lifespan_years for p in posts) / len(posts)

        return {
            "concept_name": "Project ARGONAUT: Persistent Surface Observatory & Relay Mesh Network",
            "total_planet_tailored_rovers": len(rovers),
            "total_permanent_surface_posts": total_surface_posts,
            "cumulative_astronomical_span_au": float(cumulative_network_span_au),
            "total_surface_instruments_deployed": total_surface_sensor_types,
            "average_post_lifespan_years": float(avg_surface_lifespan_years),
            "rovers": [
                {
                    "planet": r.target_planet,
                    "target": r.target_sampling_body,
                    "type": r.rover_type,
                    "environment": r.operating_environment,
                    "locomotion": r.primary_locomotion_mode,
                    "dry_mass_kg": r.dry_mass_kg,
                    "propellant_kg": r.propellant_mass_kg,
                }
                for r in rovers
            ],
            "surface_posts": [
                {
                    "landing_body": p.landing_body,
                    "site": p.landing_site,
                    "landing_dv_mps": p.surface_landing_dv_mps,
                    "braking": p.braking_mechanism,
                    "power": p.power_source,
                    "lifespan_years": p.operational_lifespan_years,
                    "payload_count": len(p.science_payload),
                    "relay_link": p.relay_telecom_link,
                }
                for p in posts
            ],
            "permanent_infrastructure_value": (
                "Zero hardware is wasted. When Project ARGONAUT's mobile mothership departs each planetary haven, "
                "it leaves behind a fully functional orbital depot relay node and a soft-landed surface observatory. "
                f"This creates a permanent {total_surface_posts}-station, {total_surface_sensor_types}-instrument "
                "interplanetary deep-space science mesh operating continuously for over 20-30 years."
            ),
        }
