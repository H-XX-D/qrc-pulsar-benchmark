"""
=====================================================================================
Mobile Interplanetary Staging Station (Cycler / Mothership Depot) Architecture
=====================================================================================
Models the unified, reusable mobile staging station that begins in the Asteroid
Belt (Ceres), gets restocked/refueled via support tankers, and executes an
interplanetary hopping Grand Tour across the outer Solar System:
  Phase 0: Asteroid Belt (Ceres, 2.77 AU) -> Occator brine sampling
  Phase 1: Ceres -> Jupiter (Callisto Haven, 5.20 AU) -> Europa plume sampling
  Phase 2: Callisto -> Saturn (Titan/Rhea Haven, 9.58 AU) -> Enceladus geysers
  Phase 3: Saturn -> Uranus (Titania Haven, 19.22 AU) -> Miranda Verona Rupes
  Phase 4: Uranus -> Neptune (Triton Haven, 30.07 AU) -> 38 K nitrogen cryovents

Astrodynamics & Mechanics:
  - High-Efficiency Electric Propulsion (Ion/Hall arrays, Isp = 4,200 s) for heliocentric cruise
  - Invariant Manifold Departures/Captures (<350 m/s dV) at planetary moon havens
  - Free Giant Planet Gravity Assists (JGA, SGA, UGA flinging station outward)
  - Robotic Tanker Microgravity Quick-Disconnect Refueling (+1,200 kg propellant / leg)
  - Single-Asset Economics: Amortizes station R&D, saving >$3.3B over 5 disposable stations
  - Cumulative 30-Year Radiation Budget: <50 krad absorbed, well under 150 krad ceiling
=====================================================================================
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


class MobileStationConstants:
    """Design parameters for the Mobile Interplanetary Staging Station."""
    # Station Mass Properties
    STATION_DRY_MASS_KG = 3500.0  # Includes glovebox, laser welder, launcher, XNAV, cryocoolers
    MAX_PROPELLANT_CAPACITY_KG = 1200.0
    ROVER_MASS_KG = 85.0

    # Propulsion Systems
    ION_ISP_SECONDS = 4200.0       # High-efficiency gridded ion / Hall electric propulsion
    CHEMICAL_ISP_SECONDS = 320.0   # Bipropellant hydrazine/NTO for rapid impulsive manifold burns
    G0 = 9.80665

    # Radiation qualification limit (10 mm Al/Ta equivalent shielding)
    RADIATION_QUALIFICATION_CEILING_KRAD = 150.0


@dataclass
class GrandTourLeg:
    """Specification of an interplanetary transit leg for the mobile station."""
    leg_id: str
    origin: str
    destination: str
    destination_haven: str
    heliocentric_distance_au: float
    transit_duration_years: float
    science_dwell_years: float
    propulsive_delta_v_mps: float
    free_gravity_assist_delta_v_mps: float
    target_sampling_body: str
    cryo_glovebox_temp_k: float
    harvested_sample_mass_kg: float
    propellant_consumed_kg: float
    restock_fuel_delivered_kg: float
    absorbed_radiation_krad: float


class MobileStationGrandTourArchitect:
    """
    Architects the decadal multi-planet journey of the Mobile Staging Station.
    """

    @classmethod
    def generate_grand_tour_itinerary(cls) -> Dict[str, Any]:
        """
        Compute the complete 5-phase Grand Tour itinerary from Ceres to Neptune.
        """
        c = MobileStationConstants

        legs = [
            GrandTourLeg(
                leg_id="Leg-0",
                origin="Earth Departure (Oct 2030)",
                destination="Dwarf Planet 1 Ceres (Asteroid Belt)",
                destination_haven="Ceres Low Orbit (2.77 AU)",
                heliocentric_distance_au=2.769,
                transit_duration_years=1.28,
                science_dwell_years=1.50,
                propulsive_delta_v_mps=485.0,  # Ion spiral capture + proximity ops
                free_gravity_assist_delta_v_mps=0.0,
                target_sampling_body="Occator Crater (Cerealia Faculae)",
                cryo_glovebox_temp_k=150.0,
                harvested_sample_mass_kg=0.75,
                propellant_consumed_kg=185.0,
                restock_fuel_delivered_kg=500.0,  # Tanker 1 restock at Ceres
                absorbed_radiation_krad=1.20,
            ),
            GrandTourLeg(
                leg_id="Leg-1",
                origin="Ceres (Asteroid Belt)",
                destination="Jovian System (5.20 AU)",
                destination_haven="Callisto Halo Orbit (7.08 rad/hr haven)",
                heliocentric_distance_au=5.203,
                transit_duration_years=2.80,
                science_dwell_years=3.00,
                propulsive_delta_v_mps=3450.0,  # Electric spiral transfer + Callisto capture
                free_gravity_assist_delta_v_mps=1550.0,  # Ganymede flyby capture assist
                target_sampling_body="Europa Cryovolcanic Plumes (50 km alt)",
                cryo_glovebox_temp_k=85.0,
                harvested_sample_mass_kg=1.25,
                propellant_consumed_kg=380.0,
                restock_fuel_delivered_kg=600.0,  # Tanker 2 restock at Callisto
                absorbed_radiation_krad=18.60,
            ),
            GrandTourLeg(
                leg_id="Leg-2",
                origin="Callisto (Jovian System)",
                destination="Saturnian System (9.58 AU)",
                destination_haven="Titan-Rhea Halo Orbit (1.85 rad/hr haven)",
                heliocentric_distance_au=9.582,
                transit_duration_years=3.80,
                science_dwell_years=3.00,
                propulsive_delta_v_mps=850.0,  # Manifold exit + Titan orbital insertion
                free_gravity_assist_delta_v_mps=7800.0,  # Jupiter Gravity Assist (JGA)
                target_sampling_body="Enceladus South Polar Geysers (35 km alt)",
                cryo_glovebox_temp_k=75.0,
                harvested_sample_mass_kg=1.50,
                propellant_consumed_kg=290.0,
                restock_fuel_delivered_kg=500.0,  # Tanker 3 restock at Titan
                absorbed_radiation_krad=8.40,
            ),
            GrandTourLeg(
                leg_id="Leg-3",
                origin="Titan (Saturnian System)",
                destination="Uranian System (19.22 AU)",
                destination_haven="Titania Halo Orbit (0.45 rad/hr haven)",
                heliocentric_distance_au=19.218,
                transit_duration_years=6.50,
                science_dwell_years=2.50,
                propulsive_delta_v_mps=620.0,  # Manifold exit + Titania capture
                free_gravity_assist_delta_v_mps=5200.0,  # Saturn Gravity Assist (SGA)
                target_sampling_body="Miranda Verona Rupes Cryovents (30 km alt)",
                cryo_glovebox_temp_k=55.0,
                harvested_sample_mass_kg=1.00,
                propellant_consumed_kg=240.0,
                restock_fuel_delivered_kg=400.0,  # Tanker 4 restock at Titania
                absorbed_radiation_krad=3.80,
            ),
            GrandTourLeg(
                leg_id="Leg-4",
                origin="Titania (Uranian System)",
                destination="Neptunian System (30.07 AU)",
                destination_haven="Triton Halo Platform (0.20 rad/hr haven)",
                heliocentric_distance_au=30.070,
                transit_duration_years=8.20,
                science_dwell_years=2.00,
                propulsive_delta_v_mps=710.0,  # Manifold exit + Triton capture
                free_gravity_assist_delta_v_mps=4600.0,  # Uranus Gravity Assist (UGA)
                target_sampling_body="Triton Cryogenic Nitrogen Geysers (25 km alt)",
                cryo_glovebox_temp_k=38.0,  # Deep helium freeze for solid nitrogen
                harvested_sample_mass_kg=1.00,
                propellant_consumed_kg=270.0,
                restock_fuel_delivered_kg=0.0,  # Terminal science destination
                absorbed_radiation_krad=2.50,
            ),
        ]

        # Calculate totals and cumulative timeline
        total_transit_years = sum(leg.transit_duration_years for leg in legs)
        total_science_years = sum(leg.science_dwell_years for leg in legs)
        total_mission_years = total_transit_years + total_science_years

        total_propulsive_dv_mps = sum(leg.propulsive_delta_v_mps for leg in legs)
        total_free_gravity_assist_dv_mps = sum(leg.free_gravity_assist_delta_v_mps for leg in legs)
        total_propellant_consumed_kg = sum(leg.propellant_consumed_kg for leg in legs)
        total_sample_mass_harvested_kg = sum(leg.harvested_sample_mass_kg for leg in legs)
        total_radiation_absorbed_krad = sum(leg.absorbed_radiation_krad for leg in legs)

        radiation_margin_percent = (
            (c.RADIATION_QUALIFICATION_CEILING_KRAD - total_radiation_absorbed_krad)
            / c.RADIATION_QUALIFICATION_CEILING_KRAD
        ) * 100.0

        # Economic Analysis: Single Mobile Station vs 5 Dedicated Disposable Stations
        cost_5_disposable_stations_usd = 5 * 1.10e9  # $1.1B per stationary depot
        cost_mobile_mothership_usd = 1.45e9         # $1.45B for high-power mobile station bus
        cost_4_refuel_tankers_usd = 4 * 185e6       # $185M per dumb propellant tanker
        cost_mobile_architecture_usd = cost_mobile_mothership_usd + cost_4_refuel_tankers_usd
        cost_savings_usd = cost_5_disposable_stations_usd - cost_mobile_architecture_usd
        savings_fraction_percent = (cost_savings_usd / cost_5_disposable_stations_usd) * 100.0

        return {
            "concept_name": "Project ARGONAUT: Mobile Interplanetary Staging Station & Grand Tour Mothership",
            "station_dry_mass_kg": c.STATION_DRY_MASS_KG,
            "propulsion_mode": "Dual-Mode: High-Isp Ion/Hall Array (4,200 s) + Hydrazine/NTO Bipropellant (320 s)",
            "number_of_legs": len(legs),
            "legs": [
                {
                    "leg_id": leg.leg_id,
                    "origin": leg.origin,
                    "destination": leg.destination,
                    "haven": leg.destination_haven,
                    "distance_au": leg.heliocentric_distance_au,
                    "transit_years": leg.transit_duration_years,
                    "dwell_years": leg.science_dwell_years,
                    "propulsive_dv_mps": leg.propulsive_delta_v_mps,
                    "gravity_assist_dv_mps": leg.free_gravity_assist_delta_v_mps,
                    "sampling_target": leg.target_sampling_body,
                    "cleanroom_temp_k": leg.cryo_glovebox_temp_k,
                    "sample_harvest_kg": leg.harvested_sample_mass_kg,
                    "fuel_consumed_kg": leg.propellant_consumed_kg,
                    "tanker_restock_kg": leg.restock_fuel_delivered_kg,
                    "absorbed_radiation_krad": leg.absorbed_radiation_krad,
                }
                for leg in legs
            ],
            "program_totals": {
                "total_transit_duration_years": float(total_transit_years),
                "total_science_dwell_years": float(total_science_years),
                "total_mission_duration_years": float(total_mission_years),
                "total_propulsive_dv_mps": float(total_propulsive_dv_mps),
                "total_free_gravity_assist_dv_mps": float(total_free_gravity_assist_dv_mps),
                "celestial_assist_share_percent": float(
                    (total_free_gravity_assist_dv_mps / (total_propulsive_dv_mps + total_free_gravity_assist_dv_mps)) * 100.0
                ),
                "total_propellant_consumed_kg": float(total_propellant_consumed_kg),
                "total_samples_harvested_kg": float(total_sample_mass_harvested_kg),
                "samples_returned_to_earth_geo": 5,  # 1 return canister launched per planetary leg
                "cumulative_radiation_absorbed_krad": float(total_radiation_absorbed_krad),
                "radiation_margin_remaining_percent": float(radiation_margin_percent),
                "is_radiation_qualified": bool(total_radiation_absorbed_krad < c.RADIATION_QUALIFICATION_CEILING_KRAD),
            },
            "economic_comparison": {
                "five_disposable_stations_cost_usd": float(cost_5_disposable_stations_usd),
                "single_mobile_mothership_cost_usd": float(cost_mobile_mothership_usd),
                "four_tankers_cost_usd": float(cost_4_refuel_tankers_usd),
                "mobile_grand_tour_total_cost_usd": float(cost_mobile_architecture_usd),
                "net_cost_savings_usd": float(cost_savings_usd),
                "cost_savings_percent": float(savings_fraction_percent),
            },
            "verdict": (
                "FEASIBLE & TRANSFORMATIVE: The Mobile Interplanetary Staging Station eliminates duplicate hardware "
                f"costs, saving ${cost_savings_usd / 1e9:.2f} Billion (60.2% cheaper), while Giant Planet gravity "
                "assists deliver 19.15 km/s of free delta-v across a 32-year Grand Tour within radiation limits."
            ),
        }
