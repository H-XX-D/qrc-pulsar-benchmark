"""
=====================================================================================
Adaptive Mothership, Outpost Colonizer, & Dynamic Resupply Chain Architecture
=====================================================================================
Models the unified role of the Project ARGONAUT Mothership as:
  1. Return Vehicle & Central Specimen Vault (consolidates 5.50 kg samples across 5 worlds)
  2. Outpost Colonizer & Infrastructure Seeder (establishes permanent robotic outposts
     with surface stations, orbital fuel depots, and telecommunications nodes)
  3. Dynamic Adaptive Resupply Chain (successive tankers adapt their manifests to
     earlier scientific discoveries, inject upgraded Earth technologies, and replace worn hardware)
=====================================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class AdaptiveMothershipConstants:
    """Core parameters for the Adaptive Mothership and Colonizer."""
    MOTHERSHIP_DRY_MASS_KG = 4500.0  # Modular bus, central vault, repair bay, manipulators
    CENTRAL_CRYOGENIC_VAULT_CAPACITY_KG = 15.0
    TOTAL_OUTPOSTS_SEEDED = 5
    NUMBER_OF_DYNAMIC_RESUPPLY_MISSIONS = 4


@dataclass
class ColonizerOutpostSpec:
    """Specification of an operational planetary outpost seeded by the Mothership."""
    outpost_name: str
    target_system: str
    haven_body: str
    orbital_depot_node_status: str
    surface_station_status: str
    isru_resource_targeted: str
    future_human_exploration_readiness: str
    telecom_relay_role: str


@dataclass
class AdaptiveResupplyManifest:
    """Specification of an adaptive, technology-refreshed resupply mission."""
    mission_id: str
    destination: str
    launch_year: int
    arrival_year: int
    propellant_payload_kg: float
    scientific_adaptations: List[str]  # Tuned to discoveries of previous legs
    technology_refresh_upgrades: List[str]  # Next-gen Earth hardware injected in-flight
    spare_parts_and_maintenance: List[str]
    commercial_launch_cost_usd: float


class AdaptiveMothershipArchitect:
    """
    Evaluates the Mothership's return vehicle capabilities, outpost seeding,
    and adaptive resupply chain dynamics.
    """

    @classmethod
    def get_seeded_outposts(cls) -> List[ColonizerOutpostSpec]:
        return [
            ColonizerOutpostSpec(
                outpost_name="Outpost Alpha (Ceres Hub)",
                target_system="Main Asteroid Belt",
                haven_body="Dwarf Planet 1 Ceres",
                orbital_depot_node_status="Active orbital fuel depot & low-gravity docking beacon",
                surface_station_status="Landed Occator micro-hopper observatory (15-yr RTG)",
                isru_resource_targeted="Water-ice clathrates, sodium carbonate, and hydrated salts",
                future_human_exploration_readiness="Precursor fuel staging depot for crewed Mars/Asteroid transit",
                telecom_relay_role="Primary inner-belt relay node linking deep-space craft to Earth",
            ),
            ColonizerOutpostSpec(
                outpost_name="Outpost Beta (Callisto Haven)",
                target_system="Jovian System",
                haven_body="Callisto (26.33 R_J)",
                orbital_depot_node_status="Permanent Jovian logistics platform & cleanroom storage depot",
                surface_station_status="Landed Valhalla Basin seismometer & laser ground station (25-yr MMRTG)",
                isru_resource_targeted="Subsurface ocean water ice & Jovian exospheric volatiles",
                future_human_exploration_readiness="Radiation-safe staging hub for future human Jovian expeditions",
                telecom_relay_role="Jovian system telecommunications gateway & DSN cross-link",
            ),
            ColonizerOutpostSpec(
                outpost_name="Outpost Gamma (Titan Logistics Base)",
                target_system="Saturnian System",
                haven_body="Titan & Rhea",
                orbital_depot_node_status="Titan-Rhea halo logistics station & propellant cache",
                surface_station_status="Landed Kraken Mare shoreline meteorology & ocean sonar (20-yr ASRG)",
                isru_resource_targeted="Abundant liquid methane/ethane rocket fuel & atmospheric nitrogen",
                future_human_exploration_readiness="Dense-atmosphere natural radiation shielding base for long-term outposts",
                telecom_relay_role="Saturnian planetary mesh router & Enceladus geyser monitor link",
            ),
            ColonizerOutpostSpec(
                outpost_name="Outpost Delta (Titania Sanctuary)",
                target_system="Uranian System",
                haven_body="Titania (436,000 km orbit)",
                orbital_depot_node_status="Active orbital depot in quiet 0.45 rad/hr radiation haven",
                surface_station_status="Landed Messina Chasma cryotectonic rift seismometer (20-yr RTG)",
                isru_resource_targeted="Ammonia-water cryomagmatic hydrates & CO2 clathrates",
                future_human_exploration_readiness="Outer Solar System deep-space refueling sanctuary",
                telecom_relay_role="Ice giant regional communications node at 19.2 AU",
            ),
            ColonizerOutpostSpec(
                outpost_name="Outpost Epsilon (Triton Platform)",
                target_system="Neptunian System",
                haven_body="Triton",
                orbital_depot_node_status="Permanent Kuiper Belt boundary platform & 38 K cryogenic depot",
                surface_station_status="Landed Cantaloupe Terrain nitrogen geyser cadence monitor (30-yr RTG)",
                isru_resource_targeted="Solid nitrogen ice, methane frost, and primordial noble gases",
                future_human_exploration_readiness="Ultimate Solar System deep-freeze outpost at 30.0 AU",
                telecom_relay_role="Interstellar boundary optical laser transmitter to Earth GEO",
            ),
        ]

    @classmethod
    def get_adaptive_resupply_missions(cls) -> List[AdaptiveResupplyManifest]:
        return [
            AdaptiveResupplyManifest(
                mission_id="Resupply-1 (Ceres -> Jupiter Window)",
                destination="Callisto Halo Haven (Jupiter)",
                launch_year=2032,
                arrival_year=2036,
                propellant_payload_kg=600.0,
                scientific_adaptations=[
                    "Re-tuned aerogel cassettes calibrated to Ceres Occator brine particle density",
                    "Added specific reagents for chiral amino acid separation based on Ceres kerogen findings",
                ],
                technology_refresh_upgrades=[
                    "Next-generation Gallium Nitride (GaN) high-efficiency RF transmitters",
                    "Upgraded radiation-hardened FPGA GNC coprocessor with 4x compute density",
                ],
                spare_parts_and_maintenance=[
                    "Replacement micro-thruster solenoid valves",
                    "Spare robotic end-effector fingers for cleanroom glovebox",
                ],
                commercial_launch_cost_usd=160e6,  # Commercial medium/heavy rideshare
            ),
            AdaptiveResupplyManifest(
                mission_id="Resupply-2 (Jupiter -> Saturn Window)",
                destination="Titan-Rhea Halo Haven (Saturn)",
                launch_year=2037,
                arrival_year=2041,
                propellant_payload_kg=500.0,
                scientific_adaptations=[
                    "Customized Enceladus plume capture funnels based on Europa grain velocity data",
                    "High-viscosity hydrocarbon intake filters for Titan atmosphere sampling",
                ],
                technology_refresh_upgrades=[
                    "Commercial solid-state lithium-metal batteries with 2x specific energy density",
                    "Deep-space optical laser communications terminal (10x bandwidth to Earth)",
                ],
                spare_parts_and_maintenance=[
                    "Laser welder Nd:YAG flashlamp replacement module",
                    "Cryocooler compressor seal refresh kit",
                ],
                commercial_launch_cost_usd=140e6,
            ),
            AdaptiveResupplyManifest(
                mission_id="Resupply-3 (Saturn -> Uranus Window)",
                destination="Titania Halo Haven (Uranus)",
                launch_year=2044,
                arrival_year=2050,
                propellant_payload_kg=400.0,
                scientific_adaptations=[
                    "Chalcogenide aerogel matrices tailored to ammonia-hydrate condensates",
                    "Specialized CO/CO2 clathrate isotopic gas mass spectrometry standards",
                ],
                technology_refresh_upgrades=[
                    "Advanced Stirling radioisotope power unit (ASRG) with 30% thermal efficiency",
                    "Optical frequency comb quantum clock for sub-centimeter autonomous XNAV",
                ],
                spare_parts_and_maintenance=[
                    "Ion thruster molybdenum accelerator grid replacement set",
                    "Autonomous docking latch mechanical springs and sensors",
                ],
                commercial_launch_cost_usd=110e6,
            ),
            AdaptiveResupplyManifest(
                mission_id="Resupply-4 (Uranus -> Neptune Window)",
                destination="Triton Halo Platform (Neptune)",
                launch_year=2052,
                arrival_year=2059,
                propellant_payload_kg=400.0,
                scientific_adaptations=[
                    "Cryo-gold thermal shielding calibrated to 38 Kelvin solid nitrogen stability",
                    "Deep-UV Raman spectrometer calibrated to Kuiper Belt complex tholins",
                ],
                technology_refresh_upgrades=[
                    "Superconducting nanowire single-photon detector (SNSPD) for deep-space laser comms",
                    "Self-healing radiation-tolerant memory arrays",
                ],
                spare_parts_and_maintenance=[
                    "Cryocooler Joule-Thomson expansion valve replacement",
                    "Main electric propulsion cathode replacement unit",
                ],
                commercial_launch_cost_usd=90e6,  # Matured commercial heavy launcher market
            ),
        ]

    @classmethod
    def evaluate_mothership_architecture(cls) -> Dict[str, Any]:
        """
        Synthesize the full Mothership Return, Colonizer Outpost, and Adaptive Resupply architecture.
        """
        c = AdaptiveMothershipConstants
        outposts = cls.get_seeded_outposts()
        resupplies = cls.get_adaptive_resupply_missions()

        total_resupply_fuel_kg = sum(r.propellant_payload_kg for r in resupplies)
        total_resupply_cost_usd = sum(r.commercial_launch_cost_usd for r in resupplies)
        total_adaptations_count = sum(len(r.scientific_adaptations) for r in resupplies)
        total_tech_refreshes_count = sum(len(r.technology_refresh_upgrades) for r in resupplies)

        return {
            "concept_name": "Project ARGONAUT: Adaptive Mothership, Outpost Colonizer & Resupply Architecture",
            "mothership_roles": {
                "role_1_canister_return_launcher": (
                    "Acts as the orbital staging and launch platform that prepares, encapsulates, "
                    "and launches the autonomous sample return canisters (Return Vehicles) toward Earth. "
                    "The Mothership itself remains deployed in the outer solar system as an Outpost Colonizer."
                ),
                "role_2_outpost_colonizer": (
                    "Precursor infrastructure seeder: establishes permanent, self-sustaining robotic outposts "
                    "at each planetary haven, featuring orbital fuel depots, soft-landed RTG surface observatories, "
                    "and ISRU resource prospecting for future human and robotic exploration."
                ),
                "role_3_adaptive_supply_chain": (
                    "Overcomes 20-year 'launch freeze' by utilizing modular resupply tankers that dynamically adapt "
                    "their manifests to earlier scientific discoveries and inject newer-generation Earth technologies in flight."
                ),
            },
            "seeded_outposts_count": len(outposts),
            "outposts": [
                {
                    "name": o.outpost_name,
                    "system": o.target_system,
                    "haven": o.haven_body,
                    "orbital_depot": o.orbital_depot_node_status,
                    "surface_post": o.surface_station_status,
                    "isru_resource": o.isru_resource_targeted,
                    "human_readiness": o.future_human_exploration_readiness,
                    "telecom": o.telecom_relay_role,
                }
                for o in outposts
            ],
            "dynamic_resupply_metrics": {
                "total_resupply_missions": len(resupplies),
                "total_fuel_delivered_kg": float(total_resupply_fuel_kg),
                "total_scientific_adaptations_injected": total_adaptations_count,
                "total_technology_refreshes_injected": total_tech_refreshes_count,
                "total_resupply_logistics_cost_usd": float(total_resupply_cost_usd),
                "average_cost_per_resupply_usd": float(total_resupply_cost_usd / len(resupplies)),
                "resupply_cost_trend": "Monotonically decreasing from $160M to $90M via commercial launcher maturity.",
            },
            "strategic_verdict": (
                "TRANSFORMATIVE OPERATIONAL DOCTRINE: The Mothership functions as an autonomous planetary colonizer "
                "and consolidated Earth return vehicle. Decadal missions no longer suffer from technological obsolescence, "
                "as adaptive resupply missions continuously upgrade the Mothership with cutting-edge Earth technology."
            ),
        }
