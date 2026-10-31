"""Asteroid Belt Low-Gravity Industrial Factory Zone Architecture.

Formalizes the transformation of the Main Asteroid Belt (2.1 to 3.3 AU) into the
primary manufacturing, smelting, and logistics switchyard of the Solar System.

Pillars:
1. Gravitational Energy Disparity: Lifting manufactured structures and refined
   propellant from low-gravity asteroids (Ceres, Vesta, Psyche) consumes 470x to
   3,860x less energy than lifting from Earth's deep gravity well.
2. Elemental Specialization & Complementarity:
   - C-Type (Ceres, Hygiea): Water-ice (25% by mass), carbonates, ammonia, hydrocarbons.
   - M-Type (16 Psyche): Metallic nickel-iron core, cobalt, titanium, platinum group metals.
   - S/V-Type (Vesta): Basaltic silicates, aluminum, magnesium, structural glass.
3. Microgravity Manufacturing Physics:
   - Perfect vacuum (1e-11 bar) electron-beam 3D printing and vapor deposition.
   - Buoyancy-free metallurgy: Immiscible alloy metal foams impossible in 1-g.
4. Logistics Crossroads: Central fueling railhead between inner terrestrial worlds
   (Earth/Moon/Mars) and outer ocean worlds (Jupiter/Saturn/Uranus/Neptune).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass(frozen=True)
class AsteroidResourceProfile:
    """Resource inventory and astrodynamics of major belt bodies."""
    name: str
    spectral_type: str
    semi_major_axis_au: float
    surface_gravity_mps2: float
    escape_velocity_mps: float
    primary_commodity: str
    annual_feedstock_capacity_metric_tons: float
    energy_per_kg_to_escape_mj: float


@dataclass(frozen=True)
class IndustrialEnergyAdvantage:
    """Quantitative comparison of launching industrial output from Earth vs Asteroid Belt."""
    source_body: str
    escape_velocity_mps: float
    kinetic_energy_per_ton_gj: float
    energy_advantage_vs_earth_ratio: float
    rocket_mass_ratio_for_escape: float  # exp(v_esc / (Isp * g0)) with Isp=380s
    payload_fraction_pct: float


@dataclass(frozen=True)
class ManufacturingProcessSpec:
    """Microgravity industrial fabrication processes unique to low-gravity space."""
    process_name: str
    target_material: str
    vacuum_advantage: str
    microgravity_advantage: str
    terrestrial_failure_mode: str
    application: str


class AsteroidIndustrialZoneArchitect:
    """Synthesizes the industrial manufacturing capabilities of the Main Asteroid Belt."""

    def __init__(self):
        self.g0 = 9.80665
        self.isp_default_s = 380.0  # High-efficiency hydrolox or methalox

    def get_belt_commodity_catalog(self) -> Dict[str, AsteroidResourceProfile]:
        """Returns the four core pillars of the Asteroid Belt manufacturing complex."""
        profiles = {
            "Dwarf_Planet_1_Ceres": AsteroidResourceProfile(
                name="1 Ceres (Outpost Alpha)",
                spectral_type="C-type (Carbonaceous)",
                semi_major_axis_au=2.769,
                surface_gravity_mps2=0.284,
                escape_velocity_mps=516.4,
                primary_commodity="Water-Ice, Hydrated Salts, Cryogenic LH2/LOX Propellant",
                annual_feedstock_capacity_metric_tons=500000.0,
                energy_per_kg_to_escape_mj=0.5 * (516.4**2) / 1e6,  # 0.133 MJ/kg
            ),
            "Asteroid_16_Psyche": AsteroidResourceProfile(
                name="16 Psyche (The Metal Foundry)",
                spectral_type="M-type (Metallic Core)",
                semi_major_axis_au=2.924,
                surface_gravity_mps2=0.144,
                escape_velocity_mps=180.0,
                primary_commodity="Refined Nickel-Iron, Cobalt, Structural Titanium, Platinum Group Metals",
                annual_feedstock_capacity_metric_tons=1200000.0,
                energy_per_kg_to_escape_mj=0.5 * (180.0**2) / 1e6,  # 0.016 MJ/kg
            ),
            "Asteroid_4_Vesta": AsteroidResourceProfile(
                name="4 Vesta (Basaltic Quarry)",
                spectral_type="V-type (Basaltic Differentiated)",
                semi_major_axis_au=2.362,
                surface_gravity_mps2=0.250,
                escape_velocity_mps=360.0,
                primary_commodity="Basalt Fiber, Silicon, Aluminum Alloys, Radiation Shielding Slag",
                annual_feedstock_capacity_metric_tons=800000.0,
                energy_per_kg_to_escape_mj=0.5 * (360.0**2) / 1e6,  # 0.065 MJ/kg
            ),
            "Asteroid_10_Hygiea": AsteroidResourceProfile(
                name="10 Hygiea (Chemical Feedstock)",
                spectral_type="C-type (Volatile Clay)",
                semi_major_axis_au=3.139,
                surface_gravity_mps2=0.088,
                escape_velocity_mps=160.0,
                primary_commodity="Ammonia Clathrates, Carbon Monoxide, Hydrocarbons, Organic Polymers",
                annual_feedstock_capacity_metric_tons=350000.0,
                energy_per_kg_to_escape_mj=0.5 * (160.0**2) / 1e6,  # 0.013 MJ/kg
            ),
        }
        return profiles

    def compute_energy_advantage(self, source_name: str) -> IndustrialEnergyAdvantage:
        """Calculates the energy and mass fraction advantage vs Earth."""
        catalog = self.get_belt_commodity_catalog()
        earth_v_esc = 11186.0  # 11.186 km/s
        earth_ke_gj = 0.5 * (earth_v_esc**2) / 1e6  # ~62.56 GJ/ton

        if source_name == "Earth":
            v_esc = earth_v_esc
            ke_gj = earth_ke_gj
            ratio = 1.0
        elif source_name in catalog:
            spec = catalog[source_name]
            v_esc = spec.escape_velocity_mps
            ke_gj = 0.5 * (v_esc**2) / 1e6
            ratio = earth_ke_gj / ke_gj
        else:
            raise KeyError(f"Unknown source body: {source_name}")

        c_e = self.isp_default_s * self.g0  # Effective exhaust velocity ~3,726.5 m/s
        mass_ratio = float(np.exp(v_esc / c_e))
        payload_fraction_pct = (1.0 / mass_ratio) * 100.0

        return IndustrialEnergyAdvantage(
            source_body=source_name,
            escape_velocity_mps=round(v_esc, 1),
            kinetic_energy_per_ton_gj=round(ke_gj, 3),
            energy_advantage_vs_earth_ratio=round(ratio, 1),
            rocket_mass_ratio_for_escape=round(mass_ratio, 3),
            payload_fraction_pct=round(payload_fraction_pct, 2),
        )

    def get_manufacturing_processes(self) -> List[ManufacturingProcessSpec]:
        """Catalog of microgravity metallurgical and additive processes."""
        return [
            ManufacturingProcessSpec(
                process_name="Electron-Beam Vacuum Additive Manufacturing",
                target_material="Psyche Nickel-Iron & Titanium Powder",
                vacuum_advantage="Infinite 1e-11 bar vacuum eliminates argon shielding gas and oxide inclusions",
                microgravity_advantage="Enables hollow truss webs and kilometer-scale gossamer parabolic reflectors without gravitational sagging",
                terrestrial_failure_mode="Atmospheric oxidation, heat convection turbulence, and gravitational structural collapse during cooling",
                application="Megawatt-class solar power arrays, large space habitat trusses, and deep-space cycler hulls",
            ),
            ManufacturingProcessSpec(
                process_name="Immiscible Alloy Metal-Foam Solidification",
                target_material="Aluminum-Lead & Titanium-Bismuth Hybrids",
                vacuum_advantage="Deep thermal radiation cooling in shadow (40 K) allows rapid quench freezing",
                microgravity_advantage="Zero buoyancy separation: dense metals stay homogeneously suspended in ultra-light liquid metal foam",
                terrestrial_failure_mode="Heavy element settling (sedimentation) and bubble buoyancy collapse in 1-g",
                application="Hypervelocity micrometeoroid armor plates and ultra-lightweight structural bulkheads",
            ),
            ManufacturingProcessSpec(
                process_name="Solar-Thermal Vacuum Fractionation & Slag Refining",
                target_material="Vesta Basalt & Pyroxene Silicates",
                vacuum_advantage="Continuous differential distillation of volatile elements at lower boiling points in vacuum",
                microgravity_advantage="Crucible-free containerless electromagnetic levitation prevents container contamination",
                terrestrial_failure_mode="Crucible wall reaction and thermal conduction degradation",
                application="Continuous spun basalt fiber insulation, high-purity fused silica optics, and radiation shielding blocks",
            ),
            ManufacturingProcessSpec(
                process_name="Electrolytic Brine Cracking & Sabatier Synthesis",
                target_material="Ceres Occator Brine Carbonates",
                vacuum_advantage="Pure oxygen and hydrogen boiloff directly cryocooled at ambient equilibrium",
                microgravity_advantage="Gas-liquid microgravity phase separation via centrifugal cyclonic vortex",
                terrestrial_failure_mode="Industrial carbon footprint and heavy terrestrial plant footprint",
                application="Deep-space cryogenic rocket propellant depots (LH2/LOX and LCH4/LOX) powering outer-planet cyclers",
            ),
        ]

    def generate_full_industrial_matrix(self) -> Dict[str, Any]:
        """Compiles the complete economic and energy matrix of the belt factory zone."""
        catalog = self.get_belt_commodity_catalog()
        advantages = {k: self.compute_energy_advantage(k) for k in catalog.keys()}
        earth_adv = self.compute_energy_advantage("Earth")
        processes = self.get_manufacturing_processes()

        total_feedstock = sum(p.annual_feedstock_capacity_metric_tons for p in catalog.values())

        return {
            "total_annual_belt_feedstock_tons": total_feedstock,
            "commodities": catalog,
            "energy_advantages": advantages,
            "earth_baseline": earth_adv,
            "manufacturing_processes": processes,
            "crossroads_role": "Central Railhead & Fueling Switchyard connecting Inner Planets (Earth/Moon/Mars) to Outer Giants (Jupiter/Saturn)",
        }
