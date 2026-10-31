"""Unit tests for the Asteroid Belt Low-Gravity Industrial Factory Zone Architecture."""

import pytest
import numpy as np

from qrc_pulsar_bench.asteroid_industrial_zone import (
    AsteroidIndustrialZoneArchitect,
    AsteroidResourceProfile,
    IndustrialEnergyAdvantage,
    ManufacturingProcessSpec,
)


def test_belt_commodity_catalog():
    """Verifies the complementary elemental composition of the 4 belt pillars (Ceres, Psyche, Vesta, Hygiea)."""
    architect = AsteroidIndustrialZoneArchitect()
    catalog = architect.get_belt_commodity_catalog()

    assert len(catalog) == 4
    assert "Dwarf_Planet_1_Ceres" in catalog
    assert "Asteroid_16_Psyche" in catalog
    assert "Asteroid_4_Vesta" in catalog
    assert "Asteroid_10_Hygiea" in catalog

    # Ceres provides water-ice and propellant
    ceres = catalog["Dwarf_Planet_1_Ceres"]
    assert "Water-Ice" in ceres.primary_commodity
    assert ceres.escape_velocity_mps == 516.4

    # Psyche provides structural metals and nickel-iron
    psyche = catalog["Asteroid_16_Psyche"]
    assert "Nickel-Iron" in psyche.primary_commodity
    assert psyche.escape_velocity_mps == 180.0


def test_gravitational_energy_advantage():
    """Verifies that launching industrial products from the belt uses 400x to 3,800x less energy than Earth."""
    architect = AsteroidIndustrialZoneArchitect()

    earth = architect.compute_energy_advantage("Earth")
    ceres = architect.compute_energy_advantage("Dwarf_Planet_1_Ceres")
    psyche = architect.compute_energy_advantage("Asteroid_16_Psyche")
    vesta = architect.compute_energy_advantage("Asteroid_4_Vesta")

    # Earth requires ~62.5 GJ/ton
    assert earth.kinetic_energy_per_ton_gj > 60.0
    assert earth.payload_fraction_pct < 10.0  # Under 10% payload fraction to escape

    # Ceres consumes >450x less energy than Earth
    assert ceres.energy_advantage_vs_earth_ratio > 450.0
    assert ceres.payload_fraction_pct > 80.0  # >85% payload fraction

    # 16 Psyche consumes >3,500x less energy than Earth
    assert psyche.energy_advantage_vs_earth_ratio > 3500.0
    assert psyche.payload_fraction_pct > 90.0  # >95% payload fraction


def test_manufacturing_processes():
    """Verifies the 4 microgravity/vacuum manufacturing processes."""
    architect = AsteroidIndustrialZoneArchitect()
    processes = architect.get_manufacturing_processes()

    assert len(processes) == 4
    names = [p.process_name for p in processes]
    assert "Electron-Beam Vacuum Additive Manufacturing" in names
    assert "Immiscible Alloy Metal-Foam Solidification" in names
    assert "Solar-Thermal Vacuum Fractionation & Slag Refining" in names
    assert "Electrolytic Brine Cracking & Sabatier Synthesis" in names

    # Immiscible alloys must solve terrestrial 1-g buoyancy separation
    metal_foam = [p for p in processes if "Metal-Foam" in p.process_name][0]
    assert "buoyancy" in metal_foam.microgravity_advantage.lower()


def test_full_industrial_matrix():
    """Verifies total annual feedstock capacity across the belt manufacturing hub."""
    architect = AsteroidIndustrialZoneArchitect()
    matrix = architect.generate_full_industrial_matrix()

    assert matrix["total_annual_belt_feedstock_tons"] >= 2000000.0  # Over 2 million tons/yr capacity
    assert "Central Railhead" in matrix["crossroads_role"]
