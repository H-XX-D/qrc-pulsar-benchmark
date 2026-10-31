"""
=====================================================================================
Tests for Mobile Interplanetary Staging Station (Cycler / Mothership Depot)
=====================================================================================
Verifies:
  1. Complete 5-phase Grand Tour itinerary from Ceres to Neptune.
  2. Sequential haven dwell at Ceres, Callisto, Titan, Titania, and Triton.
  3. Propulsive delta-v vs celestial gravity assist contributions (JGA, SGA, UGA).
  4. Robotic propellant restock ledger across 4 intermediate waystations.
  5. 30-year cumulative radiation dosage stays well below 150 krad qualification ceiling.
  6. Economic comparison: Single mobile mothership vs 5 disposable stations (>$3.3B savings).
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.mobile_station import (
    MobileStationConstants,
    GrandTourLeg,
    MobileStationGrandTourArchitect,
)


def test_mobile_station_itinerary_structure():
    """Verify the 5-leg sequential Grand Tour structure."""
    itinerary = MobileStationGrandTourArchitect.generate_grand_tour_itinerary()

    assert itinerary["number_of_legs"] == 5
    assert itinerary["station_dry_mass_kg"] == 3500.0

    legs = itinerary["legs"]
    assert legs[0]["destination"] == "Dwarf Planet 1 Ceres (Asteroid Belt)"
    assert legs[1]["destination"] == "Jovian System (5.20 AU)"
    assert legs[2]["destination"] == "Saturnian System (9.58 AU)"
    assert legs[3]["destination"] == "Uranian System (19.22 AU)"
    assert legs[4]["destination"] == "Neptunian System (30.07 AU)"

    # Verify destination havens
    assert "Ceres" in legs[0]["haven"]
    assert "Callisto" in legs[1]["haven"]
    assert "Titan" in legs[2]["haven"]
    assert "Titania" in legs[3]["haven"]
    assert "Triton" in legs[4]["haven"]


def test_grand_tour_astrodynamics_and_gravity_assists():
    """Verify delta-v ledger and free celestial gravity assists."""
    itinerary = MobileStationGrandTourArchitect.generate_grand_tour_itinerary()
    totals = itinerary["program_totals"]

    # Giant planets provide massive free delta-v (>15 km/s)
    assert totals["total_free_gravity_assist_dv_mps"] > 18000.0
    assert totals["total_propulsive_dv_mps"] < 7000.0
    assert totals["celestial_assist_share_percent"] > 70.0  # Over 70% of energy is free!

    # Verify key flyby boosts
    legs = itinerary["legs"]
    assert legs[2]["gravity_assist_dv_mps"] == 7800.0  # Jupiter flings station to Saturn
    assert legs[3]["gravity_assist_dv_mps"] == 5200.0  # Saturn flings station to Uranus
    assert legs[4]["gravity_assist_dv_mps"] == 4600.0  # Uranus flings station to Neptune


def test_propellant_consumption_and_tanker_restock():
    """Verify fuel consumption and refueling replenishment."""
    itinerary = MobileStationGrandTourArchitect.generate_grand_tour_itinerary()
    legs = itinerary["legs"]

    # Each intermediate haven receives tanker restock
    assert legs[0]["tanker_restock_kg"] == 500.0  # Tanker 1 at Ceres
    assert legs[1]["tanker_restock_kg"] == 600.0  # Tanker 2 at Callisto
    assert legs[2]["tanker_restock_kg"] == 500.0  # Tanker 3 at Titan
    assert legs[3]["tanker_restock_kg"] == 400.0  # Tanker 4 at Titania

    # Propellant consumed per leg is within mobile station tank limits (1,200 kg)
    for leg in legs:
        assert leg["fuel_consumed_kg"] < MobileStationConstants.MAX_PROPELLANT_CAPACITY_KG


def test_decadal_radiation_qualification():
    """Verify station survives 30-year odyssey within 150 krad qualification limit."""
    itinerary = MobileStationGrandTourArchitect.generate_grand_tour_itinerary()
    totals = itinerary["program_totals"]

    assert totals["is_radiation_qualified"] is True
    assert totals["cumulative_radiation_absorbed_krad"] < 50.0  # Under 50 krad total
    assert totals["radiation_margin_remaining_percent"] > 65.0  # >65% safety margin remaining


def test_economic_savings_over_disposable_fleet():
    """Verify cost reduction of single mobile mothership vs 5 disposable stations."""
    itinerary = MobileStationGrandTourArchitect.generate_grand_tour_itinerary()
    econ = itinerary["economic_comparison"]

    assert econ["five_disposable_stations_cost_usd"] == 5.5e9
    assert econ["mobile_grand_tour_total_cost_usd"] < 2.3e9
    assert econ["net_cost_savings_usd"] > 3.2e9          # Saves over $3.2 Billion
    assert econ["cost_savings_percent"] > 60.0           # Over 60% budget reduction
