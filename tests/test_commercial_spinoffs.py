"""
=====================================================================================
Tests for Commercial Technology Transfer & Market Valuation Engine
=====================================================================================
Verifies:
  1. Six commercial technology transfer pillars derived from Project ARGONAUT.
  2. Total addressable market calculation ($69.0B across commercial aerospace).
  3. ROI multiplier averaging (~8.1x industrial spillover return).
  4. Technology readiness and sector targeting across OSAM, PNT, ADR, and ISRU.
  5. Executive economic impact synthesis.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.commercial_spinoffs import (
    CommercialPillar,
    CommercialTechnologyArchitect,
)


def test_six_commercial_pillars_presence():
    """Verify all 6 commercial spin-off pillars are defined."""
    pillars = CommercialTechnologyArchitect.get_commercial_pillars()
    assert len(pillars) == 6

    names = [p.technology_name for p in pillars]
    assert any("FPGA/QRC XNAV" in n for n in names)
    assert any("Refueling Depot" in n for n in names)
    assert any("Clohessy-Wiltshire" in n for n in names)
    assert any("Dual-Mode Micro-Propulsion" in n for n in names)
    assert any("Laser Hermetic Sealing" in n for n in names)
    assert any("Geosynchronous Orbiting Logistics Hub" in n for n in names)


def test_commercial_market_sizes_and_sectors():
    """Verify market sizing and commercial sector alignment."""
    pillars = CommercialTechnologyArchitect.get_commercial_pillars()

    for p in pillars:
        assert p.addressable_market_2032_usd > 5e9  # Each pillar targets >$5B market
        assert p.commercial_roi_multiplier > 4.0     # Solid ROI
        assert len(p.target_commercial_sectors) >= 2
        assert len(p.key_commercial_beneficiaries) >= 2
        assert len(p.readiness_for_commercialization) > 10


def test_aggregate_commercial_impact_evaluation():
    """Verify cumulative market calculations and executive reporting."""
    impact = CommercialTechnologyArchitect.evaluate_total_commercial_impact()

    assert impact["total_commercial_pillars"] == 6
    assert impact["total_addressable_market_billions_usd"] >= 65.0  # Over $65 Billion
    assert impact["average_commercial_roi_multiplier"] > 7.0
    assert impact["projected_spillover_revenue_billions_usd"] > 10.0

    econ = impact["economic_conclusion"]
    assert "Billion" in econ
    assert "commercial aerospace market" in econ
    assert "commercial industrial spillover" in econ
