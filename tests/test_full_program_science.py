"""
=====================================================================================
Tests for Full-Program Scientific Value & Master Astrobiological Valuation Engine
=====================================================================================
Verifies:
  1. The 3 core pillars of the full Project ARGONAUT science campaign.
  2. The 4 grand scientific breakthroughs (Second Genesis, Protoplanetary migration,
     Comparative oceanography, and Decadal tidal geophysics).
  3. Program quantitative metrics (5 systems, 5.50 kg returned, 19 surface instruments,
     27.3 AU baseline, 32-year temporal coverage).
  4. Master scientific synthesis report generation.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.full_program_science import (
    FullProgramScienceConstants,
    ScientificBreakthrough,
    FullProgramScienceArchitect,
)


def test_full_program_science_constants():
    """Verify physical and quantitative metrics of the full program."""
    c = FullProgramScienceConstants

    assert c.TOTAL_PLANETARY_SYSTEMS_SAMPLED == 5
    assert c.TOTAL_SAMPLE_MASS_RETURNED_KG == 5.50
    assert c.TOTAL_SURFACE_STATIONS_DEPLOYED == 5
    assert c.TOTAL_SURFACE_INSTRUMENTS_DEPLOYED == 19
    assert abs(c.TOTAL_ASTRONOMICAL_BASELINE_AU - 27.30) < 0.05
    assert c.PROGRAM_TEMPORAL_BASELINE_YEARS >= 30.0


def test_four_grand_scientific_breakthroughs():
    """Verify all 4 grand scientific breakthrough themes are represented."""
    breakthroughs = FullProgramScienceArchitect.get_grand_breakthroughs()
    assert len(breakthroughs) == 4

    themes = [b.theme for b in breakthroughs]
    assert any("Second Genesis" in t for t in themes)
    assert any("Protoplanetary Nebula Migration" in t or "Cosmochemistry" in t for t in themes)
    assert any("Comparative Oceanography" in t for t in themes)
    assert any("Decadal Tidal Dynamics" in t for t in themes)

    # Verify each breakthrough details hypotheses, evidence, and advantage
    for b in breakthroughs:
        assert len(b.primary_hypothesis_confronted) > 20
        assert len(b.mechanistic_evidence_required) > 20
        assert len(b.argonaut_data_source) > 10
        assert len(b.resolving_power_advantage) > 10
        assert len(b.transformative_impact) > 20


def test_full_program_scientific_valuation_synthesis():
    """Verify master scientific synthesis generation."""
    valuation = FullProgramScienceArchitect.evaluate_full_program_scientific_value()

    assert "Project ARGONAUT" in valuation["program_name"]
    assert "executive_summary" in valuation

    pillars = valuation["program_pillars"]
    assert "pillar_1_physical_sample_return" in pillars
    assert "pillar_2_permanent_surface_observatory_mesh" in pillars
    assert "pillar_3_interplanetary_cross_sectional_gradient" in pillars

    # Verify returned sample details
    p1 = pillars["pillar_1_physical_sample_return"]
    assert p1["total_samples_harvested_kg"] == 5.50
    assert len(p1["planetary_bodies"]) == 5

    # Verify surface mesh details
    p2 = pillars["pillar_2_permanent_surface_observatory_mesh"]
    assert p2["total_stations"] == 5
    assert p2["total_instruments"] == 19

    # Verify verdict
    assert "TRANSFORMATIVE" in valuation["scientific_significance_verdict"]
