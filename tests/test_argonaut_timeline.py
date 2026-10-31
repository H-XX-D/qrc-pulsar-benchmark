"""
=====================================================================================
Tests for Project ARGONAUT Programmatic Lifecycle & Flight Timeline
=====================================================================================
Verifies:
  1. Timeline initialization anchored on Summer 2027 (July 2027) authorization.
  2. Phase progression: Formulation (Phase A) -> Preliminary (Phase B) ->
     Manufacturing (Phase C) -> Integration & Launch (Phase D).
  3. Earth-to-Jupiter transit duration (~3.25 years, Aug 2033 launch -> Dec 2036 arrival).
  4. Europa plume sampling campaign & Canister 1 launch (Oct 2038).
  5. Tanker depot restock arrival at Callisto Station (Jun 2041).
  6. Earth GEO capture of Canister 1 (Apr 2043, after 4.5-year resonant cruise).
  7. Fleet expansion to Saturn, Uranus, and Neptune (2042 - 2055+).
  8. Executive timeline summary formatting.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.argonaut_timeline import (
    ArgonautMilestone,
    ArgonautTimelineManager,
)


def test_argonaut_milestones_sequence_and_phases():
    """Verify master sequence of milestones from Phase A through Phase F."""
    milestones = ArgonautTimelineManager.get_milestones()
    assert len(milestones) == 10

    phases = [m.phase for m in milestones]
    assert phases == [
        "Phase A",
        "Phase B",
        "Phase C",
        "Phase D",
        "Phase E1",
        "Phase E2",
        "Phase E3",
        "Phase E4",
        "Phase E5",
        "Phase F",
    ]

    # Verify kickoff in Summer 2027
    assert milestones[0].start_date == "2027-07-01"
    assert milestones[0].phase == "Phase A"
    assert "Requirements" in milestones[0].milestone_name

    # Verify Launch Campaign in August 2033
    launch_milestone = milestones[3]
    assert launch_milestone.phase == "Phase D"
    assert "2033" in launch_milestone.end_date
    assert "Launch" in launch_milestone.milestone_name


def test_argonaut_flight_chronology_and_transit_physics():
    """Verify astrodynamic timings match validated orbital transit constraints."""
    milestones = {m.phase: m for m in ArgonautTimelineManager.get_milestones()}

    # Cruise E1: Aug 2033 to Nov 2036 (~3.25 years)
    e1 = milestones["Phase E1"]
    assert e1.start_date.startswith("2033-08")
    assert e1.end_date.startswith("2036-11")
    assert e1.duration_months == 39

    # Callisto Staging E2: Dec 2036 to May 2037
    e2 = milestones["Phase E2"]
    assert e2.start_date.startswith("2036-12")
    assert e2.end_date.startswith("2037-05")

    # Primary Sampling E3: Jun 2037 to Dec 2039 (Canister 1 launched Oct 2038)
    e3 = milestones["Phase E3"]
    assert "2037" in e3.start_date
    assert "Canister 1 sealed and launched to Earth (Oct 2038)" in e3.deliverables

    # Tanker Restock E4: Jun 2041
    e4 = milestones["Phase E4"]
    assert "2041" in e4.start_date
    assert "+300 kg fuel" in e4.deliverables

    # Canister 1 Earth GEO Arrival E5: Apr 2043 (4.5 years from Oct 2038)
    e5 = milestones["Phase E5"]
    assert e5.start_date.startswith("2043-04")
    assert "Aero-GEO brake" in e5.deliverables
    assert "GEO cleanroom" in e5.deliverables


def test_argonaut_timeline_summary_formatting():
    """Verify executive summary table contains key dates and flight milestones."""
    summary = ArgonautTimelineManager.generate_timeline_summary()

    assert "PROJECT ARGONAUT: MASTER PROGRAMMATIC FLIGHT & DEVELOPMENT TIMELINE" in summary
    assert "Authorization Target: Summer 2027" in summary
    assert "First Ocean Sample at Earth GEO: April 2043" in summary
    assert "Phase A" in summary
    assert "Phase D" in summary
    assert "Phase E3" in summary
    assert "Phase E5" in summary
    assert "Phase F" in summary
    assert "August 2033" in summary
    assert "April 2043" in summary
