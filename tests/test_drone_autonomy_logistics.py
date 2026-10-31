"""
=====================================================================================
Tests for Drone Autonomy & Dual-Use Haven Augmentation Logistics Architecture
=====================================================================================
Verifies:
  1. Drone autonomy constants (sub-50 ns edge compute latency, >99% DSN commanding drop).
  2. The 5 Haven augmentation mission profiles (Ceres ISRU, Callisto cryobot, Titan sub,
     Titania gravitational array, Triton interstellar probe).
  3. Terrestrial drone autonomy transfers (VIO, edge FPGA, swarm mesh, distributed logistics).
  4. Multi-agent swarm scale (20 autonomous spacecraft).
  5. Executive feasibility evaluation synthesis.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.drone_autonomy_logistics import (
    DroneAutonomyConstants,
    HavenAugmentationMission,
    DroneAutonomyLogisticsArchitect,
)


def test_drone_autonomy_constants():
    """Verify onboard edge compute and communications metrics."""
    c = DroneAutonomyConstants
    assert c.EDGE_COMPUTE_POWER_WATTS < 5.0
    assert c.EDGE_COMPUTE_LATENCY_NANOSECONDS < 50.0
    assert c.LOCAL_LINK_LATENCY_MILLISECONDS < 5.0
    assert c.DSN_COMMANDING_OVERHEAD_REDUCTION_PERCENT > 99.0


def test_haven_augmentation_missions():
    """Verify secondary dual-use resupply missions augment haven stations."""
    augs = DroneAutonomyLogisticsArchitect.get_haven_augmentations()
    assert len(augs) == 5

    names = [a.haven_name for a in augs]
    assert any("Ceres" in n for n in names)
    assert any("Callisto" in n for n in names)
    assert any("Titan" in n for n in names)
    assert any("Titania" in n for n in names)
    assert any("Triton" in n for n in names)

    # Verify Ceres augmentation includes ISRU
    ceres = next(a for a in augs if "Ceres" in a.haven_name)
    assert "ISRU" in ceres.augmentation_theme
    assert "refueling" in ceres.new_mission_objective.lower()

    # Verify Callisto augmentation includes cryobot / astronomy
    callisto = next(a for a in augs if "Callisto" in a.haven_name)
    assert "Cryobot" in callisto.delivered_payload

    # Verify Titan includes submarine
    titan = next(a for a in augs if "Titan" in a.haven_name)
    assert "Submarine" in titan.delivered_payload or "submarine" in titan.new_mission_objective.lower()


def test_drone_tech_convergence_evaluation():
    """Verify convergence evaluation report and multi-agent coordination metrics."""
    res = DroneAutonomyLogisticsArchitect.evaluate_drone_tech_convergence()

    assert res["mothership_role"] == "Outpost Colonizer & Seed Ship"
    assert res["total_haven_augmentations"] == 5
    assert res["total_autonomous_agents_in_mesh"] == 20
    assert res["dsn_telecommanding_reduction_percent"] > 99.0

    techs = res["drone_technology_transfers"]
    assert len(techs) == 4
    tech_names = [t["terrestrial_drone_technology"] for t in techs]
    assert any("Visual-Inertial Odometry" in n for n in tech_names)
    assert any("FPGA" in n for n in tech_names)
    assert any("Swarm" in n for n in tech_names)

    assert "FULLY FEASIBLE TODAY" in res["feasibility_assessment"]
