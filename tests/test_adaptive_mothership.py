"""
=====================================================================================
Tests for Adaptive Mothership, Outpost Colonizer, & Dynamic Resupply Chain
=====================================================================================
Verifies:
  1. Mothership design constants and role definitions (Return Vehicle + Colonizer).
  2. The 5 seeded planetary outposts (Ceres, Callisto, Titan, Titania, Triton).
  3. Dynamic adaptive resupply manifests (science adaptations, technology refreshes, spares).
  4. Commercial launch cost reductions over time ($160M down to $90M).
  5. Executive synthesis reporting.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.adaptive_mothership import (
    AdaptiveMothershipConstants,
    ColonizerOutpostSpec,
    AdaptiveResupplyManifest,
    AdaptiveMothershipArchitect,
)


def test_adaptive_mothership_constants_and_roles():
    """Verify physical parameters and core operational roles."""
    c = AdaptiveMothershipConstants
    assert c.MOTHERSHIP_DRY_MASS_KG == 4500.0
    assert c.CENTRAL_CRYOGENIC_VAULT_CAPACITY_KG == 15.0
    assert c.TOTAL_OUTPOSTS_SEEDED == 5
    assert c.NUMBER_OF_DYNAMIC_RESUPPLY_MISSIONS == 4

    eval_res = AdaptiveMothershipArchitect.evaluate_mothership_architecture()
    roles = eval_res["mothership_roles"]
    assert "role_1_canister_return_launcher" in roles
    assert "role_2_outpost_colonizer" in roles
    assert "role_3_adaptive_supply_chain" in roles


def test_five_seeded_outposts():
    """Verify outpost specifications across all 5 planetary systems."""
    outposts = AdaptiveMothershipArchitect.get_seeded_outposts()
    assert len(outposts) == 5

    names = [o.outpost_name for o in outposts]
    assert any("Ceres" in n for n in names)
    assert any("Callisto" in n for n in names)
    assert any("Titan" in n for n in names)
    assert any("Titania" in n for n in names)
    assert any("Triton" in n for n in names)

    for o in outposts:
        assert len(o.isru_resource_targeted) > 10
        assert len(o.future_human_exploration_readiness) > 10
        assert len(o.telecom_relay_role) > 10


def test_adaptive_resupply_chain_dynamics():
    """Verify resupply missions dynamically inject science adaptations and technology upgrades."""
    missions = AdaptiveMothershipArchitect.get_adaptive_resupply_missions()
    assert len(missions) == 4

    # Verify launch years are sequential
    years = [m.launch_year for m in missions]
    assert years == sorted(years)
    assert years[0] == 2032
    assert years[-1] == 2052

    # Verify launch cost monotonically decreases
    costs = [m.commercial_launch_cost_usd for m in missions]
    assert costs[0] == 160e6
    assert costs[-1] == 90e6
    for i in range(len(costs) - 1):
        assert costs[i] >= costs[i + 1]

    # Verify every mission carries science adaptations and tech upgrades
    for m in missions:
        assert len(m.scientific_adaptations) >= 2
        assert len(m.technology_refresh_upgrades) >= 2
        assert len(m.spare_parts_and_maintenance) >= 2
        assert m.propellant_payload_kg >= 400.0


def test_mothership_synthesis_evaluation():
    """Verify master evaluation metrics and strategic conclusion."""
    res = AdaptiveMothershipArchitect.evaluate_mothership_architecture()

    assert res["seeded_outposts_count"] == 5
    metrics = res["dynamic_resupply_metrics"]
    assert metrics["total_resupply_missions"] == 4
    assert metrics["total_fuel_delivered_kg"] == 1900.0
    assert metrics["total_scientific_adaptations_injected"] >= 8
    assert metrics["total_technology_refreshes_injected"] >= 8
    assert metrics["total_resupply_logistics_cost_usd"] < 550e6  # Under $550M total resupply cost
    assert "TRANSFORMATIVE" in res["strategic_verdict"]
