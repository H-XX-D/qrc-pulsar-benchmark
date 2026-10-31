"""
=====================================================================================
Tests for Scientific Value Engine & Astrobiological Biosignature Detection
=====================================================================================
Verifies:
  1. Enantiomeric excess (homochirality) discrimination between abiotic and biotic amino acids.
  2. Carbon Preference Index (CPI) lipid clustering vs smooth thermodynamic Fischer-Tropsch decay.
  3. Stable carbon isotope fractionation (kinetic delta^13C depletion).
  4. Hydrothermal chemical disequilibrium and Gibbs free energy affinity for methanogenesis.
  5. Mass spectrometry resolving power advantage of Earth GEO laboratory over in-situ flight probes.
  6. Target-specific science valuations across Ceres, Europa, Enceladus, and Triton.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.scientific_value import (
    ScienceConstants,
    AstrobiologyBiosignatureEvaluator,
    LaboratoryVsInSituComparison,
    PlanetaryScienceValuationEngine,
    synthesize_project_argonaut_scientific_value,
)


def test_enantiomeric_excess_homochirality():
    """Verify chiral amino acid evaluation discriminates biotic vs abiotic."""
    evaluator = AstrobiologyBiosignatureEvaluator

    # Racemic mixture (Abiotic)
    abiotic = evaluator.evaluate_enantiomeric_excess(d_conc=0.505, l_conc=0.495)
    assert abiotic["enantiomeric_excess"] < 0.05
    assert abiotic["is_abiotic_racemic"] is True
    assert abiotic["is_homochiral"] is False
    assert "ABIOTIC" in abiotic["interpretation"]

    # Strict L-enantiomer excess (Biotic)
    biotic = evaluator.evaluate_enantiomeric_excess(d_conc=0.03, l_conc=0.97)
    assert biotic["enantiomeric_excess"] > 0.90
    assert biotic["is_homochiral"] is True
    assert biotic["is_abiotic_racemic"] is False
    assert "STRONG_BIOTIC_SIGNATURE" in biotic["interpretation"]


def test_carbon_preference_index_lipid_periodicity():
    """Verify Carbon Preference Index (CPI) detects enzymatic building-block additions."""
    evaluator = AstrobiologyBiosignatureEvaluator

    # Biotic: Strong even-carbon fatty acid preference (C14, C16, C18, C20, C22)
    biotic_lipids = {
        13: 0.05, 14: 1.20,
        15: 0.08, 16: 3.50,
        17: 0.12, 18: 4.80,
        19: 0.09, 20: 1.80,
        21: 0.04, 22: 0.95,
        23: 0.02, 24: 0.40,
    }
    b_res = evaluator.evaluate_carbon_preference_index(biotic_lipids)
    assert b_res["carbon_preference_index"] > 10.0
    assert b_res["is_biogenic_clustering"] is True
    assert b_res["is_abiotic_smooth"] is False
    assert "BIOTIC_LIPID_CLUSTERING" in b_res["interpretation"]

    # Abiotic: Smooth Anderson-Schulz-Flory / Fischer-Tropsch decay (alpha ~ 0.65)
    abiotic_lipids = {c: 100.0 * (0.65 ** (c - 13)) for c in range(13, 25)}
    a_res = evaluator.evaluate_carbon_preference_index(abiotic_lipids)
    assert abs(a_res["carbon_preference_index"] - 1.0) < 0.20
    assert a_res["is_abiotic_smooth"] is True
    assert a_res["is_biogenic_clustering"] is False
    assert "ABIOTIC_SMOOTH_DISTRIBUTION" in a_res["interpretation"]


def test_carbon_isotopic_fractionation():
    """Verify delta^13C kinetic fractionation against inorganic carbonate."""
    evaluator = AstrobiologyBiosignatureEvaluator

    # Biotic carbon fixation (strong 12C preference, -30 permil depletion)
    biotic = evaluator.evaluate_carbon_isotopic_fractionation(
        delta13c_organic_permil=-34.0,
        delta13c_inorganic_carbonate_permil=-2.0,
    )
    assert biotic["fractionation_delta_permil"] == -32.0
    assert biotic["is_biogenic_fractionation"] is True
    assert "STRONG_KINETIC_DEPLETION" in biotic["interpretation"]

    # Abiotic equilibrium or small fractionation (-4 permil)
    abiotic = evaluator.evaluate_carbon_isotopic_fractionation(
        delta13c_organic_permil=-6.0,
        delta13c_inorganic_carbonate_permil=-2.0,
    )
    assert abiotic["fractionation_delta_permil"] == -4.0
    assert abiotic["is_biogenic_fractionation"] is False
    assert "ABIOTIC" in abiotic["interpretation"]


def test_methanogenesis_gibbs_free_energy_affinity():
    """Verify hydrothermal chemical disequilibrium calculation for ocean-world habitability."""
    evaluator = AstrobiologyBiosignatureEvaluator

    # Enceladus/Europa hydrothermal vent conditions: abundant H2 from serpentinization
    affinity = evaluator.evaluate_methanogenesis_gibbs_affinity(
        temp_kelvin=273.15,
        p_h2_bar=0.05,
        p_co2_bar=0.005,
        p_ch4_bar=0.0001,
    )

    assert affinity["delta_g_kj_per_mol"] < -50.0  # Strongly exergonic
    assert affinity["is_metabolically_viable"] is True
    assert "Active hydrothermal chemical disequilibrium" in affinity["energy_density_assessment"]


def test_laboratory_vs_in_situ_analytical_advantage():
    """Verify Earth GEO laboratory resolving power and contamination advantage."""
    comp = LaboratoryVsInSituComparison.compare_analytical_capabilities()

    ms = comp["mass_spectrometry_resolving_power"]
    assert ms["earth_geo_laboratory"] >= 1e6
    assert ms["resolution_advantage_factor"] >= 50.0
    assert "13CH4" in ms["critical_impact"]
    assert "14N2" in ms["critical_impact"]

    sample_pres = comp["sample_preservation_integrity"]
    assert "Graded aerogel" in sample_pres["argonaut_sample_return"]

    micro = comp["microscopic_and_isotopic_resolution"]
    assert "NanoSIMS" in micro["earth_geo_laboratory"]


def test_planetary_science_target_valuations():
    """Verify target-specific science returns across Ceres, Europa, Enceladus, Triton."""
    ceres = PlanetaryScienceValuationEngine.evaluate_target_science("ceres")
    assert "Occator" in ceres["target_feature"]
    assert any("Na2CO3" in m for m in ceres["harvestable_materials"])

    europa = PlanetaryScienceValuationEngine.evaluate_target_science("europa")
    assert "Cryovolcanic Plumes" in europa["target_feature"]
    assert any("ocean" in q.lower() for q in europa["primary_scientific_questions"])

    enceladus = PlanetaryScienceValuationEngine.evaluate_target_science("enceladus")
    assert "Tiger Stripe" in enceladus["target_feature"]
    assert any("silica nanoparticles" in m.lower() for m in enceladus["harvestable_materials"])

    triton = PlanetaryScienceValuationEngine.evaluate_target_science("triton_neptune")
    assert "Kuiper Belt" in triton["fundamental_discoveries_enabled"]


def test_master_scientific_value_synthesis():
    """Verify full synthesis payload generation."""
    dossier = synthesize_project_argonaut_scientific_value()

    assert "Project ARGONAUT" in dossier["executive_summary"]
    assert "homochirality_demo" in dossier["biosignature_criteria"]
    assert "ceres" in dossier["planetary_targets_evaluation"]
    assert "europa" in dossier["planetary_targets_evaluation"]
