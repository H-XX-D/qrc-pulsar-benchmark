"""
=====================================================================================
Scientific Value Assessment Engine for Project ARGONAUT & Sample Return
=====================================================================================
Models the quantitative geochemical, astrobiological, and planetary science yield
of pristine cryogenic sample return across:
  1. Ceres (ARGONAUT-Pathfinder: $2.77 AU, Occator Crater brine faculae)
  2. Europa (ARGONAUT-Flagship: Jovian global subsurface ocean plume)
  3. Enceladus (Saturn: Alkaline hydrothermal geysers & silica nanoparticles)
  4. Miranda / Titania (Uranus: Ice giant mantle volatiles & cryotectonics)
  5. Triton (Neptune: 38 K captured Kuiper Belt hyper-volatiles)

Mathematical & Biosignature Formulations:
  - Enantiomeric Excess (Homochirality): ee = |D - L| / (D + L)
  - Carbon Preference Index (CPI for lipid/fatty acid biogenicity):
      CPI = 0.5 * [ sum(C_even[14..22]) / sum(C_odd[13..21]) + sum(C_even[14..22]) / sum(C_odd[15..23]) ]
  - Stable Isotopic Fractionation (delta^13C relative to VPDB standard):
      delta^13C = ((R_sample / R_standard) - 1.0) * 1000.0 (permil)
  - Hydrothermal Gibbs Free Energy Affinity (Methanogenesis):
      Delta_G = Delta_G0 + R * T * ln( (a_CH4 * a_H2O^2) / (a_CO2 * a_H2^4) )
  - Mass Spectrometer Resolving Power: R = m / Delta_m
=====================================================================================
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ScienceConstants:
    """Fundamental physical and geochemical reference constants."""
    R_GAS = 8.314462618  # J / (mol * K)
    VPDB_13C_RATIO = 0.0112372  # Vienna Pee Dee Belemnite standard 13C/12C
    VSMOW_D_RATIO = 0.00015576  # Vienna Standard Mean Ocean Water 2H/1H

    # Standard Gibbs free energy of reaction at 298.15 K, 1 bar (kJ/mol)
    # CO2(aq) + 4 H2(aq) -> CH4(aq) + 2 H2O(l)
    DELTA_G0_METHANOGENESIS_KJ_PER_MOL = -131.0


class AstrobiologyBiosignatureEvaluator:
    """
    Quantitative evaluation of astrobiological signatures in harvested ocean/brine samples.
    Discriminates between abiotic Fischer-Tropsch-type (FTT) / Strecker synthesis
    and authentic biological molecular organization.
    """

    @staticmethod
    def evaluate_enantiomeric_excess(d_conc: float, l_conc: float) -> Dict[str, Any]:
        """
        Calculate chiral enantiomeric excess (ee) for amino acids or sugars.
        Abiotic synthesis yields racemic mixtures (ee ~ 0.00 +/- 0.05).
        Terrestrial and hypothetical extraterrestrial biology requires strict homochirality (ee > 0.60).
        """
        total = d_conc + l_conc
        if total <= 1e-15:
            ee = 0.0
        else:
            ee = abs(d_conc - l_conc) / total

        is_homochiral = ee > 0.60
        is_abiotic_racemic = ee < 0.10

        if is_homochiral:
            verdict = "STRONG_BIOTIC_SIGNATURE: Non-racemic homochirality indicates enzymatic enantioselective synthesis."
        elif is_abiotic_racemic:
            verdict = "CONSISTENT_WITH_ABIOTIC: Racemic distribution indicates abiotic Strecker or cosmochemical synthesis."
        else:
            verdict = "AMBIGUOUS: Partial enantiomeric enrichment; potential chiral prebiotic autocatalysis or degraded signature."

        return {
            "d_concentration": float(d_conc),
            "l_concentration": float(l_conc),
            "enantiomeric_excess": float(ee),
            "enantiomeric_excess_percent": float(ee * 100.0),
            "is_homochiral": bool(is_homochiral),
            "is_abiotic_racemic": bool(is_abiotic_racemic),
            "interpretation": verdict,
        }

    @staticmethod
    def evaluate_carbon_preference_index(
        chain_abundances: Dict[int, float]
    ) -> Dict[str, Any]:
        """
        Calculate Carbon Preference Index (CPI) for hydrocarbon/fatty acid series.
        Abiotic FTT synthesis produces an exponential decrease with carbon number:
            P(C_n) = (1 - alpha) * alpha^(n - 1), yielding CPI ~ 1.00 +/- 0.10.
        Biogenic lipid synthesis proceeds via discrete enzymatic building blocks
        (e.g., acetyl-CoA additions producing strong even-over-odd or odd-over-even clustering, CPI > 3.0 or < 0.33).
        """
        even_sum = sum(chain_abundances.get(c, 0.0) for c in range(14, 24, 2))
        odd_sum_lower = sum(chain_abundances.get(c, 0.0) for c in range(13, 23, 2))
        odd_sum_upper = sum(chain_abundances.get(c, 0.0) for c in range(15, 25, 2))

        if odd_sum_lower <= 1e-15 or odd_sum_upper <= 1e-15:
            cpi = 1.0
        else:
            cpi = 0.5 * ((even_sum / odd_sum_lower) + (even_sum / odd_sum_upper))

        is_biogenic_clustering = (cpi > 3.0) or (cpi < 0.33)
        is_abiotic_smooth = 0.85 <= cpi <= 1.15

        if is_biogenic_clustering:
            verdict = "BIOTIC_LIPID_CLUSTERING: Discrete chain-length periodicity indicates biosynthetic enzymatic assembly."
        elif is_abiotic_smooth:
            verdict = "ABIOTIC_SMOOTH_DISTRIBUTION: Log-linear thermodynamic decay matches abiotic Fischer-Tropsch growth."
        else:
            verdict = "INDETERMINATE: Modest chain selectivity; requires high-precision isotopic mass spectrometry."

        return {
            "carbon_preference_index": float(cpi),
            "is_biogenic_clustering": bool(is_biogenic_clustering),
            "is_abiotic_smooth": bool(is_abiotic_smooth),
            "interpretation": verdict,
        }

    @staticmethod
    def evaluate_carbon_isotopic_fractionation(
        delta13c_organic_permil: float,
        delta13c_inorganic_carbonate_permil: float,
    ) -> Dict[str, Any]:
        """
        Evaluate carbon kinetic isotope fractionation:
            Delta_delta13C = delta13C_organic - delta13C_carbonate
        Enzymatic carbon fixation (Rubisco, acetyl-CoA, reductive TCA) strongly favors 12C over 13C,
        depleting biomass by -20 to -45 permil relative to ambient dissolved inorganic carbon (DIC).
        Abiotic FTT synthesis typically produces reverse or negligible fractionation (+10 to -10 permil).
        """
        delta_split = delta13c_organic_permil - delta13c_inorganic_carbonate_permil
        is_biogenic_fractionation = delta_split < -20.0

        if is_biogenic_fractionation:
            verdict = (
                f"STRONG_KINETIC_DEPLETION (Delta = {delta_split:.1f}‰): "
                "Matches enzymatic 12C preference characteristic of autotrophic carbon fixation."
            )
        else:
            verdict = (
                f"ABIOTIC_OR_EQUILIBRIUM (Delta = {delta_split:.1f}‰): "
                "Consistent with thermodynamic abiotic carbon equilibrium or catalytic synthesis."
            )

        return {
            "delta13c_organic_permil": float(delta13c_organic_permil),
            "delta13c_inorganic_permil": float(delta13c_inorganic_carbonate_permil),
            "fractionation_delta_permil": float(delta_split),
            "is_biogenic_fractionation": bool(is_biogenic_fractionation),
            "interpretation": verdict,
        }

    @staticmethod
    def evaluate_methanogenesis_gibbs_affinity(
        temp_kelvin: float,
        p_h2_bar: float,
        p_co2_bar: float,
        p_ch4_bar: float,
    ) -> Dict[str, Any]:
        """
        Calculate chemical thermodynamic disequilibrium (Gibbs free energy affinity)
        for alkaline hydrothermal methanogenesis:
            CO2(aq) + 4 H2(aq) -> CH4(aq) + 2 H2O(l)
        A negative Delta_G indicates a viable metabolic energy source capable of sustaining life.
        """
        t = max(200.0, min(500.0, temp_kelvin))
        r = ScienceConstants.R_GAS

        # Reaction quotient Q_r
        q_r = p_ch4_bar / (p_co2_bar * (p_h2_bar ** 4.0) + 1e-30)
        q_r = max(1e-20, min(1e20, q_r))

        # Temperature-adjusted Delta_G0 (kJ/mol)
        delta_g0_j = ScienceConstants.DELTA_G0_METHANOGENESIS_KJ_PER_MOL * 1e3
        delta_g_j = delta_g0_j + r * t * math.log(q_r)
        delta_g_kj = delta_g_j / 1e3

        has_chemical_energy_for_life = delta_g_kj < -15.0  # Biological threshold ~15-20 kJ/mol ATP synthesis

        return {
            "temperature_kelvin": float(t),
            "h2_partial_pressure_bar": float(p_h2_bar),
            "co2_partial_pressure_bar": float(p_co2_bar),
            "ch4_partial_pressure_bar": float(p_ch4_bar),
            "reaction_quotient_log10": float(math.log10(q_r)),
            "delta_g_kj_per_mol": float(delta_g_kj),
            "is_metabolically_viable": bool(has_chemical_energy_for_life),
            "energy_density_assessment": (
                f"Metabolic energy yield is {-delta_g_kj:.1f} kJ/mol. "
                "Active hydrothermal chemical disequilibrium provides abundant fuel for chemosynthesis."
                if has_chemical_energy_for_life else
                "Thermodynamic chemical equilibrium; insufficient chemical disequilibrium to sustain methanogenesis."
            ),
        }


class LaboratoryVsInSituComparison:
    """
    Quantifies why returning pristine samples to Earth Geosynchronous Orbit
    outclasses in-situ flyby spacecraft instrumentation by orders of magnitude.
    """

    @classmethod
    def compare_analytical_capabilities(cls) -> Dict[str, Any]:
        return {
            "mass_spectrometry_resolving_power": {
                "in_situ_flyby_spacecraft": 25000,   # High-end spaceflight TOF / MASPEX (m / Delta_m)
                "earth_geo_laboratory": 1500000,     # Ultra-high-resolution FT-ICR / Orbitrap
                "resolution_advantage_factor": 60.0,
                "critical_impact": (
                    "Resolves identical nominal molecular masses (isobars) impossible in deep space: "
                    "e.g., separating 13CH4 (17.0347 Da) from 12CH3D (17.0375 Da) at 0.0028 Da split; "
                    "separating 14N2 (28.0061 Da) from 12C16O (27.9949 Da)."
                ),
            },
            "sample_preservation_integrity": {
                "in_situ_impact_sampling": "Hypervelocity flyby (>=2.0 km/s) causes severe shock-pyrolysis and fragmentation into small atomic ions.",
                "argonaut_sample_return": "Graded aerogel captures intact ice crystals and organics at <=120 deg C; encapsulated at 85 K / 38 K under vacuum.",
                "preservation_advantage": "Macromolecules, biopolymers, lipids, and quaternary cell-like structures remain physically intact.",
            },
            "microscopic_and_isotopic_resolution": {
                "in_situ_flyby_spacecraft": "Spatial resolution limited to ~1 mm bulk averaging; bulk isotope ratios with 2-5 permil error.",
                "earth_geo_laboratory": "NanoSIMS and Cryo-TEM yield sub-nanometer imaging and 0.05 permil isotopic precision on individual grains.",
                "scientific_impact": "Identifies micro-fossils, cellular membranes, and spatial compartmentalization inside individual ice grains.",
            },
            "repeatability_and_evolving_hypotheses": {
                "in_situ_flyby_spacecraft": "One-shot observation during flyby corridor; zero opportunity to modify instruments after launch.",
                "earth_geo_laboratory": "1.0 kg physical sample in permanent GEO archive; analyzed repeatedly over decades with next-generation instruments.",
            },
            "planetary_protection_and_biosafety": {
                "in_situ_mission": "Zero biological containment or validation protocol.",
                "argonaut_geo_station": "Infinite (>1,000,000 year) orbital lifetime outside Earth's biosphere; absolute quarantine with 24/7 gigabit optical link.",
            },
        }


class PlanetaryScienceValuationEngine:
    """
    Evaluates the distinct scientific return across the full Project ARGONAUT fleet.
    """

    @classmethod
    def evaluate_target_science(cls, target_name: str) -> Dict[str, Any]:
        targets = {
            "ceres": {
                "system": "Main Asteroid Belt",
                "body": "Dwarf Planet 1 Ceres",
                "target_feature": "Occator Crater (Cerealia & Vinalia Faculae)",
                "primary_scientific_questions": [
                    "Is Ceres a relict ocean world with an active brine mantle reservoir?",
                    "Did Ceres form beyond the gas giants and migrate inward (Grand Tack / Nice Model)?",
                    "What is the source of rich ammonia, sodium carbonate, and ammonium chloride salts?",
                    "Did prebiotic organic synthesis occur inside Ceres' interior before freezing?",
                ],
                "harvestable_materials": [
                    "Pristine sodium carbonate (Na2CO3) brine crusts",
                    "Ammonium chloride (NH4Cl) and phyllosilicates",
                    "Hydrated subsurface water-ice clathrates",
                    "Kerogen-like refractory macromolecular carbon",
                ],
                "fundamental_discoveries_enabled": (
                    "Determines nitrogen-to-carbon ratios of the protoplanetary disk and proves whether Ceres "
                    "harbored an ancient alkaline ocean analogous to Enceladus and Europa."
                ),
            },
            "europa": {
                "system": "Jovian System",
                "body": "Europa",
                "target_feature": "Cryovolcanic Plumes & Exospheric Ice Grains",
                "primary_scientific_questions": [
                    "Does Europa's global 100-km-deep liquid ocean harbor extraterrestrial life?",
                    "What is the chemical composition, pH, and redox state of the subsurface ocean?",
                    "Are surface radiolytic oxidants (O2, H2O2) transported downward into the ocean?",
                    "Are organic compounds in plumes derived from abiotic hydrothermal activity or biogenic processes?",
                ],
                "harvestable_materials": [
                    "Intact cryovolcanic ocean brine droplets and ice grains",
                    "Sodium chloride (NaCl) and magnesium sulfate (MgSO4) oceanic salts",
                    "Volatile organic molecules, amino acids, and potential biopolymers",
                    "Atmospheric radiolytic oxidants preserving oxidant-reductant disequilibrium",
                ],
                "fundamental_discoveries_enabled": (
                    "First direct physical contact and chemical characterization of an extraterrestrial global ocean, "
                    "providing the definitive test for life in the outer Solar System."
                ),
            },
            "enceladus": {
                "system": "Saturnian System",
                "body": "Enceladus",
                "target_feature": "South Polar 'Tiger Stripe' Plumes & E-Ring Grains",
                "primary_scientific_questions": [
                    "What is the active temperature and extent of hydrothermal serpentinization at the ocean core?",
                    "Are the macromolecular organics detected by Cassini complex biogenic polymers or abiotic refractory tholins?",
                    "Is there ongoing metabolic methanogenesis powered by dissolved H2 and CO2?",
                    "Do hydrothermal vent chimneys on the seafloor mirror alkaline vents where terrestrial life originated?",
                ],
                "harvestable_materials": [
                    "Active hydrothermal silica nanoparticles (SiO2, 2-8 nm diameter)",
                    "Alkaline ocean water droplets (pH 9-11)",
                    "Macromolecular organic compounds (mass > 200 Da)",
                    "Molecular hydrogen (H2), methane (CH4), and carbon monoxide (CO)",
                ],
                "fundamental_discoveries_enabled": (
                    "Unambiguous confirmation of active hydrothermal chemosynthesis and testing whether life "
                    "originates universally at alkaline water-rock interfaces."
                ),
            },
            "miranda_uranus": {
                "system": "Uranian System",
                "body": "Miranda / Titania",
                "target_feature": "Verona Rupes Faults & Ice Giant Volatiles",
                "primary_scientific_questions": [
                    "What was the primordial volatile inventory (CO, N2, noble gases) at 20 AU?",
                    "Did Miranda experience catastrophic tidal disruption and chaotic re-accretion?",
                    "Is there a modern ammonia-water rich liquid mantle layer?",
                ],
                "harvestable_materials": [
                    "Ammonia-water cryomagmatic hydrates",
                    "Carbon monoxide (CO) and carbon dioxide (CO2) clathrates",
                    "Pristine outer solar nebula organic tholins",
                ],
                "fundamental_discoveries_enabled": (
                    "First pristine sample of the ice giant formation zone, rewriting models of giant planet core accretion."
                ),
            },
            "triton_neptune": {
                "system": "Neptunian System",
                "body": "Triton",
                "target_feature": "Cryogenic Nitrogen Geysers & Solid N2/CH4 Frost",
                "primary_scientific_questions": [
                    "What is the bulk chemical composition of a captured Kuiper Belt Object (KBO)?",
                    "Does tidal circularization heating sustain an active liquid ammonia-water ocean today?",
                    "How do hyper-volatile nitrogen/methane cycles operate at 38 Kelvin?",
                ],
                "harvestable_materials": [
                    "Deep-frozen (38 K) solid nitrogen and methane frost",
                    "Dark organic tholins formed by magnetospheric proton bombardment",
                    "Captured subsurface ocean exhalations",
                ],
                "fundamental_discoveries_enabled": (
                    "Direct physical analysis of the primordial Kuiper Belt without having to travel to the Oort cloud, "
                    "preserving hyper-volatiles unheated since 4.56 billion years ago."
                ),
            },
        }

        key = target_name.lower().strip()
        if key not in targets:
            raise KeyError(f"Unknown target '{target_name}'. Available: {list(targets.keys())}")

        return targets[key]


def synthesize_project_argonaut_scientific_value() -> Dict[str, Any]:
    """Generate master scientific valuation dossier for Project ARGONAUT."""
    evaluator = AstrobiologyBiosignatureEvaluator
    lab_comp = LaboratoryVsInSituComparison.compare_analytical_capabilities()

    # Benchmark typical biotic vs abiotic sample scenarios
    biotic_aa = evaluator.evaluate_enantiomeric_excess(d_conc=0.04, l_conc=0.96)
    abiotic_aa = evaluator.evaluate_enantiomeric_excess(d_conc=0.51, l_conc=0.49)

    # Fatty acid series: biogenic even-preference vs abiotic smooth
    biotic_lipids = {14: 0.8, 15: 0.1, 16: 2.5, 17: 0.2, 18: 3.1, 19: 0.15, 20: 1.2, 21: 0.1, 22: 0.9}
    abiotic_lipids = {c: 10.0 * (0.65 ** (c - 12)) for c in range(13, 24)}
    biotic_cpi = evaluator.evaluate_carbon_preference_index(biotic_lipids)
    abiotic_cpi = evaluator.evaluate_carbon_preference_index(abiotic_lipids)

    # Carbon isotope fractionation
    biotic_iso = evaluator.evaluate_carbon_isotopic_fractionation(-32.0, -2.0)
    abiotic_iso = evaluator.evaluate_carbon_isotopic_fractionation(-8.0, -2.0)

    # Methanogenesis energy affinity at Enceladus ocean-floor conditions (273 K, alkaline)
    methano_affinity = evaluator.evaluate_methanogenesis_gibbs_affinity(
        temp_kelvin=273.15,
        p_h2_bar=0.01,
        p_co2_bar=0.001,
        p_ch4_bar=0.0001,
    )

    targets = {
        "ceres": PlanetaryScienceValuationEngine.evaluate_target_science("ceres"),
        "europa": PlanetaryScienceValuationEngine.evaluate_target_science("europa"),
        "enceladus": PlanetaryScienceValuationEngine.evaluate_target_science("enceladus"),
        "triton": PlanetaryScienceValuationEngine.evaluate_target_science("triton_neptune"),
    }

    return {
        "executive_summary": (
            "Project ARGONAUT represents the supreme scientific return paradigm in modern planetary science: "
            "physically retrieving pristine cryogenic extraterrestrial ocean brine and volatile organic samples "
            "and delivering them into a permanent Geosynchronous quarantine laboratory. "
            "This completely surpasses flyby spectroscopy, providing 60x higher mass resolution, sub-nanometer "
            "imaging, and definitive chiral/isotopic biosignature verification."
        ),
        "biosignature_criteria": {
            "homochirality_demo": {"biotic": biotic_aa, "abiotic": abiotic_aa},
            "carbon_preference_index_demo": {"biotic": biotic_cpi, "abiotic": abiotic_cpi},
            "isotopic_fractionation_demo": {"biotic": biotic_iso, "abiotic": abiotic_iso},
            "hydrothermal_energy_affinity": methano_affinity,
        },
        "laboratory_resolution_superiority": lab_comp,
        "planetary_targets_evaluation": targets,
    }
