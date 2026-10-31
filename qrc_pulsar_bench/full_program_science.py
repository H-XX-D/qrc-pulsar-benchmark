"""
=====================================================================================
Full-Program Scientific Value & Master Astrobiological Valuation Engine
=====================================================================================
Synthesizes the complete, holistic scientific return of Project ARGONAUT across
its three coupled scientific pillars:
  Pillar 1: Physical Sample Return to Earth GEO Quarantine (5.50 kg pristine ice/salts)
  Pillar 2: Decadal In-Situ Surface Geophysical Observatory Mesh (5 stations, 19 sensors)
  Pillar 3: Cross-Solar-System Cosmochemical & Isotopic Gradient (2.77 AU to 30.07 AU)

Four Grand Scientific Breakthroughs Solved:
  1. The "Second Genesis" of Life (definitive discrimination between abiotic & biotic)
  2. Protoplanetary Disk Volatile Migration & Giant Planet Evolution (Grand Tack / Nice)
  3. Comparative Oceanography of Extraterrestrial Subsurface Oceans
  4. Decadal Tidal Geophysics & Cryotectonic Dynamics
=====================================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FullProgramScienceConstants:
    """Core parameters of the unified Project ARGONAUT science campaign."""
    TOTAL_PLANETARY_SYSTEMS_SAMPLED = 5
    TOTAL_SAMPLE_MASS_RETURNED_KG = 5.50
    TOTAL_SURFACE_STATIONS_DEPLOYED = 5
    TOTAL_SURFACE_INSTRUMENTS_DEPLOYED = 19
    TOTAL_ASTRONOMICAL_BASELINE_AU = 27.301  # Span from Ceres (2.769 AU) to Triton (30.070 AU)
    TOTAL_ASTRONOMICAL_BASELINE_KM = 4.084e9
    PROGRAM_TEMPORAL_BASELINE_YEARS = 32.0


@dataclass
class ScientificBreakthrough:
    """Specification of a primary scientific breakthrough enabled by the full program."""
    breakthrough_id: str
    theme: str
    primary_hypothesis_confronted: str
    mechanistic_evidence_required: str
    argonaut_data_source: str
    resolving_power_advantage: str
    transformative_impact: str


class FullProgramScienceArchitect:
    """
    Evaluates the holistic, cross-disciplinary scientific return of Project ARGONAUT.
    """

    @classmethod
    def get_grand_breakthroughs(cls) -> List[ScientificBreakthrough]:
        return [
            ScientificBreakthrough(
                breakthrough_id="Breakthrough-1",
                theme="Astrobiology & The Second Genesis of Life",
                primary_hypothesis_confronted=(
                    "Is life a universal emergent property of hydrothermal ocean chemistry, "
                    "or an extraordinarily rare terrestrial fluke?"
                ),
                mechanistic_evidence_required=(
                    "Coupled homochiral enantiomeric excess (ee > 0.60), non-random lipid clustering (CPI > 3.0), "
                    "kinetic carbon isotope depletion (delta13C < -20 permil), and alternative nucleic/amino acids."
                ),
                argonaut_data_source="Pristine cryovolcanic plume brine return from Europa (1.25 kg) and Enceladus (1.50 kg).",
                resolving_power_advantage="Laboratory FT-ICR and Orbitrap (m/Delta_m > 1.5M) in GEO resolve isobaric organic multiplets.",
                transformative_impact=(
                    "Unambiguous confirmation of an independent Second Genesis rewrites biology forever, proving "
                    "that life is widespread throughout the universe."
                ),
            ),
            ScientificBreakthrough(
                breakthrough_id="Breakthrough-2",
                theme="Cosmochemistry & Protoplanetary Nebula Migration",
                primary_hypothesis_confronted=(
                    "How did water, carbon, nitrogen, and hyper-volatiles migrate during giant planet accretion "
                    "(testing Grand Tack and Nice Model dynamics)?"
                ),
                mechanistic_evidence_required=(
                    "Continuous empirical transect of D/H, 14N/15N, 12C/13C, and noble gas isotopic ratios "
                    "across 5 solar distances from 2.77 AU to 30.07 AU."
                ),
                argonaut_data_source="Five returned sample canisters spanning Ceres, Europa, Enceladus, Miranda, and Triton.",
                resolving_power_advantage="Multi-collector ICP-MS in GEO provides 0.05 permil precision on volatile clathrates.",
                transformative_impact=(
                    "Replaces theoretical numerical simulations with ground-truth physical chemistry, creating "
                    "the definitive empirical map of early Solar System formation."
                ),
            ),
            ScientificBreakthrough(
                breakthrough_id="Breakthrough-3",
                theme="Comparative Oceanography of Extraterrestrial Oceans",
                primary_hypothesis_confronted=(
                    "What are the thermodynamic, chemical, and energetic states of subsurface oceans across "
                    "relict, hydrothermal, and cryomagmatic regimes?"
                ),
                mechanistic_evidence_required=(
                    "Direct measurement of ocean salinity, sulfate/chloride ratios, dissolved H2/CH4, pH, "
                    "and hydrothermal silica nanoparticles (2-8 nm)."
                ),
                argonaut_data_source="Plume and brine salt returns from Ceres, Europa, and Enceladus.",
                resolving_power_advantage="Intact graded aerogel capture at <=120 deg C preserves chemical species without shock pyrolysis.",
                transformative_impact=(
                    "Establishes extraterrestrial limnology/oceanography as an empirical science, revealing how interior "
                    "oceans evolve and survive under thick ice shells over billions of years."
                ),
            ),
            ScientificBreakthrough(
                breakthrough_id="Breakthrough-4",
                theme="Decadal Tidal Dynamics & Ice Shell Geophysics",
                primary_hypothesis_confronted=(
                    "How thick are outer moon ice shells, and how does tidal dissipation transfer internal heat "
                    "and drive cryovolcanic eruptions?"
                ),
                mechanistic_evidence_required=(
                    "Continuous multi-year surface seismic records, tidal strain measurements, and cryogeyser eruption cadences."
                ),
                argonaut_data_source="5 permanent RTG-powered surface stations operating for 15-30 years on Ceres, Callisto, Titan, Titania, Triton.",
                resolving_power_advantage="Eliminates the 'flyby snapshot' limitation by recording continuous real-time waveforms over decades.",
                transformative_impact=(
                    "Directly maps ice shell thickness, ocean-floor depth, and internal convection patterns, unlocking "
                    "the long-term geophysics of dynamic icy moons."
                ),
            ),
        ]

    @classmethod
    def evaluate_full_program_scientific_value(cls) -> Dict[str, Any]:
        """
        Synthesize the holistic scientific return of the entire Project ARGONAUT program.
        """
        c = FullProgramScienceConstants
        breakthroughs = cls.get_grand_breakthroughs()

        pillars = {
            "pillar_1_physical_sample_return": {
                "description": "Pristine Cryogenic Sample Return to Earth Geosynchronous Orbit Quarantine",
                "total_samples_harvested_kg": c.TOTAL_SAMPLE_MASS_RETURNED_KG,
                "planetary_bodies": ["Ceres (0.75 kg)", "Europa (1.25 kg)", "Enceladus (1.50 kg)", "Miranda (1.00 kg)", "Triton (1.00 kg)"],
                "laboratory_capabilities": "FT-ICR mass spectrometry (m/Delta_m > 1.5M), NanoSIMS, Cryo-TEM, multi-collector ICP-MS",
                "preservation_fidelity": "Encapsulated in titanium canisters under vacuum at 150 K -> 85 K -> 75 K -> 55 K -> 38 K",
            },
            "pillar_2_permanent_surface_observatory_mesh": {
                "description": "Permanent Decadal Surface Geophysical & Seismic Network",
                "total_stations": c.TOTAL_SURFACE_STATIONS_DEPLOYED,
                "total_instruments": c.TOTAL_SURFACE_INSTRUMENTS_DEPLOYED,
                "operational_lifespans": "15 to 30 years powered by radioisotope thermoelectric generators (RTGs)",
                "key_measurements": "Broadband 3-axis seismology, tidal strain, exospheric mass spectrometry, meteorology",
            },
            "pillar_3_interplanetary_cross_sectional_gradient": {
                "description": "Continuous Empirical Cosmochemical & Isotopic Gradient Across the Solar System",
                "spatial_baseline_au": c.TOTAL_ASTRONOMICAL_BASELINE_AU,
                "spatial_baseline_km": c.TOTAL_ASTRONOMICAL_BASELINE_KM,
                "temporal_baseline_years": c.PROGRAM_TEMPORAL_BASELINE_YEARS,
                "gradient_coverage": "Spans Asteroid Belt (2.77 AU) through Neptune / Kuiper Belt boundary (30.07 AU)",
            },
        }

        return {
            "program_name": "Project ARGONAUT: Master Scientific Valuation",
            "executive_summary": (
                "Project ARGONAUT delivers the most comprehensive, transformative scientific return in the history "
                "of planetary exploration. By coupling physical sample return of 5.50 kg of pristine ocean/cryogenic volatiles "
                "with a permanent 5-station, 19-instrument surface seismic network and a 27.3 AU cosmochemical gradient, "
                "it definitively resolves the existence of an independent Second Genesis of life and establishes the "
                "empirical foundation of planetary evolution."
            ),
            "program_pillars": pillars,
            "grand_breakthroughs": [
                {
                    "id": b.breakthrough_id,
                    "theme": b.theme,
                    "hypothesis": b.primary_hypothesis_confronted,
                    "evidence": b.mechanistic_evidence_required,
                    "source": b.argonaut_data_source,
                    "advantage": b.resolving_power_advantage,
                    "impact": b.transformative_impact,
                }
                for b in breakthroughs
            ],
            "scientific_significance_verdict": (
                "TRANSFORMATIVE: Solves the origin of life, early Solar System migration, and subsurface ocean dynamics "
                "using unassailable physical laboratory specimens and continuous decadal surface monitoring."
            ),
        }
