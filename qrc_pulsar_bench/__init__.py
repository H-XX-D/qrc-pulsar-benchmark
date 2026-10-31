"""
=====================================================================================
QRC-PULSAR-BENCH: Open-Source Relational Pulsar Timing & Multi-Body Benchmark Suite
=====================================================================================
A small toolkit for standard post-Keplerian formula evaluation, hierarchical
perturbation scale estimates, Newtonian three-body evolution, and residual inspection.

Features:
- Unweighted geometric triangle shapes on a hemisphere of S².
- Hierarchical tidal scales (ℓ=2,3,4) and a separate Newtonian Jacobi/Cartesian integrator.
- Leading-order post-Keplerian formulas.
- Published system examples and residual data:
    1. PSR J0337+1715 (Hierarchical Stellar Triple — 27,110 real Zenodo observations)
    2. PSR J0737−3039A/B (The Double Pulsar — 16-year PRX 2021 relativistic benchmark)
    3. PSR B1913+16 (Hulse-Taylor Binary Pulsar — 30-year GW decay test)
    4. PSR J1738+0333 (Asymmetric NS-WD circular binary — dipolar radiation test)
    5. Kepler-16b (Circumbinary 3-body exoplanet — TTV & nodal precession)
=====================================================================================
"""

__version__ = "1.1.0"
__author__ = "H-XX-D / Todd Hendrixx"
__license__ = "MIT"

from .constants import G, c, GM_sun, T_sun, M_sun, AU, day_s, year_s, Z_H
from .shape_sphere import ShapeSphereReduction, multipole_potential_expansion
from .systems import (
    SYSTEMS,
    BinaryPulsarSystem,
    HierarchicalTripleSystem,
    CircumbinaryExoplanetSystem,
    PSR_J0337_1715,
    PSR_J0737_3039,
    PSR_B1913_16,
    PSR_J1738_0333,
    KEPLER_16B,
)
from .engine import RelationalTimingEngine, PostKeplerianSolver, RelationalTripleEngine, CircumbinarySolver
from .residuals import GenericPulsarDataEvaluator, ZenodoResidualEvaluator, ZenodoWalkForwardEvaluator

from .dynamics import (RelationalState, RelationalThreeBodyIntegrator,
                       CartesianThreeBodyIntegrator, DynamicalBenchmarkSuite,
                       orbital_elements_to_vectors, compute_eih_accelerations)
from .symplectic import SymplecticThreeBodyIntegrator
from .fitter import RelationalTimingFitter
from .pint_model import RelationalTripleDelayComponent, register_qrc_with_pint
from .chaos_metrics import (
    ChaosFingerprint,
    compute_moment_of_inertia,
    compute_virial_kinetics,
    compute_power_spectral_slope,
    compute_spectral_entropy,
    extract_chaos_fingerprint,
)
from .qrc_engine import (
    QRCBenchmarkResult,
    JacobiRelationalReservoir,
    compute_effective_dimension,
    compute_fading_memory_capacity,
    benchmark_qrc_reservoir,
)
from .fpga_orbit_engine import (
    FPGASpecification,
    HardwareCycleReport,
    PipelinedInvCubeUnit,
    FPGAOrbitEngine,
)
from .station_keeping import (
    EarthMoonConstants,
    StationKeepingPolicy,
    StationKeepingReport,
    EarthMoonCR3BP,
    PeriodicLyapunovOrbit,
    PredictiveStationKeepingEngine,
)
from .capstone_replay import (
    MultiDatasetFlightReplay,
    FlightReplayReport,
    CapstoneFlightReplay,
    CapstoneReplayReport,
)
from .multi_path_optimizer import (
    FlightSafetyState,
    SafetyThresholds,
    MultiPathReport,
    MultiPathTrajectoryOptimizer,
)
from .interbody_transfer import (
    MultiBodySystemConstants,
    PeriodicOrbit,
    TransferPlanReport,
    InterBodyDynamics,
    AutonomousInterBodyPlanner,
)
from .jovian_tour import (
    MoonData,
    JovianSystemConstants,
    JovianRadiationModel,
    JovianMoonFlyby,
    JovianTourReport,
    JovianTourPlanner,
)
from .sample_return import (
    SampleReturnConstants,
    StationaryCollectorLauncher,
    DualModePropulsionSystem,
    AutonomousInterplanetaryGNC,
    EarthOrbitalStationCapture,
    SampleReturnMissionReport,
    plan_jovian_sample_return_mission,
)
