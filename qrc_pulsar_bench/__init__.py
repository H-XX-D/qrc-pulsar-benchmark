"""
=====================================================================================
qrc-pulsar-bench: Relativistic Binary Post-Keplerian & 3-Body Dynamics Toolkit
=====================================================================================
Features:
- Time-dependent 3-body orbital evolution in gauge-free Jacobi relational coordinates.
- Continuous Montgomery-Dragt shape sphere (S²) trajectory tracking.
- Side-by-side benchmark comparison against 18-variable Cartesian ODE integrators.
- Exact post-Keplerian relativistic solvers (periastron advance, Einstein delay,
  Shapiro delay, Peters-Mathews GW orbital decay).
- Benchmark reference systems from peer-reviewed literature.
=====================================================================================
"""

__version__ = "1.1.0"
__author__ = "Todd Hendrixx"
__license__ = "MIT"

from .constants import G, c, M_sun, AU, day_s, year_s
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
from .engine import (
    RelationalTimingEngine,
    PostKeplerianSolver,
    RelationalTripleEngine,
    CircumbinarySolver,
    ZenodoWalkForwardEvaluator,
    GenericPulsarDataEvaluator
)
from .dynamics import (
    RelationalState,
    RelationalThreeBodyIntegrator,
    CartesianThreeBodyIntegrator,
    DynamicalBenchmarkSuite,
    orbital_elements_to_vectors
)
