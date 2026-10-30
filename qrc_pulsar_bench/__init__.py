"""
=====================================================================================
QRC-PULSAR-BENCH: Open-Source Relational Pulsar Timing & Multi-Body Benchmark Suite
=====================================================================================
A high-precision, parameter-free relativistic timing and n-body benchmarking engine
grounded in Quantum Relational Causality (QRC) and Post-Keplerian General Relativity.

Features:
- Parameter-free relational 3-body reduction on the 2D Shape Sphere (S²).
- Decoupled multipole expansion (ℓ=2 Quadrupole, ℓ=3 Octupole, ℓ=4 Hexadecapole).
- Exact post-Keplerian relativistic solvers (Periastron advance, Einstein delay, Shapiro delay, GW damping).
- Validated on gold-standard empirical systems:
    1. PSR J0337+1715 (Hierarchical Stellar Triple — 27,110 real Zenodo observations)
    2. PSR J0737−3039A/B (The Double Pulsar — 16-year PRX 2021 relativistic benchmark)
    3. PSR B1913+16 (Hulse-Taylor Binary Pulsar — 30-year GW decay test)
    4. PSR J1738+0333 (Asymmetric NS-WD circular binary — dipolar radiation test)
    5. Kepler-16b (Circumbinary 3-body exoplanet — TTV & nodal precession)
=====================================================================================
"""

__version__ = "1.0.0"
__author__ = "H-XX-D / Todd Hendrixx"
__license__ = "MIT"

from .constants import G, c, M_sun, AU, day_s, year_s, Z_H
from .shape_sphere import ShapeSphereReduction, multipole_potential_expansion
from .systems import (
    SYSTEMS,
    PSR_J0337_1715,
    PSR_J0737_3039,
    PSR_B1913_16,
    PSR_J1738_0333,
    KEPLER_16B,
)
from .engine import RelationalTimingEngine, PostKeplerianSolver
