"""
=====================================================================================
Shape Sphere (S²) Geometric Reduction & Multipole Expansion Functions
=====================================================================================
Provides equal-mass (unweighted) geometric coordinates for triangle shapes and
standard Legendre integrals. This is not a mass-weighted dynamical reduction.
See Montgomery, https://arxiv.org/abs/1402.0841.

The Montgomery-Dragt shape coordinates map three pairwise distances (a, b, c) to:
    w1 = (2*c^2 - a^2 - b^2) / R^2
    w2 = sqrt(3) * (a^2 - b^2) / R^2
    w3 = 4 * sqrt(3) * Area / R^2
where R^2 = a^2 + b^2 + c^2.
Collinear configurations (Area = 0) lie on the equator (w3 = 0), and equilateral
triangles lie at the poles (w3 = +-1), with w1^2 + w2^2 + w3^2 = 1 identically.
=====================================================================================
"""

import numpy as np
from numbers import Integral
from scipy.special import legendre
from scipy.integrate import quad

class ShapeSphereReduction:
    """Unweighted triangle geometry; size and dynamical state are not retained."""

    @staticmethod
    def compute_shape_coordinates(r12: float, r23: float, r31: float, *, orientation=1) -> tuple[float, float, float]:
        """
        Map side lengths (a=r12, b=r23, c=r31) to shape coordinates.

        Distances alone do not determine planar orientation. The default chooses
        the upper hemisphere; orientation=-1 chooses the reflected shape.
        Collinear shapes and binary collisions are allowed; total collision is not.
        """
        sides = np.asarray([r12, r23, r31], dtype=float)
        if not np.all(np.isfinite(sides)) or np.any(sides < 0) or sides.max() == 0:
            raise ValueError("Side lengths must be finite, nonnegative, and not all zero.")
        if isinstance(orientation, bool) or orientation not in (-1, 1):
            raise ValueError("orientation must be +1 or -1.")
        # Normalize first to avoid squaring astronomical or microscopic scales.
        a, b, c = sides / sides.max()
        x, y, z = sorted((a, b, c), reverse=True)
        if x > y + z + 8*np.finfo(float).eps:
            raise ValueError("Side lengths violate the triangle inequality.")
        R_sq = a*a + b*b + c*c
        # Stable Heron form (16 * area^2), including degenerate boundaries.
        area16 = (x+(y+z)) * max(0.0, z-(x-y)) * (z+(x-y)) * (x+(y-z))
        area = np.sqrt(area16) / 4

        w1 = (2.0 * c**2 - a**2 - b**2) / R_sq
        w2 = np.sqrt(3.0) * (a**2 - b**2) / R_sq
        w3 = orientation * 4.0 * np.sqrt(3.0) * area / R_sq

        return float(w1), float(w2), float(w3)

    @staticmethod
    def verify_shape_invariance(w1: float, w2: float, w3: float, tolerance: float = 1e-12) -> bool:
        """Verify that computed coordinates lie on the unit 2-sphere S²."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        values = np.asarray([w1, w2, w3], dtype=float)
        return bool(np.all(np.isfinite(values)) and abs(np.dot(values, values)-1) < tolerance)


def _degree(l):
    if isinstance(l, bool) or not isinstance(l, Integral) or l < 0:
        raise ValueError("Legendre degree must be a nonnegative integer.")

def multipole_potential_expansion(l: int, cos_theta: np.ndarray) -> np.ndarray:
    """Evaluate Legendre multipole mode P_l(cos theta)."""
    _degree(l)
    P = legendre(l)
    return P(cos_theta)

def verify_laplace_beltrami_orthogonality(l1: int, l2: int) -> float:
    """
    Compute inner product <P_l1, P_l2> = integral_{-1}^1 P_l1(x) P_l2(x) dx
    using SciPy adaptive quadrature.
    Note: Orthogonality of Legendre modes does not prevent dynamical multi-frequency
    interaction or secular orbital evolution (e.g., Kozai-Lidov effects) in full N-body systems.
    """
    _degree(l1)
    _degree(l2)
    P1 = legendre(l1)
    P2 = legendre(l2)
    val, _ = quad(lambda x: P1(x) * P2(x), -1.0, 1.0)
    return float(val)
