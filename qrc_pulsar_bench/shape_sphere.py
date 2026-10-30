"""
=====================================================================================
Shape Sphere (S²) Geometric Reduction & Multipole Expansion Functions
=====================================================================================
Provides geometric coordinates for 3-body triangle configurations on the unit
2-sphere S² (Montgomery 2002, Dragt 1965) and standard Legendre multipole
integrals.

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
from scipy.special import legendre
from scipy.integrate import quad

class ShapeSphereReduction:
    """Standard Montgomery-Dragt geometric shape map of 3-body configurations onto S²."""

    @staticmethod
    def compute_shape_coordinates(a: float, b: float, c: float) -> tuple[float, float, float]:
        """
        Map three pairwise inter-particle distances (a, b, c) onto the unit 2-sphere S².
        """
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Side lengths must be strictly positive.")

        # Triangle inequality check
        if a + b < c or a + c < b or b + c < a:
            raise ValueError(f"Side lengths ({a}, {b}, {c}) violate the triangle inequality.")

        R_sq = a**2 + b**2 + c**2
        s = (a + b + c) / 2.0
        # Heron formula for area
        area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
        area = np.sqrt(area_sq)

        w1 = (2.0 * c**2 - a**2 - b**2) / R_sq
        w2 = np.sqrt(3.0) * (a**2 - b**2) / R_sq
        w3 = 4.0 * np.sqrt(3.0) * area / R_sq

        return float(w1), float(w2), float(w3)

    @staticmethod
    def verify_shape_invariance(w1: float, w2: float, w3: float, tolerance: float = 1e-12) -> bool:
        """Verify that computed coordinates lie on the unit 2-sphere S²."""
        sphere_sum = w1**2 + w2**2 + w3**2
        return abs(sphere_sum - 1.0) < tolerance

def multipole_potential_expansion(l: int, cos_theta: np.ndarray) -> np.ndarray:
    """Evaluate Legendre multipole mode P_l(cos theta)."""
    P = legendre(l)
    return P(cos_theta)

def verify_laplace_beltrami_orthogonality(l1: int, l2: int) -> float:
    """
    Compute inner product <P_l1, P_l2> = integral_{-1}^1 P_l1(x) P_l2(x) dx
    using adaptive Gauss-Legendre quadrature.
    Note: Orthogonality of Legendre modes does not prevent dynamical multi-frequency
    interaction or secular orbital evolution (e.g., Kozai-Lidov effects) in full N-body systems.
    """
    P1 = legendre(l1)
    P2 = legendre(l2)
    val, _ = quad(lambda x: P1(x) * P2(x), -1.0, 1.0)
    return float(val)
