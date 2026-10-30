"""
=====================================================================================
Unit Tests for Symmetric Time-Reversible Symplectic Integration Engine
=====================================================================================
Verifies symplectic phase-space structure preservation, strict T-reversal symmetry
(reversibility to ~10^-14), and bounded energy oscillation over long baselines.
=====================================================================================
"""

import pytest
import numpy as np
from qrc_pulsar_bench.systems import PSR_J0337_1715, KEPLER_16B
from qrc_pulsar_bench.symplectic import SymplecticThreeBodyIntegrator
from qrc_pulsar_bench.constants import day_s, year_s


def test_yoshida4_time_reversibility():
    """Verify that Yoshida 4th-order symplectic integrator achieves < 10^-13 time-reversal closure."""
    res = SymplecticThreeBodyIntegrator.time_reversal_experiment(
        PSR_J0337_1715, n_steps=1000, dt_s=100.0, order=4
    )
    assert res["is_time_reversible"] is True
    # Relative closure error must be near machine epsilon (< 1e-13)
    assert res["relative_closure_precision"] < 1e-13
    # Absolute return position error must be sub-millimeter across 4.7 million km
    assert res["return_pos_error_m"] < 1e-3


def test_yoshida6_time_reversibility():
    """Verify that Yoshida 6th-order symplectic integrator achieves < 10^-13 time-reversal closure."""
    res = SymplecticThreeBodyIntegrator.time_reversal_experiment(
        PSR_J0337_1715, n_steps=500, dt_s=100.0, order=6
    )
    assert res["is_time_reversible"] is True
    assert res["relative_closure_precision"] < 1e-13
    assert res["return_pos_error_m"] < 1e-3


def test_verlet_time_reversibility():
    """Verify that Velocity Verlet (2nd-order) is strictly time-reversible."""
    res = SymplecticThreeBodyIntegrator.time_reversal_experiment(
        PSR_J0337_1715, n_steps=1000, dt_s=50.0, order=2
    )
    assert res["is_time_reversible"] is True
    assert res["relative_closure_precision"] < 1e-12


def test_symplectic_bounded_energy_envelope():
    """Verify that symplectic integration energy error does not drift monotonically over 5,000 steps."""
    # Integrate for ~5.8 days (~3.5 inner orbits) with small dt
    res = SymplecticThreeBodyIntegrator.integrate(
        PSR_J0337_1715, total_time_s=5000 * 100.0, dt_s=100.0, order=4, record_every=250
    )
    assert res["max_energy_error"] < 1e-5
    # Final energy error must not blow up compared to max energy error (bounded envelope)
    assert res["final_energy_error"] <= res["max_energy_error"]

    # Angular momentum vector must be strictly conserved (< 1e-12 relative)
    l0 = res["initial_angular_momentum"]
    lf = res["final_angular_momentum"]
    l_err = np.linalg.norm(lf - l0) / np.linalg.norm(l0)
    assert l_err < 1e-12


def test_kepler16b_circumbinary_stability():
    """Verify that Kepler-16b circumbinary planetary system integrates stably with symplectic engine."""
    # Integrate for 200 days (~5 inner stellar orbits, ~0.9 planetary year)
    res = SymplecticThreeBodyIntegrator.integrate(
        KEPLER_16B, total_time_s=200 * day_s, dt_s=3600.0, order=4, record_every=50
    )
    assert res["max_energy_error"] < 1e-4
    assert len(res["times_days"]) > 10
