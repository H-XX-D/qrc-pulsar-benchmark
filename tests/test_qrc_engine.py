"""Tests for the QRC Dynamical Relational Reservoir Computing Engine."""

import pytest
import numpy as np
from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.qrc_engine import (
    JacobiRelationalReservoir,
    compute_effective_dimension,
    compute_fading_memory_capacity,
    benchmark_qrc_reservoir,
)


def test_jacobi_reservoir_state_generation():
    """Verify that JacobiRelationalReservoir produces finite 18-dim feature matrices."""
    res = JacobiRelationalReservoir(PSR_J0337_1715, enable_1pn=False)
    times, states = res.generate_reservoir_states(duration_days=30.0, samples_per_day=5)

    assert len(times) == 150
    assert states.shape == (150, 18)
    assert np.all(np.isfinite(states))


def test_effective_dimension_bounds():
    """Verify participation ratio / effective dimension calculation."""
    # Low rank: all features proportional
    low_rank = np.outer(np.linspace(1, 10, 50), np.ones(5))
    d_low = compute_effective_dimension(low_rank)
    assert d_low == pytest.approx(1.0, rel=0.1)

    # Higher rank orthogonal features
    high_rank = np.random.default_rng(42).standard_normal((100, 10))
    d_high = compute_effective_dimension(high_rank)
    assert 7.0 <= d_high <= 10.0


def test_fading_memory_capacity_and_prediction():
    """Verify memory capacity evaluation and phase reconstruction."""
    result = benchmark_qrc_reservoir(
        PSR_J0337_1715,
        duration_days=40.0,
        samples_per_day=10,
        max_lag=10
    )

    assert result.num_samples == 400
    assert result.reservoir_dim == 18
    assert result.effective_dimension > 1.5
    # Memory capacity across 10 lags should be substantial
    assert result.total_memory_capacity > 5.0
    # High out-of-sample reconstruction accuracy
    assert result.test_r2 > 0.90
    assert result.test_nrmse < 0.35
