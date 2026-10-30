import numpy as np
import pytest
from scipy.integrate import solve_ivp
from qrc_pulsar_bench.dynamics import RelationalState, RelationalThreeBodyIntegrator
from qrc_pulsar_bench.shape_dynamics import (
    integrate_planar_shape,
    integrate_spatial_shape,
)
from qrc_pulsar_bench.observed_timing import PUBLISHED, initial_state


def planar_j0337():
    parameters = dict(PUBLISHED)
    parameters["delta_node_deg"] = 0.0
    parameters["acos2"] = parameters["x2"] * parameters["acos1"] / parameters["x1"]
    return initial_state(parameters=parameters)[0]


def test_nonplanar_input_rejected():
    state, _ = initial_state()
    with pytest.raises(ValueError, match="coplanar"):
        integrate_planar_shape(state, 1.0)


def test_roundtrip_and_coupled_trajectory():
    state = planar_j0337()
    end = 6 * 86400.0
    shape = integrate_planar_shape(state, end, rtol=1e-12, atol=1e-15)
    initial = np.r_[state.r1, state.v1, state.r2, state.v2]
    np.testing.assert_allclose(shape.sol(0)[:12], initial, rtol=1e-9, atol=1e-4)
    reference = solve_ivp(
        RelationalThreeBodyIntegrator._jacobi_derivatives,
        (0, end),
        initial,
        args=(state.m1, state.m2, state.m3),
        method="DOP853",
        rtol=2e-13,
        atol=1e-6,
        dense_output=True,
    )
    times = np.linspace(0, end, 300)
    result = shape.sol(times)[:12]
    truth = reference.sol(times)
    # Full position and velocity reconstruction, not just the sphere identity.
    assert np.max(np.linalg.norm(result[:3] - truth[:3], axis=0)) < 2.0
    assert np.max(np.linalg.norm(result[6:9] - truth[6:9], axis=0)) < 5.0
    assert np.max(np.linalg.norm(result[3:6] - truth[3:6], axis=0)) < 1e-4
    np.testing.assert_allclose(np.sum(shape.shape(times) ** 2, axis=0), 1.0, atol=1e-14)
    # Oriented shape must cross the equator, not fold it with abs(area).
    assert np.min(shape.shape(times)[2]) < 0 < np.max(shape.shape(times)[2])


def test_spatial_rotation_preserves_planar_solution():
    state = planar_j0337()
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = RelationalState(
        state.m1,
        state.m2,
        state.m3,
        *[Q @ v for v in [state.r1, state.v1, state.r2, state.v2]],
    )
    a = integrate_planar_shape(state, 1000.0, rtol=1e-12)
    b = integrate_planar_shape(rotated, 1000.0, rtol=1e-12)
    expected = a.sol(1000.0)[:12].reshape(4, 3) @ Q.T
    np.testing.assert_allclose(
        b.sol(1000.0)[:12].reshape(4, 3), expected, rtol=2e-12, atol=1e-3
    )


def test_spatial_shape_reconstructs_nonplanar_dynamics():
    state, _ = initial_state()
    end = 2 * 86400.0
    result = integrate_spatial_shape(state, end, rtol=1e-12, atol=1e-16)
    initial = np.r_[state.r1, state.v1, state.r2, state.v2]
    np.testing.assert_allclose(result.sol(0)[:12], initial, rtol=1e-9, atol=1e-4)
    reference = solve_ivp(
        RelationalThreeBodyIntegrator._jacobi_derivatives,
        (0, end),
        initial,
        args=(state.m1, state.m2, state.m3),
        method="DOP853",
        rtol=2e-13,
        atol=1e-6,
        dense_output=True,
    )
    times = np.linspace(0, end, 150)
    actual = result.sol(times)[:12]
    expected = reference.sol(times)
    assert np.max(np.linalg.norm(actual[:3] - expected[:3], axis=0)) < 3.0
    assert np.max(np.linalg.norm(actual[6:9] - expected[6:9], axis=0)) < 10.0
    np.testing.assert_allclose(
        np.sum(result.shape(times) ** 2, axis=0), 1.0, atol=1e-14
    )
