import numpy as np
from qrc_pulsar_bench.constants import G, c
from qrc_pulsar_bench.relativity import eih_correction, timing_rhs
from qrc_pulsar_bench.dynamics import RelationalThreeBodyIntegrator


def test_schwarzschild_test_particle_limit():
    mass = 2e30
    r = np.array([1e10, 2e9, 0.0])
    v = np.array([1e4, 1e5, 0.0])
    radius = np.linalg.norm(r)
    correction = eih_correction(
        np.array([np.zeros(3), r]), np.array([np.zeros(3), v]), np.array([mass, 0.0])
    )
    expected = (
        G
        * mass
        / (c * c * radius**3)
        * ((4 * G * mass / radius - v @ v) * r + 4 * (r @ v) * v)
    )
    np.testing.assert_allclose(correction[1], expected, rtol=2e-15, atol=1e-25)


def test_two_body_relative_acceleration():
    masses = np.array([1.4e30, 0.7e30])
    mt = masses.sum()
    eta = np.prod(masses) / mt**2
    r = np.array([1e10, 2e9, -3e9])
    v = np.array([1e4, 1e5, -2e4])
    R = np.linalg.norm(r)
    n = r / R
    vr = n @ v
    positions = np.array([-masses[1] / mt * r, masses[0] / mt * r])
    velocities = np.array([-masses[1] / mt * v, masses[0] / mt * v])
    pn = eih_correction(positions, velocities, masses)
    expected = (
        G
        * mt
        / (c * c * R * R)
        * (
            n
            * (
                (4 + 2 * eta) * G * mt / R
                - (1 + 3 * eta) * (v @ v)
                + 1.5 * eta * vr * vr
            )
            + (4 - 2 * eta) * vr * v
        )
    )
    np.testing.assert_allclose(pn[1] - pn[0], expected, rtol=2e-15, atol=1e-25)


def test_newtonian_rhs_parity_and_com_retention():
    rng = np.random.default_rng(8)
    for _ in range(8):
        masses = np.array([1.4e30, 0.2e30, 0.4e30])
        y = np.zeros(19)
        y[:12] = rng.normal(size=12)
        y[:3] *= 1e10
        y[6:9] *= 1e11
        y[3:6] *= 1e5
        y[9:12] *= 1e4
        result = timing_rhs(0, y, masses, False)
        expected = RelationalThreeBodyIntegrator._jacobi_derivatives(0, y[:12], *masses)
        np.testing.assert_allclose(result[:12], expected, rtol=3e-15, atol=1e-16)
        assert np.all(result[13:] == 0)
        assert np.any(timing_rhs(0, y, masses, True)[16:] != 0)


def test_eih_translation_rotation_and_c_scaling():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3, 3)) * 1e10
    v = rng.normal(size=(3, 3)) * 1e5
    m = np.array([1e30, 2e29, 4e29])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = eih_correction(x, v, m)
    np.testing.assert_allclose(
        eih_correction((x + 1e12) @ q, v @ q, m), a @ q, rtol=2e-13, atol=1e-23
    )
    np.testing.assert_allclose(eih_correction(x, v, m, 2 * c), a / 4, rtol=1e-15)
