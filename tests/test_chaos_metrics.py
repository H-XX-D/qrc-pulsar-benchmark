"""Tests for chaos parameterization via Moment of Inertia and Power Spectral Slopes."""

import numpy as np
import pytest
from qrc_pulsar_bench.constants import G, M_sun, AU
from qrc_pulsar_bench.chaos_metrics import (
    compute_moment_of_inertia,
    compute_virial_kinetics,
    compute_power_spectral_slope,
    compute_spectral_entropy,
    extract_chaos_fingerprint,
)
from qrc_pulsar_bench.symplectic import SymplecticThreeBodyIntegrator


def test_lagrange_jacobi_identity_analytical_vs_numerical():
    """Verify that analytical 2*(2K + U) identically equals numerical d^2 I / dt^2."""
    # Two-body eccentric Keplerian orbit (sun-jupiter scale)
    m1 = 1.0 * M_sun
    m2 = 0.01 * M_sun
    masses = np.array([m1, m2])
    m_tot = m1 + m2

    # Initial state at periastron: r = 1 AU, e = 0.5
    a = 1.0 * AU
    e = 0.5
    r_peri = a * (1 - e)
    v_peri = np.sqrt(G * m_tot * (1 + e) / r_peri)

    # Center-of-mass positions
    r1 = np.array([-m2 / m_tot * r_peri, 0.0, 0.0])
    r2 = np.array([m1 / m_tot * r_peri, 0.0, 0.0])
    v1 = np.array([0.0, -m2 / m_tot * v_peri, 0.0])
    v2 = np.array([0.0, m1 / m_tot * v_peri, 0.0])

    # Propagate with Velocity Verlet for 500 steps with fine dt
    dt = 200.0  # seconds
    steps = 1000
    positions = np.zeros((steps, 2, 3))
    velocities = np.zeros((steps, 2, 3))

    pos = np.array([r1, r2])
    vel = np.array([v1, v2])

    def calc_acc(p):
        r_rel = p[1] - p[0]
        dist = np.linalg.norm(r_rel)
        f = G * m1 * m2 / dist**3 * r_rel
        return np.array([f / m1, -f / m2])

    acc = calc_acc(pos)
    for step in range(steps):
        positions[step] = pos
        velocities[step] = vel
        # Verlet step
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc = calc_acc(pos)
        vel = vel_half + 0.5 * dt * acc

    I, R = compute_moment_of_inertia(masses, positions)
    K, U, E, I_ddot_analytic, Q_vir = compute_virial_kinetics(masses, positions, velocities)

    # Numerical second derivative using central differences: (I[i+1] - 2*I[i] + I[i-1]) / dt^2
    I_ddot_num = (I[2:] - 2.0 * I[1:-1] + I[:-2]) / (dt**2)
    I_ddot_analytic_mid = I_ddot_analytic[1:-1]

    # Check relative agreement: O(dt^2) truncation error
    diff = np.abs(I_ddot_num - I_ddot_analytic_mid)
    scale = np.max(np.abs(I_ddot_analytic_mid))
    rel_error = np.max(diff) / scale

    # Truncation error for dt=200s over ~year orbit is tiny (< 1e-4)
    assert rel_error < 5e-4, f"Lagrange-Jacobi identity mismatch: rel_error={rel_error:.2e}"


def test_power_spectral_slope_synthetic_noises():
    """Verify power-law slope beta recovery on synthetic white, pink (1/f), and brownian (1/f^2) signals."""
    rng = np.random.default_rng(42)
    N = 8192
    dt = 0.01

    # 1. White noise: beta ~ 0
    white = rng.standard_normal(N)
    beta_w, r2_w, _, _ = compute_power_spectral_slope(white, dt)
    assert abs(beta_w) < 0.35, f"White noise expected beta ~ 0, got {beta_w}"

    # 2. Brownian noise (random walk): beta ~ 2
    brown = np.cumsum(white)
    beta_b, r2_b, _, _ = compute_power_spectral_slope(brown, dt)
    assert 1.6 < beta_b < 2.4, f"Brownian noise expected beta ~ 2, got {beta_b}"

    # 3. Pink noise (1/f): synthesize via frequency domain 1/sqrt(f) filter
    freqs = np.fft.rfftfreq(N, d=dt)
    freqs[0] = freqs[1]  # avoid divide by zero
    f_filter = 1.0 / np.sqrt(freqs)
    f_filter[0] = 0.0
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs)))
    pink_fft = f_filter * phases
    pink = np.fft.irfft(pink_fft, n=N)
    beta_p, r2_p, _, _ = compute_power_spectral_slope(pink, dt)
    assert 0.6 < beta_p < 1.4, f"Pink noise expected beta ~ 1, got {beta_p}"


def test_spectral_entropy_pure_tone_vs_noise():
    """Verify spectral entropy bounds: pure sinusoidal tone H ~ 0, white noise H ~ 1."""
    t = np.linspace(0, 10, 2048, endpoint=False)
    pure_tone = np.sin(2 * np.pi * 5.0 * t)
    _, _, _, psd_tone = compute_power_spectral_slope(pure_tone, dt=t[1] - t[0])
    h_tone = compute_spectral_entropy(psd_tone)
    assert h_tone < 0.25, f"Pure tone should have low spectral entropy, got {h_tone}"

    rng = np.random.default_rng(123)
    white = rng.standard_normal(2048)
    _, _, _, psd_white = compute_power_spectral_slope(white, dt=t[1] - t[0])
    h_white = compute_spectral_entropy(psd_white)
    assert h_white > 0.85, f"White noise should have high spectral entropy, got {h_white}"


def test_quasi_periodic_vs_chaotic_system():
    """Compare chaos fingerprints of a regular circular system vs. an untuned chaotic triple."""
    # 1. Regular circular binary
    m1, m2 = 1.0 * M_sun, 1.0 * M_sun
    r = 1.0 * AU
    # For equal masses m, distance r, radius of orbit is r/2
    # Centripetal: m v^2 / (r/2) = G m^2 / r^2 => v = sqrt(G m / (2r))
    v = np.sqrt(G * m1 / (2.0 * r))
    masses = np.array([m1, m2, 0.001 * M_sun])

    dt = 86400.0  # 1 day
    steps = 500
    times = np.arange(steps) * dt

    pos = np.zeros((steps, 3, 3))
    vel = np.zeros((steps, 3, 3))
    omega = v / (r / 2.0)
    for i, t in enumerate(times):
        # Body 1
        pos[i, 0] = [-0.5 * r * np.cos(omega * t), -0.5 * r * np.sin(omega * t), 0]
        vel[i, 0] = [0.5 * r * omega * np.sin(omega * t), -0.5 * r * omega * np.cos(omega * t), 0]
        # Body 2
        pos[i, 1] = [0.5 * r * np.cos(omega * t), 0.5 * r * np.sin(omega * t), 0]
        vel[i, 1] = [-0.5 * r * omega * np.sin(omega * t), 0.5 * r * omega * np.cos(omega * t), 0]
        # Distant tertiary
        pos[i, 2] = [10.0 * AU, 0, 0]
        vel[i, 2] = [0, 0, 0]

    fp_reg = extract_chaos_fingerprint(times, pos, vel, masses)
    # Regular system has low virial fluctuation and modest entropy
    assert fp_reg.virial_fluctuation_index < 0.15
    assert fp_reg.virial_ratio_mean == pytest.approx(1.0, rel=0.05)

    # 2. Chaotic 3-body encounter: equal mass non-hierarchical
    # Pythagorean / Burrau style starting configuration
    m = 1.0 * M_sun
    masses3 = np.array([m, m, m])
    pos0 = np.array([
        [-1.0 * AU, 0.0, 0.0],
        [1.0 * AU, 0.0, 0.0],
        [0.0, 0.5 * AU, 0.0]
    ])
    vel0 = np.array([
        [0.0, -5000.0, 0.0],
        [0.0, 5000.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    # Propagate with Velocity Verlet
    dt_c = 3600.0 * 6  # 6 hours
    steps_c = 1500
    times_c = np.arange(steps_c) * dt_c
    pos_c = np.zeros((steps_c, 3, 3))
    vel_c = np.zeros((steps_c, 3, 3))

    p_curr = pos0.copy()
    v_curr = vel0.copy()

    def calc_acc3(p):
        a_out = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_ij = p[j] - p[i]
                    d = np.linalg.norm(r_ij)
                    a_out[i] += G * m * r_ij / (d**3 + 1e6)
        return a_out

    a_curr = calc_acc3(p_curr)
    for s in range(steps_c):
        pos_c[s] = p_curr
        vel_c[s] = v_curr
        v_half = v_curr + 0.5 * dt_c * a_curr
        p_curr = p_curr + dt_c * v_half
        a_curr = calc_acc3(p_curr)
        v_curr = v_half + 0.5 * dt_c * a_curr

    fp_chaotic = extract_chaos_fingerprint(times_c, pos_c, vel_c, masses3)
    # Chaotic untuned system departs strongly from virial equilibrium
    assert fp_chaotic.virial_fluctuation_index > 0.3
    # Moment of inertia expands as system flies apart ("shakes itself to pieces")
    assert fp_chaotic.moment_of_inertia_rate > 0.0
    # Virial ratio departs strongly from 1.0 (un-virialized)
    assert abs(fp_chaotic.virial_ratio_mean - 1.0) > 0.1

