"""
=====================================================================================
Automated Verification Suite for the Relational 3-Body Dynamical Engine
=====================================================================================
Tests time-dependent orbital evolution, energy conservation, shape-sphere invariance,
gauge elimination (zero COM drift), and trajectory consistency with Cartesian ODEs.
=====================================================================================
"""

import pytest
import numpy as np
from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.constants import G, M_sun, day_s
from qrc_pulsar_bench.dynamics import (
    RelationalState,
    RelationalThreeBodyIntegrator,
    CartesianThreeBodyIntegrator,
    DynamicalBenchmarkSuite,
    orbital_elements_to_vectors
)

def test_relational_state_energy_and_momentum_equivalence():
    m1 = 1.4378 * M_sun
    m2 = 0.1975 * M_sun
    m3 = 0.4104 * M_sun

    r1 = np.array([4.77e9, 0.0, 0.0])
    v1 = np.array([0.0, 2.13e5, 0.0])
    r2 = np.array([0.0, 1.76e11, 0.0])
    v2 = np.array([-3.8e4, 0.0, 0.0])

    state = RelationalState(m1, m2, m3, r1, v1, r2, v2)
    x1, v1_c, x2, v2_c, x3, v3_c = state.to_barycentric_cartesian()

    # 1. Center of mass must be identically zero
    M_tot = m1 + m2 + m3
    com_pos = (m1 * x1 + m2 * x2 + m3 * x3) / M_tot
    com_vel = (m1 * v1_c + m2 * v2_c + m3 * v3_c) / M_tot
    assert np.linalg.norm(com_pos) < 1e-5
    assert np.linalg.norm(com_vel) < 1e-10

    # 2. Kinetic energy in Jacobi coordinates must match Cartesian kinetic energy
    T_cart = 0.5 * (m1 * np.dot(v1_c, v1_c) + m2 * np.dot(v2_c, v2_c) + m3 * np.dot(v3_c, v3_c))
    mu1 = (m1 * m2) / (m1 + m2)
    mu2 = ((m1 + m2) * m3) / M_tot
    T_jac = 0.5 * (mu1 * np.dot(v1, v1) + mu2 * np.dot(v2, v2))
    assert abs(T_cart - T_jac) / T_cart < 1e-14

    # 3. Total angular momentum must match
    J_cart = m1 * np.cross(x1, v1_c) + m2 * np.cross(x2, v2_c) + m3 * np.cross(x3, v3_c)
    J_jac = state.total_angular_momentum()
    assert np.linalg.norm(J_cart - J_jac) / np.linalg.norm(J_cart) < 1e-14

def test_shape_sphere_invariance_throughout_trajectory():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=5.0, num_output_points=200, rtol=1e-10, atol=1e-10
    )
    # Norm on S² must be 1.0 to machine precision along the entire trajectory
    assert res["sphere_invariance_err"] < 1e-14

    # w3 must oscillate smoothly and reach zero at collinear crossings
    assert res["w3"].min() >= 0.0
    assert res["w3"].max() > 0.01

def test_energy_conservation_during_time_evolution():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=10.0, num_output_points=300, rtol=1e-10, atol=1e-10
    )
    # Mechanical energy must be conserved within adaptive ODE tolerance
    assert res["energy_conservation_err"] < 1e-8

def test_center_of_mass_drift_elimination():
    bench = DynamicalBenchmarkSuite.compare(
        PSR_J0337_1715, num_orbits=5.0, num_output_points=200, tolerance=1e-10
    )
    # Relational Jacobi formulation has identically zero COM drift
    assert bench["relational_com_drift_m"] == 0.0

    # Cartesian formulation has small accumulating numerical drift from roundoff
    assert bench["cartesian_com_drift_m"] > 0.0

def test_trajectory_consistency_relational_vs_cartesian():
    num_orbits = 5.0
    rel_res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=num_orbits, num_output_points=250, rtol=1e-10, atol=1e-10
    )
    cart_res = CartesianThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=num_orbits, num_output_points=250, rtol=1e-10, atol=1e-10
    )

    # Relative separation r1 = x2 - x1 in Cartesian vs r1 in Relational
    m1 = PSR_J0337_1715.m1_msun * M_sun
    m2 = PSR_J0337_1715.m2_msun * M_sun
    a1 = (G * (m1 + m2) / (2.0 * np.pi / (PSR_J0337_1715.Pb1_days * day_s))**2)**(1.0 / 3.0)

    for i in range(len(rel_res["t_eval_s"])):
        r1_c = cart_res["sol_y"][6:9, i] - cart_res["sol_y"][0:3, i]
        d_c = np.linalg.norm(r1_c)
        d_rel = rel_res["r1_norm_km"][i] * 1e3
        rel_diff = abs(d_c - d_rel) / a1
        assert rel_diff < 1e-6, f"Step {i}: relative difference {rel_diff} exceeds 1e-6"

def test_pulsar_roemer_delay_periodicity():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=3.0, num_output_points=150
    )
    roemer = res["roemer_delay_us"]
    # Roemer delay must vary smoothly over the orbit with multi-second amplitude
    assert np.ptp(roemer) > 1.0e6 # Peak-to-peak > 1 second
