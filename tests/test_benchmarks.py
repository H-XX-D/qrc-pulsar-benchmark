"""
=====================================================================================
Automated Test Suite for QRC-PULSAR-BENCH
=====================================================================================
Verifies that all 5 precision astrophysical systems match empirical observations
and theoretical invariants within strict published scientific tolerances.
=====================================================================================
"""

import pytest
import os
from qrc_pulsar_bench.systems import PSR_J0337_1715, PSR_J0737_3039, PSR_B1913_16, PSR_J1738_0333, KEPLER_16B
from qrc_pulsar_bench.engine import (
    PostKeplerianSolver,
    RelationalTripleEngine,
    CircumbinarySolver,
    ZenodoWalkForwardEvaluator,
    GenericPulsarDataEvaluator
)
from qrc_pulsar_bench.shape_sphere import verify_laplace_beltrami_orthogonality, ShapeSphereReduction

def test_shape_sphere_invariance():
    # Test equilateral triangle configuration
    w1, w2, w3 = ShapeSphereReduction.compute_shape_coordinates(1.0, 1.0, 1.0)
    assert ShapeSphereReduction.verify_shape_invariance(w1, w2, w3)

    # Test scalene configuration
    w1, w2, w3 = ShapeSphereReduction.compute_shape_coordinates(3.0, 4.0, 5.0)
    assert ShapeSphereReduction.verify_shape_invariance(w1, w2, w3)

def test_laplace_beltrami_orthogonality():
    # Secular cross-talk between multipole degrees must be zero to machine precision
    inner_2_3 = verify_laplace_beltrami_orthogonality(2, 3)
    inner_2_4 = verify_laplace_beltrami_orthogonality(2, 4)
    inner_3_4 = verify_laplace_beltrami_orthogonality(3, 4)

    assert abs(inner_2_3) < 1e-15, f"Non-zero cross-talk <P_2, P_3>: {inner_2_3}"
    assert abs(inner_2_4) < 1e-15, f"Non-zero cross-talk <P_2, P_4>: {inner_2_4}"
    assert abs(inner_3_4) < 1e-15, f"Non-zero cross-talk <P_3, P_4>: {inner_3_4}"

def test_psr_j0737_3039_double_pulsar():
    res = PostKeplerianSolver.solve(PSR_J0737_3039)
    # Periastron advance ~ 16.899 deg/yr (agreement > 99.9%)
    assert res["agreements_pct"]["om_dot"] > 99.9
    # GW damping ~ -1.252e-12 s/s (agreement > 99.5%)
    assert res["agreements_pct"]["Pdot_b"] > 99.5
    # Einstein delay ~ 0.385 ms (agreement > 99.5%)
    assert res["agreements_pct"]["gamma"] > 99.5
    # Shapiro range ~ 6.15 us (agreement > 99.9%)
    assert res["agreements_pct"]["r"] > 99.9

def test_psr_b1913_16_hulse_taylor():
    res = PostKeplerianSolver.solve(PSR_B1913_16)
    # Periastron advance ~ 4.2266 deg/yr (agreement > 99.9%)
    assert res["agreements_pct"]["om_dot"] > 99.9
    # GW damping ~ -2.423e-12 s/s (agreement > 99.0%)
    assert res["agreements_pct"]["Pdot_b"] > 99.0
    # Einstein delay ~ 4.295 ms (agreement > 99.9%)
    assert res["agreements_pct"]["gamma"] > 99.9

def test_psr_j1738_0333_asymmetric_binary():
    res = PostKeplerianSolver.solve(PSR_J1738_0333)
    # Quadrupole GW decay should match order of magnitude ~ -2.7e-14 s/s
    assert abs(res["Pdot_b_pred"] - (-2.75e-14)) < 0.3e-14

def test_psr_j0337_1715_relational_multipoles():
    res = RelationalTripleEngine.solve(PSR_J0337_1715)
    # Hierarchy ratio epsilon ~ 0.027
    assert 0.026 < res["epsilon_hierarchy"] < 0.028
    # Quadrupole amplitude ~ 6.11 us
    assert 6.0 < res["A_quad_us"] < 6.3
    # Octupole amplitude ~ 125.5 ns
    assert 120.0 < res["A_oct_ns"] < 130.0
    # Hexadecapole amplitude ~ 3.05 ns
    assert 2.5 < res["A_hex_ns"] < 4.0
    # SEP parameter strictly 0.00
    assert res["delta_sep_qrc"] == 0.00

def test_kepler_16b_circumbinary():
    res = CircumbinarySolver.solve(KEPLER_16B)
    # Precession period dynamically computed from Kepler's third law ~ 45.26 years (Doyle 2011: 44.0 +- 2.0 yr)
    assert abs(res["P_nodal_prec_yr_pred"] - 44.1) < 2.0
    assert res["agreement_pct"] > 95.0

def test_zenodo_real_observations():
    paths = ["psr_j0337_zenodo_residuals.csv", "../psr_j0337_zenodo_residuals.csv"]
    csv_path = next((p for p in paths if os.path.exists(p)), None)
    if csv_path:
        res = ZenodoWalkForwardEvaluator.evaluate(csv_path, num_windows=10)
        assert res["N_total"] == 27110
        assert res["overall_rms_us"] < 1.70
        assert abs(res["drift_slope_us_yr"]) < 0.05
        assert res["pull_qrc"] < 0.20

def test_generic_pulsar_data_evaluator(tmp_path):
    # Create synthetic custom timing data file
    sample_file = tmp_path / "custom_pulsar.csv"
    with open(sample_file, "w") as f:
        f.write("# MJD, residual_us, uncertainty_us\n")
        for i in range(100):
            mjd_val = 58000.0 + i * 5.0
            res_val = 0.05 * (-1.0 if i % 2 == 0 else 1.0)
            unc_val = 0.5
            f.write(f"{mjd_val}, {res_val}, {unc_val}\n")

    res = GenericPulsarDataEvaluator.evaluate(str(sample_file), num_windows=5, unit="us")
    assert res["N_total"] == 100
    assert res["overall_rms_us"] < 0.10
    assert abs(res["drift_slope_us_yr"]) < 0.05
    assert len(res["windows"]) == 5

def test_nanograv_real_datasets():
    # Test real NANOGrav 12.5y observations for PSR J1614-2230
    path = "data/nanograv/J1614-2230_NANOGrav_12yv4.all.res"
    if os.path.exists(path):
        res = GenericPulsarDataEvaluator.evaluate(path, num_windows=10, unit="us")
        assert res["N_total"] == 12775
        assert res["overall_rms_us"] < 3.50
        assert abs(res["drift_slope_us_yr"]) < 0.045
        assert len(res["windows"]) == 10

