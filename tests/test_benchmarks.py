"""Formula limits, geometric invariants, and documented approximation behavior."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from qrc_pulsar_bench.constants import GM_sun, T_sun, c
from qrc_pulsar_bench.engine import PostKeplerianSolver, RelationalTripleEngine, CircumbinarySolver, RelationalTimingEngine
from qrc_pulsar_bench.shape_sphere import ShapeSphereReduction, verify_laplace_beltrami_orthogonality
from qrc_pulsar_bench.systems import PSR_J0337_1715, PSR_J0737_3039, KEPLER_16B


def test_solar_mass_parameter_uses_iau_nominal_conversion():
    assert GM_sun == 1.3271244e20
    assert T_sun == pytest.approx(4.9254909483e-6, rel=1e-11)
    assert T_sun*c**3 == pytest.approx(GM_sun)


def test_binary_scaling_and_circular_limits():
    system = replace(PSR_J0737_3039, eccentricity=0)
    base = PostKeplerianSolver.solve(system)
    doubled_mass = PostKeplerianSolver.solve(replace(system, m1_msun=2*system.m1_msun, m2_msun=2*system.m2_msun))
    doubled_period = PostKeplerianSolver.solve(replace(system, Pb_days=2*system.Pb_days))
    assert base['gamma_pred_ms'] == 0
    assert base['Pdot_b_pred'] < 0
    assert doubled_mass['om_dot_pred_deg_yr']/base['om_dot_pred_deg_yr'] == pytest.approx(2**(2/3))
    assert doubled_mass['Pdot_b_pred']/base['Pdot_b_pred'] == pytest.approx(2**(5/3))
    assert doubled_period['Pdot_b_pred']/base['Pdot_b_pred'] == pytest.approx(2**(-5/3))
    assert doubled_mass['r_pred_us']/base['r_pred_us'] == pytest.approx(2)


def test_binary_exchange_symmetry_and_reference_separation():
    base = PostKeplerianSolver.solve(PSR_J0737_3039)
    swapped = PostKeplerianSolver.solve(replace(PSR_J0737_3039,
        m1_msun=PSR_J0737_3039.m2_msun, m2_msun=PSR_J0737_3039.m1_msun))
    for key in ['Pdot_b_pred', 'om_dot_pred_deg_yr']:
        assert swapped[key] == pytest.approx(base[key])
    changed = PostKeplerianSolver.solve(replace(PSR_J0737_3039, Pdot_b_int_obs=-9e-12))
    assert changed['Pdot_b_pred'] == base['Pdot_b_pred']
    assert changed['relative_difference_pct']['Pdot_b_int'] != base['relative_difference_pct']['Pdot_b_int']
    assert base['targets']['Pdot_b_obs'] != base['targets']['Pdot_b_int_obs']
    assert 's_pred' not in base


@pytest.mark.parametrize('changes', [
    {'m1_msun': 0}, {'m2_msun': -1}, {'Pb_days': 0}, {'eccentricity': 1},
    {'eccentricity': -0.1}, {'sin_i': 1.1}, {'sin_i': np.nan}, {'Pb_days': np.inf},
])
def test_invalid_binary_inputs_rejected(changes):
    with pytest.raises(ValueError):
        PostKeplerianSolver.solve(replace(PSR_J0737_3039, **changes))


def test_zero_reference_does_not_divide_by_zero():
    result = PostKeplerianSolver.solve(replace(PSR_J0737_3039, Pdot_b_obs=0))
    assert result['relative_difference_pct']['Pdot_b'] is None


def test_triple_hierarchy_scaling_equal_mass_octupole_and_gw_baseline():
    base = RelationalTripleEngine.solve(PSR_J0337_1715, baseline_years=1)
    far = RelationalTripleEngine.solve(replace(PSR_J0337_1715, Pb2_days=2*PSR_J0337_1715.Pb2_days))
    equal = RelationalTripleEngine.solve(replace(PSR_J0337_1715, m2_msun=PSR_J0337_1715.m1_msun))
    longer = RelationalTripleEngine.solve(PSR_J0337_1715, baseline_years=2)
    eccentric = RelationalTripleEngine.solve(replace(PSR_J0337_1715, e1=0.2))
    assert far['A_quad_us']/base['A_quad_us'] == pytest.approx(0.25)
    assert far['A_oct_ns']/base['A_oct_ns'] == pytest.approx(2**(-8/3))
    assert equal['A_oct_ns'] == 0
    assert longer['gw_delay_scale_ns']/base['gw_delay_scale_ns'] == pytest.approx(4)
    assert abs(eccentric['Pdot_b1']) > abs(base['Pdot_b1'])
    assert eccentric['A_quad_us'] == base['A_quad_us']  # Explicit circular tidal approximation.
    assert 'delta_sep_qrc' not in base
    assert base['published_sep']['origin'].startswith('Reference metadata')


def test_nonhierarchical_triple_rejected():
    with pytest.raises(ValueError, match='hierarchical'):
        RelationalTripleEngine.solve(replace(PSR_J0337_1715, Pb2_days=0.5))


def test_circumbinary_inclination_and_supplied_ttv_metadata():
    base = CircumbinarySolver.solve(KEPLER_16B)
    tilted = CircumbinarySolver.solve(replace(KEPLER_16B, mutual_inclination_deg=60))
    polar = CircumbinarySolver.solve(replace(KEPLER_16B, mutual_inclination_deg=90))
    changed = CircumbinarySolver.solve(replace(KEPLER_16B, ttv_amplitude_hours=999))
    assert base['P_nodal_prec_yr_pred'] == pytest.approx(45.2631759, rel=1e-8)
    assert tilted['P_nodal_prec_yr_pred']/base['P_nodal_prec_yr_pred'] == pytest.approx(np.cos(np.radians(0.42))/0.5)
    assert polar['P_nodal_prec_yr_pred'] is None
    assert changed['P_nodal_prec_yr_pred'] == base['P_nodal_prec_yr_pred']
    assert changed['ttv_reference_hours'] == 999


def test_facade_does_not_silently_ignore_missing_data(tmp_path):
    assert len(RelationalTimingEngine.run_all()) == 5
    with pytest.raises(FileNotFoundError):
        RelationalTimingEngine.run_all(tmp_path/'missing.csv')


@pytest.mark.parametrize('sides,expected', [
    ((1, 1, 1), (0, 0, 1)),
    ((1, 1, 2), (1, 0, 0)),
    ((0, 1, 1), (0.5, -np.sqrt(3)/2, 0)),
    ((3, 4, 5), (0.5, -7*np.sqrt(3)/50, 24*np.sqrt(3)/50)),
])
def test_shape_known_geometry(sides, expected):
    result = ShapeSphereReduction.compute_shape_coordinates(*sides)
    assert result == pytest.approx(expected, abs=1e-14)
    assert ShapeSphereReduction.verify_shape_invariance(*result)


def test_shape_against_independent_jacobi_vectors_and_scale_invariance():
    rng = np.random.default_rng(20261030)
    for _ in range(30):
        points = rng.normal(size=(3, 2))
        x1, x2, x3 = points
        sides = [np.linalg.norm(x1-x2), np.linalg.norm(x2-x3), np.linalg.norm(x3-x1)]
        rho = (x3-x1)/np.sqrt(2)
        lam = (2*x2-x1-x3)/np.sqrt(6)
        inertia = rho@rho + lam@lam
        expected = [(rho@rho-lam@lam)/inertia, 2*(rho@lam)/inertia,
                    2*abs(rho[0]*lam[1]-rho[1]*lam[0])/inertia]
        for scale in [1e-200, 1, 1e200]:
            actual = ShapeSphereReduction.compute_shape_coordinates(*(np.asarray(sides)*scale))
            assert actual == pytest.approx(expected, abs=1e-13)
        reflected = ShapeSphereReduction.compute_shape_coordinates(*sides, orientation=-1)
        assert reflected[2] == pytest.approx(-expected[2])


@pytest.mark.parametrize('sides', [(1, 1, 3), (-1, 1, 1), (0, 0, 0), (np.nan, 1, 1), (np.inf, 1, 1)])
def test_shape_rejects_invalid_geometry(sides):
    with pytest.raises(ValueError):
        ShapeSphereReduction.compute_shape_coordinates(*sides)


def test_legendre_orthogonality_is_not_nonlinear_decoupling():
    assert abs(verify_laplace_beltrami_orthogonality(2, 4)) < 1e-14
    assert verify_laplace_beltrami_orthogonality(2, 2) == pytest.approx(2/5)
    nonlinear, _ = quad(lambda x: eval_legendre(2, x)**2*eval_legendre(4, x), -1, 1)
    assert nonlinear == pytest.approx(4/35)
