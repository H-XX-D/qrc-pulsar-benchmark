"""Independent data, statistics, and nuisance-covariance regression checks."""

import json

import numpy as np
import pytest

from qrc_pulsar_bench.residuals import (
    GenericPulsarDataEvaluator, ZenodoResidualEvaluator, inspect_residuals,
    linear_parameter_diagnostic,
)


def test_constant_offset_is_not_zero_rms():
    result = inspect_residuals([0, 1, 2], [100, 100, 100], [1, 1, 1])
    assert result['overall_rms_us'] == 100
    assert result['overall_std_us'] == 0
    assert result['overall_mean_us'] == 100
    assert result['drift_slope_us_yr'] == 0
    assert result['chi2_per_observation'] == 10000
    assert 'global_chi2_dof' not in result


def test_known_rms_std_and_single_assignment_at_window_boundaries():
    result = inspect_residuals([0, 1, 2], [0, 3, 4], [1, 1, 2], num_windows=2)
    assert result['overall_rms_us'] == pytest.approx(5/np.sqrt(3))
    assert result['overall_std_us']**2 + result['overall_mean_us']**2 == pytest.approx(25/3)
    assert [w['n_obs'] for w in result['windows']] == [1, 2]
    assert result['chi2_per_observation'] == pytest.approx(13/3)


def test_empty_windows_are_explicit_and_json_safe():
    result = inspect_residuals([0, 10], [1, 2], [1, 1], num_windows=5)
    assert [w['n_obs'] for w in result['windows']] == [1, 0, 0, 0, 1]
    assert result['windows'][2]['rms_us'] is None
    json.dumps(result, allow_nan=False)


def test_slope_is_unweighted_and_order_independent():
    years = np.asarray([2, 0, 3, 1])
    result = inspect_residuals(58000+365.25*years, 10+2*years, [100, 1, 2, 3])
    assert result['drift_slope_us_yr'] == pytest.approx(2)


@pytest.mark.parametrize('times,residuals,uncertainties,windows', [
    ([0], [1], [1], 1), ([0, 0], [1, 2], [1, 1], 1),
    ([0, 1], [1, np.nan], [1, 1], 1), ([0, np.inf], [1, 2], [1, 1], 1),
    ([0, 1], [1, 2], [1, 0], 1), ([0, 1], [1, 2], [-1, 1], 1),
    ([0, 1], [1, 2], [1, 1], 0), ([0, 1], [1, 2], [1, 1], -1),
    ([0, 1], [1, 2], [1, 1], 1.5), ([0, 1], [1, 2], [1, 1], True),
])
def test_bad_statistics_inputs_fail_clearly(times, residuals, uncertainties, windows):
    with pytest.raises(ValueError):
        inspect_residuals(times, residuals, uncertainties, windows)


def test_ambiguous_extra_columns_require_explicit_format(tmp_path):
    path = tmp_path/'custom.csv'
    path.write_text('58000,200,1,2\n58001,300,1,2\n')
    with pytest.raises(ValueError, match='Ambiguous'):
        GenericPulsarDataEvaluator.evaluate(path)
    result = GenericPulsarDataEvaluator.evaluate(path, format='standard')
    assert result['overall_mean_us'] == 250
    assert result['overall_rms_us'] == pytest.approx(np.sqrt(65000))


def test_nanograv_header_drives_schema_even_for_low_frequency(tmp_path):
    path = tmp_path/'pulsar.res'
    path.write_text('# MJD frequency(MHz) residual(us) uncertainty(us) orb_phase(turns) flag\n'
                    '58000 40 3 1 0 A\n58001 40 4 2 0 B\n')
    result = GenericPulsarDataEvaluator.evaluate(path)
    assert result['input_format'] == 'nanograv'
    assert result['overall_rms_us'] == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError, match='specifies us'):
        GenericPulsarDataEvaluator.evaluate(path, unit='s')


@pytest.mark.parametrize('unit,factor', [('us', 1), ('ns', 1000), ('s', 1e-6)])
def test_unit_conversion_preserves_chi_square(tmp_path, unit, factor):
    path = tmp_path/'units.csv'
    np.savetxt(path, [[58000, 3*factor, factor], [58001, 4*factor, 2*factor]], delimiter=',')
    result = GenericPulsarDataEvaluator.evaluate(path, unit=unit)
    assert result['overall_rms_us'] == pytest.approx(np.sqrt(12.5))
    assert result['chi2_per_observation'] == pytest.approx(6.5)


def test_invalid_file_schema_units_and_missing_file(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('filename 1400 58000 1 observatory\nfilename 1400 58001 1 observatory\n')
    with pytest.raises(ValueError, match='numeric'):
        GenericPulsarDataEvaluator.evaluate(path, format='standard')
    with pytest.raises(ValueError, match='unit'):
        GenericPulsarDataEvaluator.evaluate(path, unit='minutes')
    with pytest.raises(FileNotFoundError):
        GenericPulsarDataEvaluator.evaluate(tmp_path/'missing')


def test_covariance_recovers_injected_increment_with_correlated_nuisance():
    # Explicitly correlated model columns: one-parameter uncertainty is too small.
    t = np.linspace(-1, 1, 101)
    nuisance = np.column_stack([np.ones_like(t), t])
    derivative = 100*t + np.cos(4*t)
    sigma = np.linspace(0.5, 2, len(t))
    residual = -(0.25*derivative + nuisance @ np.asarray([1.2, -3.4]))
    result = linear_parameter_diagnostic(residual, sigma, derivative, nuisance)
    design = np.column_stack([derivative, nuisance]) / sigma[:, None]
    expected_sigma = np.sqrt(np.linalg.inv(design.T@design)[0, 0])
    assert result['increment_about_reference_fit'] == pytest.approx(0.25)
    assert result['marginal_sigma'] == pytest.approx(expected_sigma, rel=1e-10)
    assert result['marginal_sigma'] > 50*result['conditional_sigma']
    # Nuisance units and redundant columns must not change the physical result.
    redundant = np.column_stack([nuisance[:, 0]*1e20, nuisance[:, 1]*1e-20, nuisance[:, 1], np.zeros_like(t)])
    second = linear_parameter_diagnostic(residual, sigma, derivative, redundant)
    assert second['marginal_sigma'] == pytest.approx(result['marginal_sigma'])
    assert second['nuisance_rank'] == 2


def test_unidentifiable_parameter_does_not_report_false_precision():
    result = linear_parameter_diagnostic([1, 2, 3], [1, 1, 1], [1, 2, 3], [[1], [2], [3]])
    assert not result['identifiable']
    assert result['marginal_sigma'] is None
    assert result['increment_about_reference_fit'] is None
    json.dumps(result, allow_nan=False)


def test_zenodo_requires_annotated_phase_schema_and_does_not_claim_absolute_delta(tmp_path):
    path = tmp_path/'phase.csv'
    path.write_text('# MJD (approx), residual, uncertainty (phase), phase, delta\n'
                    '58000,0.001,0.001,1,1\n58001,-0.001,0.001,1,-1\n58002,0,0.001,1,0\n')
    result = ZenodoResidualEvaluator.evaluate(path, include_sep_diagnostic=True)
    assert 'delta_obs' not in result and 'pull_qrc' not in result
    assert result['sep_diagnostic']['nuisance_rank'] == 1
    assert result['published_sep']['absolute_bound_95'] == 2.6e-6
    path.write_text('58000,1,1\n58001,2,1\n')
    with pytest.raises(ValueError, match='annotated'):
        ZenodoResidualEvaluator.evaluate(path)


def test_arecibo_selection_uses_labels_not_column_numbers(tmp_path):
    path=tmp_path/'named.csv'
    path.write_text('# MJD (approx), residual, uncertainty (phase), j_GBT_1400, delta, j_WSRT_1400\n'
                    '58000,0.001,0.001,0,1,0\n58001,100,0.001,1,1,0\n'
                    '58002,100,0.001,0,1,1\n58003,-0.001,0.001,0,1,0\n')
    result=ZenodoResidualEvaluator.evaluate(path,backend='arecibo')
    assert result['N_total']==2
    assert result['overall_mean_us']==pytest.approx(0.)
    assert result['overall_rms_us']<3
    path.write_text(path.read_text().replace('0,1,1','0,1,2'))
    with pytest.raises(ValueError,match='binary backend'):
        ZenodoResidualEvaluator.evaluate(path,backend='arecibo')
