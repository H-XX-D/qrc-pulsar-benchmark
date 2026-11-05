"""Published-file regressions; these test inspection, not predictive performance."""

import hashlib
import os
from pathlib import Path

import pytest

from qrc_pulsar_bench.datasets import ZENODO_MD5
from qrc_pulsar_bench.residuals import GenericPulsarDataEvaluator, ZenodoResidualEvaluator

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.real_data


@pytest.fixture(scope='module')
def zenodo_path():
    explicit = os.environ.get('QRC_ZENODO_CSV')
    candidates = [Path(explicit)] if explicit else [
        ROOT/'psr_j0337_zenodo_residuals.csv', ROOT.parent/'psr_j0337_zenodo_residuals.csv']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        if explicit or os.environ.get('QRC_REQUIRE_ZENODO') == '1':
            pytest.fail('Zenodo regression data required but missing; set QRC_ZENODO_CSV.')
        pytest.skip('Optional Zenodo CSV absent; set QRC_ZENODO_CSV to enable the 27,110-row regression.')
    return path


def test_zenodo_statistics_and_covariance(zenodo_path):
    assert hashlib.md5(zenodo_path.read_bytes()).hexdigest() == ZENODO_MD5
    result = ZenodoResidualEvaluator.evaluate(zenodo_path, include_sep_diagnostic=True)
    assert result['N_total'] == 27110
    assert sum(w['n_obs'] for w in result['windows']) == result['N_total']
    assert result['overall_rms_us'] == pytest.approx(1.54260196, rel=1e-7)
    # 1.5424795253 us at the former unsourced 365.953339305 Hz, rescaled to
    # Archibald et al. (2018) f = 365.953363080 Hz.
    assert result['overall_std_us'] == pytest.approx(1.5424795253*365.953339305/365.953363080, rel=1e-9)
    assert result['drift_slope_us_yr'] == pytest.approx(0.0134999903, rel=1e-7)
    diagnostic = result['sep_diagnostic']
    assert diagnostic['nuisance_rank'] == 39
    assert diagnostic['conditional_sigma'] == pytest.approx(7.300126207e-13, rel=1e-9)
    assert diagnostic['marginal_sigma'] == pytest.approx(1.91002275e-7, rel=1e-6)
    assert diagnostic['marginal_sigma'] > 200000*diagnostic['conditional_sigma']


@pytest.mark.parametrize('name,count,std,slope', [
    ('J1614-2230', 12775, 2.992033772108, 0.02647091231),
    ('J0740+6620', 3328, 3.827041742026, 0.04208275789),
])
def test_committed_nanograv_files(name, count, std, slope):
    path = ROOT/'data'/'nanograv'/f'{name}_NANOGrav_12yv4.all.res'
    assert path.is_file(), 'Committed NANOGrav fixture missing'
    result = GenericPulsarDataEvaluator.evaluate(path)
    assert result['N_total'] == count
    assert sum(w['n_obs'] for w in result['windows']) == count
    assert result['overall_std_us'] == pytest.approx(std, rel=1e-10)
    assert result['overall_rms_us']**2 == pytest.approx(std**2 + result['overall_mean_us']**2)
    assert result['drift_slope_us_yr'] == pytest.approx(slope, rel=1e-8)


@pytest.mark.parametrize('backend,count',[('arecibo',5355),('gbt',18487),('wsrt',3268)])
def test_archibald_backend_counts_match_published_table_and_tim_flags(zenodo_path,backend,count):
    result=ZenodoResidualEvaluator.evaluate(zenodo_path,backend=backend)
    assert result['N_total']==count
    assert result['backend']==backend
