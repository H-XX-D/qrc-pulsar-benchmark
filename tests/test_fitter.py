"""
=====================================================================================
Unit & Integration Tests for Standalone Fitter and PINT Integration
=====================================================================================
"""

import os
import pytest
import numpy as np

from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.fitter import RelationalTimingFitter
from qrc_pulsar_bench.pint_model import RelationalTripleDelayComponent, register_qrc_with_pint

def test_pint_delay_component():
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    toas_mjd = np.array([56000.0, 56001.0, 56002.0, 56005.0])
    delays = component.delay(toas_mjd)
    assert len(delays) == len(toas_mjd)
    assert np.all(np.isfinite(delays))
    # Delays should be in seconds, within outer Roemer semi-amplitude bounds (+/- 100 s)
    assert np.all(np.abs(delays) < 100.0)

def test_register_qrc_with_pint():
    success, msg = register_qrc_with_pint()
    # Whether PINT is installed or not, function should return a bool and informative string
    assert isinstance(success, bool)
    assert isinstance(msg, str)

def test_relational_fitter_synthetic():
    np.random.seed(42)
    mjd = np.linspace(56000.0, 56050.0, 20)
    unc = np.full_like(mjd, 1.0) # 1 us uncertainty
    # Synthetic residuals with 0.1 us noise
    res = 0.5 + 0.05 * (mjd - 56000.0) / 365.25 + 0.1 * np.random.randn(len(mjd))

    fit = RelationalTimingFitter.fit(mjd, res, unc, PSR_J0337_1715)
    assert fit["num_toas"] == 20
    assert "phase_offset_us" in fit["params"]
    assert "delta_tasc1_s" in fit["params"]
    assert "delta_tasc2_s" in fit["params"]
    assert fit["residual_after_perturbation_rms_us"] < fit["input_postfit_rms_us"]
    assert fit["chi2_dof"] > 0.0

@pytest.mark.real_data
def test_relational_fitter_real_zenodo():
    paths = ["psr_j0337_zenodo_residuals.csv", "../psr_j0337_zenodo_residuals.csv"]
    csv_path = next((p for p in paths if os.path.exists(p)), None)
    if not csv_path:
        pytest.skip("Zenodo dataset not found locally")

    import pandas as pd
    df = pd.read_csv(csv_path, comment='#', header=None)
    mjd = df[0].values[:500]  # Subsample for fast regression testing
    res_phase = df[1].values[:500]
    unc_phase = df[2].values[:500]

    P0_us = (1.0 / PSR_J0337_1715.f0_hz) * 1e6
    res_us = res_phase * P0_us
    unc_us = unc_phase * P0_us

    fit = RelationalTimingFitter.fit(mjd, res_us, unc_us, PSR_J0337_1715)
    assert fit["num_toas"] == 500
    assert fit["residual_after_perturbation_rms_us"] < 2.0
    assert abs(fit["params"]["delta_tasc1_s"]) < 1.0  # Under 1 second
    assert abs(fit["params"]["delta_tasc2_s"]) < 1.0  # Under 1 second
