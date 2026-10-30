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

def test_pint_delay_time_invariance():
    """Ensure delay calculation is strictly invariant with respect to TOA chunking."""
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    toas_chunk1 = np.array([56000.0, 56010.0, 56020.0, 56030.0])
    toas_chunk2 = np.array([56020.0, 56030.0, 56040.0, 56050.0])

    delays1 = component.delay(toas_chunk1)
    delays2 = component.delay(toas_chunk2)

    # Overlapping epochs at MJD 56020.0 and 56030.0
    val_from_c1_20 = delays1[2]
    val_from_c2_20 = delays2[0]
    val_direct_20 = component.delay(np.array([56020.0]))[0]

    val_from_c1_30 = delays1[3]
    val_from_c2_30 = delays2[1]
    val_direct_30 = component.delay(np.array([56030.0]))[0]

    assert np.isclose(val_from_c1_20, val_from_c2_20, atol=1e-11)
    assert np.isclose(val_from_c1_20, val_direct_20, atol=1e-11)
    assert np.isclose(val_from_c1_30, val_from_c2_30, atol=1e-11)
    assert np.isclose(val_from_c1_30, val_direct_30, atol=1e-11)

def test_pint_delay_backward_epochs():
    """Ensure delay component cleanly evaluates epochs prior to PEPOCH."""
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    # PEPOCH is 55956.70; test epochs before PEPOCH
    past_toas = np.array([55940.0, 55950.0, 55956.70])
    delays = component.delay(past_toas)
    assert len(delays) == 3
    assert np.all(np.isfinite(delays))

def test_pint_delay_parameter_sensitivity():
    """Ensure changing system mass or orbital parameters actively changes the delay."""
    comp1 = RelationalTripleDelayComponent(PSR_J0337_1715)
    comp2 = RelationalTripleDelayComponent(PSR_J0337_1715)
    comp2.params["M1"] = 1.60  # Shift pulsar mass from 1.4378 to 1.60 M_sun

    toas = np.array([56010.0, 56020.0])
    delays1 = comp1.delay(toas)
    delays2 = comp2.delay(toas)

    # Delay MUST be different because underlying masses differ
    assert not np.allclose(delays1, delays2)

def test_register_qrc_with_pint():
    success, msg = register_qrc_with_pint()
    # Whether PINT is installed or not, function should return a bool and informative string
    assert isinstance(success, bool)
    assert isinstance(msg, str)
    assert "pint" in msg.lower()

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
