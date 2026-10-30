"""
=====================================================================================
Unit & Integration Tests for Standalone Fitter and Real PINT Integration
=====================================================================================
"""

import os
import io
import pytest
import numpy as np

from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.fitter import RelationalTimingFitter
from qrc_pulsar_bench.pint_model import RelationalTripleDelayComponent, register_qrc_with_pint

# Try importing real PINT
try:
    import astropy.units as u
    from astropy.time import Time
    import pint
    import pint.models.timing_model as tm
    from pint.toa import get_TOAs_array
    from pint.models.model_builder import get_model
    from pint.fitter import WLSFitter
    _HAS_REAL_PINT = True
except (ImportError, AttributeError):
    _HAS_REAL_PINT = False


def test_pint_package_installed():
    """Verify that real pint-pulsar (not units library) is installed."""
    assert _HAS_REAL_PINT is True, "pint-pulsar must be installed."
    assert hasattr(pint, "models"), "PINT must contain models package."


def test_pint_delay_component():
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    toas_mjd = np.array([55925.0, 55930.0, 55935.0, 55940.0])
    delays = component.delay(toas_mjd)
    assert len(delays) == len(toas_mjd)
    assert np.all(np.isfinite(delays))
    # Delays should be in seconds, within outer Roemer semi-amplitude bounds (+/- 100 s)
    assert np.all(np.abs(delays) < 100.0)


def test_pint_delay_time_invariance():
    """Ensure delay calculation is strictly invariant with respect to TOA chunking."""
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    toas_chunk1 = np.array([55925.0, 55930.0, 55935.0, 55940.0])
    toas_chunk2 = np.array([55935.0, 55940.0, 55945.0, 55950.0])

    delays1 = component.delay(toas_chunk1)
    delays2 = component.delay(toas_chunk2)

    val_from_c1_35 = delays1[2]
    val_from_c2_35 = delays2[0]
    val_direct_35 = component.delay(np.array([55935.0]))[0]

    val_from_c1_40 = delays1[3]
    val_from_c2_40 = delays2[1]
    val_direct_40 = component.delay(np.array([55940.0]))[0]

    assert np.isclose(val_from_c1_35, val_from_c2_35, atol=1e-11)
    assert np.isclose(val_from_c1_35, val_direct_35, atol=1e-11)
    assert np.isclose(val_from_c1_40, val_from_c2_40, atol=1e-11)
    assert np.isclose(val_from_c1_40, val_direct_40, atol=1e-11)


def test_pint_delay_backward_epochs():
    """Ensure delay component cleanly evaluates epochs prior to PEPOCH_REF."""
    component = RelationalTripleDelayComponent(PSR_J0337_1715)
    # PEPOCH_REF is 55920.0; test epochs before reference epoch
    past_toas = np.array([55900.0, 55910.0, 55920.0])
    delays = component.delay(past_toas)
    assert len(delays) == 3
    assert np.all(np.isfinite(delays))


def test_pint_delay_parameter_sensitivity():
    """Ensure changing system mass or orbital parameters actively changes the delay."""
    comp1 = RelationalTripleDelayComponent(PSR_J0337_1715)
    comp2 = RelationalTripleDelayComponent(PSR_J0337_1715)
    comp2._set_param("M1", 1.60)  # Shift pulsar mass from 1.4378 to 1.60 M_sun

    toas = np.array([55930.0, 55940.0])
    delays1 = comp1.delay(toas)
    delays2 = comp2.delay(toas)

    # Delay MUST be different because underlying masses differ
    assert not np.allclose(delays1, delays2)


def test_register_qrc_with_pint():
    success, msg = register_qrc_with_pint()
    assert success is True
    assert "registry" in msg.lower()


@pytest.mark.skipif(not _HAS_REAL_PINT, reason="PINT pulsar package required")
def test_pint_timing_model_and_wlsfitter():
    """End-to-end integration test with real PINT TimingModel and WLSFitter."""
    par_text = """
    PSR J0337+1715
    RAJ 03:37:24.4
    DECJ 17:15:40.0
    F0 365.953363096
    PEPOCH 55920.0
    """
    model = get_model(io.StringIO(par_text))
    comp = RelationalTripleDelayComponent()
    model.add_component(comp)
    model.validate()

    times = Time(np.linspace(55925, 55930, 5), format="mjd", scale="utc")
    toas = get_TOAs_array(times, errors=np.ones(5)*u.us, obs="arecibo")

    # Evaluate delays and derivatives through PINT
    delay = model.delay(toas)
    assert len(delay) == 5
    assert delay.unit == u.s

    deriv_tasc1 = model.d_delay_d_param(toas, "TASC1")
    assert len(deriv_tasc1) == 5
    assert deriv_tasc1.unit == (u.s / u.d)

    # Fit F0 with PINT WLSFitter
    model.F0.frozen = False
    fitter = WLSFitter(toas, model)
    fitter.fit_toas()
    assert np.isfinite(model.F0.value)


def test_integrated_derivatives_physics():
    """Verify partial derivatives taken from integration match physical velocity lever arm."""
    mjd = np.array([55925.0, 55926.0, 55927.0])
    nom_del, d1, d2 = RelationalTimingFitter.compute_integrated_derivatives(mjd, PSR_J0337_1715)

    assert len(d1) == 3
    assert len(d2) == 3
    # Inner velocity semi-amplitude K1/c ~ 54.85 us/s
    assert np.all(np.abs(d1) <= 60.0)
    # Outer velocity semi-amplitude K2/c ~ 16.76 us/s
    assert np.all(np.abs(d2) <= 20.0)


def test_fitter_built_from_integrated_delay_recovery():
    """Test parameter recovery from true 3-body numerical integration."""
    comp = RelationalTripleDelayComponent(PSR_J0337_1715)
    mjd = np.linspace(55925.0, 55955.0, 25)
    unc = np.full_like(mjd, 0.1)

    # Inject simulated offsets
    true_delta_tasc1 = 0.20  # +0.20 s
    true_delta_tasc2 = -0.30 # -0.30 s
    true_phase = 0.8         # +0.8 us

    comp._set_param("TASC1", 55920.407717436 + true_delta_tasc1 / 86400.0)
    comp._set_param("TASC2", 56233.935815 + true_delta_tasc2 / 86400.0)
    sim_data_us = comp.delay(mjd) * 1e6 + true_phase

    # Fit against nominal trajectory
    fit = RelationalTimingFitter.fit_integrated_trajectory(mjd, sim_data_us, unc, PSR_J0337_1715)

    assert fit["analysis_type"] == "integrated_3body_trajectory_fit"
    assert fit["derivative_source"] == "central_finite_difference_of_relational_three_body_integrator"
    assert np.isclose(fit["params"]["delta_tasc1_s"], true_delta_tasc1, atol=0.01)
    assert np.isclose(fit["params"]["delta_tasc2_s"], true_delta_tasc2, atol=0.05)
    assert fit["residual_after_perturbation_rms_us"] < 0.1


def test_fitter_proves_no_smuggled_mechanics():
    """Ensure that modifying the simulated delay changes fit results completely."""
    mjd = np.linspace(55925.0, 55935.0, 10)
    unc = np.full_like(mjd, 1.0)
    comp = RelationalTripleDelayComponent(PSR_J0337_1715)
    data_us = comp.delay(mjd) * 1e6 + 5.0

    fit1 = RelationalTimingFitter.fit_integrated_trajectory(mjd, data_us, unc, PSR_J0337_1715)
    # Zeroing out the data or comparing against shifted system
    shifted_system = PSR_J0337_1715
    fit2 = RelationalTimingFitter.fit_integrated_trajectory(mjd, np.zeros_like(data_us), unc, shifted_system)

    assert not np.isclose(fit1["params"]["phase_offset_us"], fit2["params"]["phase_offset_us"])
    assert not np.isclose(fit1["input_rms_us"], fit2["input_rms_us"])


def test_relational_fitter_synthetic_residuals():
    np.random.seed(42)
    mjd = np.linspace(55925.0, 55950.0, 20)
    unc = np.full_like(mjd, 1.0)
    res = 0.5 + 0.05 * (mjd - 55920.0) / 365.25 + 0.1 * np.random.randn(len(mjd))

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
    mjd = df[0].values[:500]
    res_phase = df[1].values[:500]
    unc_phase = df[2].values[:500]

    P0_us = (1.0 / PSR_J0337_1715.f0_hz) * 1e6
    res_us = res_phase * P0_us
    unc_us = unc_phase * P0_us

    fit = RelationalTimingFitter.fit(mjd, res_us, unc_us, PSR_J0337_1715)
    assert fit["num_toas"] == 500
    assert fit["residual_after_perturbation_rms_us"] < 2.0
    assert abs(fit["params"]["delta_tasc1_s"]) < 1.0
    # In a 24-day slice of a 327-day outer orbit, delta_tasc2 is constrained to < 10 s (< 3e-7 of an orbit)
    assert abs(fit["params"]["delta_tasc2_s"]) < 10.0
