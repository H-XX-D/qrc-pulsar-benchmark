import numpy as np
import pytest
from qrc_pulsar_bench.observed_timing import evolve, observables
from qrc_pulsar_bench.timing_fit import (
    predict,
    calibrate,
    TimingDesign,
    PARAMETERS,
    REFERENCE_FREQUENCY,
)


def test_noiseless_continuous_phase_recovers_independent_orbital_perturbations():
    """A synthetic phase-recovery test, not a real observation or accuracy claim."""
    rng = np.random.default_rng(30)
    t = np.sort(rng.uniform(86400, 30 * 86400, 150))
    data = dict(
        tcb_s=t,
        uncertainty_us=np.ones(len(t)),
        frequency_mhz=rng.uniform(700, 1800, len(t)),
        observatory=np.resize(np.array(["ao", "gbt", "wsrt"]), len(t)),
        pulse_number=np.zeros(len(t)),
    )
    truth = np.zeros(14)
    truth[PARAMETERS.index("x1")] = 12
    truth[PARAMETERS.index("p1")] = 20
    prediction, _, _ = predict(data, truth)
    data["pulse_number"] = (t - prediction["delay_s"] - 3.0) * REFERENCE_FREQUENCY
    fit = calibrate(data, max_iterations=3, active_parameters=["x1", "p1"])
    np.testing.assert_allclose(fit["offsets"][[0, 1]], truth[[0, 1]], atol=0.08, rtol=0)
    assert np.sqrt(np.mean(fit["residual_s"] ** 2)) < 2e-9


def test_design_uses_frozen_training_reference_and_center():
    data = dict(
        tcb_s=np.array([10.0, 20.0, 30.0]),
        pulse_number=np.array([100, 200, 300]),
        frequency_mhz=np.array([1400.0, 1200.0, 800.0]),
        observatory=np.array(["ao", "gbt", "wsrt"]),
    )
    design = TimingDesign.from_training(data, dm_order=1)
    prediction = dict(proper_seconds=data["tcb_s"] - 1, delay_s=np.ones(3))
    whole, names = design.matrix(data, prediction)
    idx = np.array([2, 0])
    subset = {k: v[idx] for k, v in data.items()}
    subpred = {k: v[idx] for k, v in prediction.items()}
    matrix, _ = design.matrix(subset, subpred)
    np.testing.assert_array_equal(matrix, whole[idx])
    np.testing.assert_array_equal(
        design.target(subset, subpred), design.target(data, prediction)[idx]
    )


def test_no_silent_time_extrapolation():
    with pytest.raises(ValueError, match="nonempty"):
        evolve([])
    with pytest.raises(ValueError, match="reference epoch"):
        evolve([-1.0])
    sol, state, _, _ = evolve([1000.0])
    with pytest.raises(ValueError, match="outside"):
        observables(sol, state, [2000.0])
