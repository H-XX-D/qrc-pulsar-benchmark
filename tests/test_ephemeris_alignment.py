"""The relational delay must reproduce the published J0337+1715 orbits.

Oracle: Keplerian Roemer delays built from Ransom et al. (2014) Table 1
(projected axes, periods, Laplace-Lagrange eccentricity components and
ascending-node times). Over one outer orbit the three-body perturbation
departs from these fixed ellipses by well under 0.1 s, while ignoring the
periapsis longitude or the eccentric-anomaly timing shifts the delay by
seconds.
"""

import dataclasses

import numpy as np
import pytest

from qrc_pulsar_bench.fitter import RelationalTimingFitter
from qrc_pulsar_bench.pint_model import RelationalTripleDelayComponent
from qrc_pulsar_bench.systems import PSR_J0337_1715

RANSOM = {
    "outer": dict(x=74.6727101, pb=327.257541, eps1=3.5186279e-2, eps2=-3.462131e-3, tasc=56233.935815),
    "inner": dict(x=1.21752844, pb=1.629401788, eps1=6.8567e-4, eps2=-9.171e-5, tasc=55920.407717436),
}


def _kepler_delay(t, x, pb, eps1, eps2, tasc):
    e, w = np.hypot(eps1, eps2), np.arctan2(eps1, eps2)
    f_asc = -w
    E_asc = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f_asc / 2))
    t0 = tasc - (E_asc - e * np.sin(E_asc)) / (2 * np.pi) * pb
    M = 2 * np.pi * (t - t0) / pb
    E = M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return x * ((np.cos(E) - e) * np.sin(w) + np.sqrt(1 - e * e) * np.sin(E) * np.cos(w))


# Ransom et al. (2014) Table 1 masses, periods, eccentricities; inclinations
# from the projected axes: tan i = (a sin i) / (a cos i).
RANSOM_SYSTEM = dataclasses.replace(
    PSR_J0337_1715, m1_msun=1.4378, m2_msun=0.19751, m3_msun=0.4101,
    Pb1_days=1.629401788, Pb2_days=327.257541, e1=6.9178e-4, e2=3.53561955e-2,
    cos_i1=float(np.cos(np.arctan2(1.21752844, 1.4900))),
    cos_i2=float(np.cos(np.arctan2(74.6727101, 91.42))))


def _ransom_delay(t):
    return _kepler_delay(t, **RANSOM["outer"]) + _kepler_delay(t, **RANSOM["inner"])


def test_default_component_matches_ransom_orbits_over_first_outer_orbit():
    t = np.linspace(55920.0, 55920.0 + 327.26, 4000)
    delay = RelationalTripleDelayComponent(RANSOM_SYSTEM).delay(t)
    assert np.max(np.abs(delay - _ransom_delay(t))) < 0.1


def test_component_matches_ransom_before_reference_epoch():
    # Backward integration must use the same geometry.
    t = np.linspace(55920.0 - 60.0, 55920.0, 1500)
    delay = RelationalTripleDelayComponent(RANSOM_SYSTEM).delay(t)
    assert np.max(np.abs(delay - _ransom_delay(t))) < 0.1


def test_periapsis_parameters_change_the_delay():
    t = np.linspace(56000.0, 56100.0, 200)
    comp = RelationalTripleDelayComponent()
    base = comp.delay(t)
    comp._set_param("OM2", 95.619493 + 10.0)
    # With the node time held fixed, rotating periastron changes the delay
    # shape at order e * x * d_omega = 0.0354 * 74.67 s * 0.175 rad = 0.46 s.
    change = np.max(np.abs(comp.delay(t) - base))
    assert 0.25 < change < 0.75


def test_fitter_recovers_injected_node_shifts():
    mjd = np.sort(np.random.default_rng(1).uniform(56000.0, 56600.0, 300))
    for d1, d2 in [(0.0, 0.01), (0.001, 0.0)]:
        comp = RelationalTripleDelayComponent(PSR_J0337_1715)
        comp._set_param("TASC1", RelationalTimingFitter.DEFAULT_TASC1_MJD + d1)
        comp._set_param("TASC2", RelationalTimingFitter.DEFAULT_TASC2_MJD + d2)
        observed = comp.delay(mjd) * 1e6
        fit = RelationalTimingFitter.fit_integrated_trajectory(mjd, observed, np.ones(len(mjd)), PSR_J0337_1715)
        assert fit["params"]["delta_tasc1_s"] == pytest.approx(d1 * 86400.0, abs=0.5)
        assert fit["params"]["delta_tasc2_s"] == pytest.approx(d2 * 86400.0, abs=0.5)
