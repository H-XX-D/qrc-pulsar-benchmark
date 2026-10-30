"""Pin formula outputs to independent references.

Each test compares against a value that does not come from this package:
a published GR prediction, an analytic orbit projection, or a quantity
solved by a different route (linear algebra instead of the closed form).
These were added because sampled mutation testing showed that operator and
constant changes in these lines survived the earlier suite.
"""

import dataclasses

import numpy as np
import pytest

from qrc_pulsar_bench.constants import G, M_sun, c, day_s
from qrc_pulsar_bench.dynamics import RelationalState, RelationalThreeBodyIntegrator
from qrc_pulsar_bench.engine import PostKeplerianSolver, RelationalTripleEngine
from qrc_pulsar_bench.residuals import ZenodoResidualEvaluator
from qrc_pulsar_bench.systems import PSR_B1913_16, PSR_J0337_1715, PSR_J0737_3039


# Published GR predictions computed by the observers from their own masses.
def test_j0737_shapiro_range_matches_kramer_gr_prediction():
    # Kramer et al. (2021) Sect. VII: r_GR = 6.1518(11) us.
    r = PostKeplerianSolver.solve(PSR_J0737_3039)["r_pred_us"]
    assert r == pytest.approx(6.1518, abs=0.0011)


def test_j0737_orbital_decay_matches_kramer_gr_prediction():
    # Kramer et al. (2021) eq. (47): Pbdot_GR = -1.247827e-12. The leading-order
    # quadrupole formula omits 1PN corrections of order 1e-5 relative.
    pdot = PostKeplerianSolver.solve(PSR_J0737_3039)["Pdot_b_pred"]
    assert pdot == pytest.approx(-1.247827e-12, rel=2e-4)


def test_b1913_orbital_decay_matches_weisberg_huang_gr_prediction():
    # Weisberg & Huang (2016): Pbdot_GR = -2.40263(5)e-12.
    pdot = PostKeplerianSolver.solve(PSR_B1913_16)["Pdot_b_pred"]
    assert pdot == pytest.approx(-2.40263e-12, rel=2e-4)


def test_j0737_periastron_and_einstein_delay_match_measurement():
    # Kramer et al. (2021) masses come from a GR fit to these same PK
    # parameters, so the leading-order prediction must sit within the
    # 2PN size of the measured values: 16.899323 deg/yr and 0.384045 ms.
    res = PostKeplerianSolver.solve(PSR_J0737_3039)
    assert res["om_dot_pred_deg_yr"] == pytest.approx(16.899323, rel=5e-5)
    assert res["gamma_pred_ms"] == pytest.approx(0.384045, rel=1e-3)


def test_b1913_einstein_delay_matches_measurement():
    # Weisberg & Huang (2016) Table 2: gamma = 4.307(4) ms; their masses
    # 1.438(1) and 1.390(1) Msun come from <omdot> and gamma.
    res = PostKeplerianSolver.solve(PSR_B1913_16)
    assert res["gamma_pred_ms"] == pytest.approx(4.307, abs=0.004)
    assert PSR_B1913_16.gamma_obs_ms == 4.307


def test_geodetic_precession_of_pulsar_b_matches_breton_measurement():
    # Breton et al. (2008) measured 4.77 (+0.66, -0.65) deg/yr for pulsar B.
    # The formula is evaluated for body 1, so swap A and B.
    swapped = dataclasses.replace(PSR_J0737_3039,
                                  m1_msun=PSR_J0737_3039.m2_msun,
                                  m2_msun=PSR_J0737_3039.m1_msun)
    rate = PostKeplerianSolver.solve(swapped)["omega_geod_deg_yr"]
    assert 4.77 - 0.65 < rate < 4.77 + 0.66


@pytest.mark.parametrize("seed", range(10))
def test_geodetic_precession_scaling(seed):
    # Barker and O'Connell rate scales as Pb^(-5/3) and (1 - e^2)^(-1).
    rng = np.random.default_rng(200 + seed)
    base = dataclasses.replace(PSR_J0737_3039, Pb_days=rng.uniform(0.05, 10.0),
                               eccentricity=rng.uniform(0.0, 0.7))
    k = rng.uniform(1.2, 4.0)
    slower = dataclasses.replace(base, Pb_days=base.Pb_days * k)
    circ = dataclasses.replace(base, eccentricity=0.0)
    a = PostKeplerianSolver.solve(base)["omega_geod_deg_yr"]
    assert PostKeplerianSolver.solve(slower)["omega_geod_deg_yr"] / a == pytest.approx(k**(-5/3), rel=1e-12)
    assert a / PostKeplerianSolver.solve(circ)["omega_geod_deg_yr"] == pytest.approx(1 / (1 - base.eccentricity**2), rel=1e-12)


@pytest.mark.parametrize("seed", range(25))
def test_pk_scaling_laws_hold_for_random_binaries(seed):
    # Properties of the leading-order formulas that hold for any binary:
    # omdot scales as M^(2/3) at fixed m1/m2 ratio, gamma vanishes with e,
    # and Pbdot scales as the chirp-mass combination m1 m2 M^(-1/3).
    rng = np.random.default_rng(seed)
    base = dataclasses.replace(PSR_J0737_3039,
                               m1_msun=rng.uniform(0.2, 3.0), m2_msun=rng.uniform(0.2, 3.0),
                               Pb_days=rng.uniform(0.05, 30.0), eccentricity=rng.uniform(0.0, 0.8))
    k = rng.uniform(1.1, 3.0)
    scaled = dataclasses.replace(base, m1_msun=base.m1_msun * k, m2_msun=base.m2_msun * k)
    a, b = PostKeplerianSolver.solve(base), PostKeplerianSolver.solve(scaled)
    assert b["om_dot_pred_deg_yr"] / a["om_dot_pred_deg_yr"] == pytest.approx(k**(2/3), rel=1e-12)
    assert b["Pdot_b_pred"] / a["Pdot_b_pred"] == pytest.approx(k**(5/3), rel=1e-12)
    assert b["r_pred_us"] / a["r_pred_us"] == pytest.approx(k, rel=1e-12)
    circular = PostKeplerianSolver.solve(dataclasses.replace(base, eccentricity=0.0))
    assert circular["gamma_pred_ms"] == 0.0


# Triple-engine amplitude scales, pinned to a separate SI computation.
def _independent_triple_scales(t):
    m1, m2, m3 = (x * M_sun for x in (t.m1_msun, t.m2_msun, t.m3_msun))
    p1, p2 = t.Pb1_days * day_s, t.Pb2_days * day_s
    a1 = (G * (m1 + m2) * p1**2 / (4 * np.pi**2))**(1/3)
    a2 = (G * (m1 + m2 + m3) * p2**2 / (4 * np.pi**2))**(1/3)
    sini = np.sin(np.arccos(t.cos_i1))
    quad = (m2 / (m1 + m2)) * a1 * (m3 / (m1 + m2 + m3)) * (p1 / p2)**2 / c * sini
    eps = a1 / a2
    eta = m1 * m2 / (m1 + m2)**2
    return {
        "A_quad_us": quad * 1e6,
        "A_oct_ns": quad * eps * (m1 - m2) / (m1 + m2) * 1e9,
        "A_hex_ns": quad * eps**2 * (1 - 3 * eta) * 1e9,
        "projected_axis_s": (m2 / (m1 + m2)) * a1 * sini / c,
        "n1": 2 * np.pi / p1, "a1": a1, "p1": p1, "eta": eta,
    }


def test_triple_amplitudes_match_independent_si_computation():
    res = RelationalTripleEngine.solve(PSR_J0337_1715)
    ref = _independent_triple_scales(PSR_J0337_1715)
    for key in ("A_quad_us", "A_oct_ns", "A_hex_ns"):
        assert res[key] == pytest.approx(ref[key], rel=1e-6), key


def test_triple_inner_decay_and_delay_scales():
    t = PSR_J0337_1715
    res = RelationalTripleEngine.solve(t, baseline_years=5.0)
    ref = _independent_triple_scales(t)
    m_in = (t.m1_msun + t.m2_msun) * M_sun
    e = t.e1
    fe = (1 + 73/24 * e**2 + 37/96 * e**4) / (1 - e**2)**3.5
    pdot = -(192 * np.pi / 5) * (ref["n1"] * G * m_in / c**3)**(5/3) * ref["eta"] * fe
    assert res["Pdot_b1"] == pytest.approx(pdot, rel=1e-6)
    v = ref["n1"] * ref["a1"] / c
    assert res["delta_t_2pn_ps"] == pytest.approx(ref["projected_axis_s"] * v**4 * 1e12, rel=1e-6)
    n_dot = -ref["n1"] * pdot / ref["p1"]
    expected = ref["projected_axis_s"] * 0.5 * n_dot * (5.0 * 365.25 * day_s)**2 * 1e9
    assert res["gw_delay_scale_ns"] == pytest.approx(expected, rel=1e-6)
    doubled = RelationalTripleEngine.solve(t, baseline_years=10.0)
    assert doubled["gw_delay_scale_ns"] / res["gw_delay_scale_ns"] == pytest.approx(4.0, rel=1e-12)


def test_triple_octupole_flips_sign_when_inner_bodies_swap():
    t = PSR_J0337_1715
    swapped = dataclasses.replace(t, m1_msun=t.m2_msun, m2_msun=t.m1_msun)
    a, b = RelationalTripleEngine.solve(t), RelationalTripleEngine.solve(swapped)
    ref_b = _independent_triple_scales(swapped)
    assert np.sign(a["A_oct_ns"]) == -np.sign(b["A_oct_ns"])
    assert b["A_quad_us"] == pytest.approx(ref_b["A_quad_us"], rel=1e-6)


# Pulsar position and Roemer delay.
@pytest.mark.parametrize("seed", range(20))
def test_pulsar_roemer_delay_reconstruction_from_relational_state(seed):
    # The dynamics stay relational (Jacobi r1, r2). This checks only the
    # observable map: a telescope times pulses against the triple's
    # barycentre, so the delay needs the pulsar's offset from it. Oracle:
    # recover that offset by solving m1 x1 + m2 x2 + m3 x3 = 0,
    # x2 - x1 = r1, x3 - (m1 x1 + m2 x2)/(m1 + m2) = r2 as a linear system.
    rng = np.random.default_rng(100 + seed)
    m1, m2, m3 = rng.uniform(0.1, 3.0, 3) * M_sun
    r1, r2, los = rng.normal(size=(3, 3)) * [[1e9], [3e10], [1.0]]
    state = RelationalState(m1, m2, m3, r1, np.zeros(3), r2, np.zeros(3))
    A = np.array([[m1, m2, m3], [-1, 1, 0], [-m1/(m1+m2), -m2/(m1+m2), 1]])
    x = np.linalg.solve(A, np.vstack([np.zeros(3), r1, r2]))
    expected = -np.dot(x[0], los / np.linalg.norm(los)) / c
    assert state.pulsar_roemer_delay_s(los) == pytest.approx(expected, rel=1e-10, abs=1e-9)


def _roemer_semi_amplitude_s(triple, orbits):
    res = RelationalThreeBodyIntegrator.integrate(triple, num_inner_orbits=orbits, num_output_points=3000)
    return 0.5 * np.ptp(res["roemer_delay_us"]) / 1e6


def test_inner_orbit_roemer_amplitude_matches_analytic_projection():
    # With a negligible outer body the pulsar traces only the inner orbit,
    # whose projected semi-major axis is (m2/M) a1 sin(i1) / c.
    t = dataclasses.replace(PSR_J0337_1715, m3_msun=1e-12)
    expected = _independent_triple_scales(t)["projected_axis_s"]
    assert _roemer_semi_amplitude_s(t, 2.0) == pytest.approx(expected, rel=2e-3)
    assert expected == pytest.approx(1.2175, rel=0.02)  # Archibald et al. asini_i


def test_outer_orbit_roemer_amplitude_matches_analytic_projection():
    # With a negligible inner companion the pulsar moves only on the outer
    # orbit: semi-amplitude (m3/M_tot) a2 sin(i2) / c over one outer period.
    t = dataclasses.replace(PSR_J0337_1715, m2_msun=1e-9, e2=0.0)
    m1, m3 = t.m1_msun * M_sun, t.m3_msun * M_sun
    p2 = t.Pb2_days * day_s
    a2 = (G * (m1 + m3) * p2**2 / (4 * np.pi**2))**(1/3)
    expected = m3 / (m1 + m3) * a2 * np.sin(np.arccos(t.cos_i2)) / c
    orbits = t.Pb2_days / t.Pb1_days
    assert _roemer_semi_amplitude_s(t, orbits) == pytest.approx(expected, rel=2e-3)


# Archibald loader: uncertainties must be converted with the same factor as residuals.
def test_archibald_loader_scales_uncertainty_like_residual(tmp_path):
    path = tmp_path / "archibald.csv"
    header = "# MJD (approx),residual,uncertainty (phase),j_WSRT_1400,j_GBT_1400,delta\n"
    rows = [f"{56000 + i},{1e-3 * (-1)**i},{2e-3},0,0,1\n" for i in range(20)]
    path.write_text(header + "".join(rows))
    res = ZenodoResidualEvaluator.evaluate(str(path), num_windows=2)
    period_us = 1e6 / 365.953339305
    assert res["overall_rms_us"] == pytest.approx(1e-3 * period_us, rel=1e-12)
    assert res["chi2_per_observation"] == pytest.approx(0.25, rel=1e-12)


# QRC prediction is kept as a labeled theory input and compared with the
# independent published measurement, never fitted to the residuals.
def test_qrc_sep_prediction_is_reported_and_checked_against_published_bound():
    res = RelationalTripleEngine.solve(PSR_J0337_1715)
    qrc = res["qrc_sep_prediction"]
    assert qrc["value"] == 0.0
    assert qrc["origin"].startswith("QRC theory prediction")
    assert qrc["within_published_95_bound"] is True
    assert qrc["pull_vs_published_sigma"] == pytest.approx(1.09 / 0.74, rel=1e-12)


def test_qrc_sep_check_reports_conflict_when_prediction_exceeds_bound():
    t = dataclasses.replace(PSR_J0337_1715, delta_sep_qrc_prediction=5e-6)
    qrc = RelationalTripleEngine.solve(t)["qrc_sep_prediction"]
    assert qrc["within_published_95_bound"] is False
    assert qrc["pull_vs_published_sigma"] == pytest.approx((5e-6 + 1.09e-6) / 0.74e-6, rel=1e-12)
