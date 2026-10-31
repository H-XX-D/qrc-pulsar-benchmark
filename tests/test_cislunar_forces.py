"""Independent physics, sampling and native-language bridge regressions."""
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from qrc_pulsar_bench.cislunar_forces import (
    AU, GM_MOON, GM_SUN, R_MOON, CislunarForceParameters, PlanetaryEphemeris,
    acceleration_components, cislunar_acceleration, j2_acceleration,
    lunar_pole_icrf, sunlight_fraction, third_body_tide,
)
from qrc_pulsar_bench.native_cislunar import NativeCislunarKernel
from qrc_pulsar_bench.station_keeping import EarthMoonCR3BP
from qrc_pulsar_bench.multi_path_optimizer import MultiPathTrajectoryOptimizer


@pytest.mark.parametrize('r', [[2e6, 0, 0], [0, 0, 2e6], [2e6, -3e6, 4e6]])
def test_j2_is_negative_gradient_of_quadrupole_potential(r):
    r = np.array(r, dtype=float)
    k = np.array([.2, -.3, .9]); k /= np.linalg.norm(k)
    j2 = 2e-4
    def potential(x):
        rn = np.linalg.norm(x)
        return GM_MOON * j2 * R_MOON**2 / (2 * rn**3) * (3*(np.dot(x,k)/rn)**2-1)
    h = .2
    expected = np.array([-(potential(r+h*e)-potential(r-h*e))/(2*h) for e in np.eye(3)])
    np.testing.assert_allclose(j2_acceleration(r, GM_MOON, R_MOON, j2, k), expected, rtol=1e-7, atol=1e-13)
    np.testing.assert_allclose(j2_acceleration(2*r, GM_MOON, R_MOON, j2, k), expected/16, rtol=1e-7, atol=1e-13)


def test_j2_equator_inward_pole_outward_and_cr3bp_units():
    model = EarthMoonCR3BP()
    centre = np.array([1-model.mu, 0, 0])
    equator = model.compute_disturbances(0, centre+[.2,0,0], srp_mps2=0)
    pole = model.compute_disturbances(0, centre+[0,0,.2], srp_mps2=0)
    scale = 1.5 * model.consts.j2_moon * model.mu * model.consts.r_moon_norm**2 / .2**4
    np.testing.assert_allclose(equator, [-scale,0,0], rtol=1e-13)
    np.testing.assert_allclose(pole, [0,0,2*scale], rtol=1e-13)


def test_solar_tide_has_indirect_term_and_correct_small_distance_limit():
    body = np.array([AU, 0, 0])
    np.testing.assert_array_equal(third_body_tide(np.zeros(3), body, GM_SUN), np.zeros(3))
    x = 1e6
    expected = GM_SUN / AU**3 * np.array([2*x, -x, 0])
    np.testing.assert_allclose(third_body_tide([x,x,0], body, GM_SUN), expected, rtol=3e-5)


def test_srp_direction_inverse_square_area_mass_and_umbra():
    r = np.array([2e6, 0, 0]); earth = np.array([0, 384e6, 0]); sun = np.array([AU, 0, 0])
    p = CislunarForceParameters(area_mass_m2_kg=.02)
    srp = acceleration_components(r, earth, sun, [0,0,1], p)['srp']
    assert srp[0] < 0 and srp[1] == srp[2] == 0
    p2 = CislunarForceParameters(area_mass_m2_kg=.04)
    np.testing.assert_allclose(acceleration_components(r, earth, sun, [0,0,1], p2)['srp'], 2*srp)
    np.testing.assert_allclose(acceleration_components(r, earth, r+2*(sun-r), [0,0,1], p)['srp'], srp/4)
    assert sunlight_fraction(-r, earth, sun) == 0
    assert sunlight_fraction(r, earth, sun) == 1
    np.testing.assert_array_equal(acceleration_components(-r, earth, sun, [0,0,1], p)['srp'], np.zeros(3))
    np.testing.assert_array_equal(acceleration_components(r, earth, sun, [0,0,1])['srp'], np.zeros(3))


def test_ephemeris_uses_tdb_and_hermite_not_trajectory_cadence():
    pytest.importorskip('astropy')
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time
    jd = 2459898.5
    provider = PlanetaryEphemeris(jd, 7200)
    earth, sun, pole = provider(0)
    t = Time(jd, format='jd', scale='tdb')
    moon = get_body_barycentric_posvel('moon', t)[0].xyz.to_value('m')
    # Subtracting before/after AU-to-m conversion differs by micrometres.
    np.testing.assert_allclose(earth, get_body_barycentric_posvel('earth', t)[0].xyz.to_value('m')-moon, atol=5e-5, rtol=0)
    np.testing.assert_allclose(sun, get_body_barycentric_posvel('sun', t)[0].xyz.to_value('m')-moon, atol=1e-4, rtol=0)
    assert np.linalg.norm(pole) == pytest.approx(1)
    with pytest.raises(ValueError): provider(7201)
    assert np.linalg.norm(pole[:2]) > .3
    assert np.linalg.norm(pole-lunar_pole_icrf(jd+180)) > 1e-3


@pytest.fixture(scope='module')
def native_library(tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp('native') / 'libcislunar.so'
    subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC', '-Wall', '-Wextra',
                    '-Wno-unknown-pragmas', str(root/'hardware/cislunar_force_core.cpp'),
                    '-o', str(library)], check=True, capture_output=True, text=True)
    return library


def environment(t):
    angle = t/375190.3
    return (np.array([384e6*np.cos(angle),384e6*np.sin(angle),2e7]),
            np.array([AU, -AU*.2, AU*.1]), lunar_pole_icrf(2459898.5+t/86400))


def test_cpp_python_force_parity_for_rotated_states_and_each_term(native_library):
    rng = np.random.default_rng(17)
    for parameters in [CislunarForceParameters(), CislunarForceParameters(area_mass_m2_kg=.02),
                       CislunarForceParameters(gm_earth=0,gm_sun=0,moon_j2=0)]:
        native = NativeCislunarKernel(native_library, parameters)
        for _ in range(30):
            r = rng.normal(size=3); r *= rng.uniform(2e6, 1e8)/np.linalg.norm(r)
            env = environment(rng.uniform(0,86400))
            np.testing.assert_allclose(native.acceleration(r, env), cislunar_acceleration(r, *env, parameters), rtol=2e-12, atol=2e-17)
    env = (np.array([0,384e6,0]),np.array([AU,0,0]),np.array([0,0,1]))
    parameters = CislunarForceParameters(area_mass_m2_kg=.02)
    native = NativeCislunarKernel(native_library, parameters)
    np.testing.assert_allclose(native.acceleration([-2e6,0,0],env), cislunar_acceleration([-2e6,0,0],*env,parameters))


def test_cpp_time_dependent_step_converges_to_independent_solver(native_library):
    native = NativeCislunarKernel(native_library)
    initial = np.array([2e6, 0, 0, 0, 1500, 100.0])
    def rhs(t,y): return np.r_[y[3:], cislunar_acceleration(y[:3], *environment(t))]
    reference = solve_ivp(rhs, [0,600], initial, method='DOP853', rtol=2e-13, atol=1e-9).y[:3,-1]
    errors = []
    for h in [10.,5.,2.5]:
        r, v = initial[:3].copy(), initial[3:].copy()
        for t in np.arange(0,600,h):
            r,v = native.step(r,v,h,environment(t),environment(t+h))
        errors.append(np.linalg.norm(r-reference))
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1


def test_cpp_invalid_inputs_do_not_mutate_state(native_library):
    native = NativeCislunarKernel(native_library)
    r=np.array([2e6,0,0.]); v=np.array([0,1500,0.]); env=np.asarray(environment(0)).reshape(-1)
    before_r, before_v = r.copy(), v.copy()
    assert native.library.cislunar_verlet_step(r,v,-1.,env,env,native.parameters) == 1
    np.testing.assert_array_equal(r,before_r); np.testing.assert_array_equal(v,before_v)
    bad_env=env.copy(); bad_env[0]=np.nan
    out=np.array([123.,456.,789.])
    assert native.library.cislunar_acceleration(r,bad_env,native.parameters,out) == 1
    np.testing.assert_array_equal(out,[123.,456.,789.])


def test_optimizer_checks_final_endpoint_and_uses_supplied_forces():
    calls=[]
    def acceleration(r,t):
        calls.append(t)
        return np.zeros(3)
    opt=MultiPathTrajectoryOptimizer(acceleration=acceleration)
    _, radius, safe = opt.evaluate_candidate_path(np.array([R_MOON+500010,0,0]),
        np.array([-100.,0,0]),np.zeros(3),np.zeros(3),horizon_s=1,dt_sub=1,start_time_s=100)
    assert not safe and radius == pytest.approx(R_MOON+499910)
    assert calls == [100]


def test_lunar_pole_matches_independent_naif_spice_fixture():
    # Generated with CSPICE pxform(IAU_MOON,J2000,et), pck00011.tpc;
    # et = (JD_TDB - 2451545) * 86400. No optional SPICE runtime required.
    np.testing.assert_allclose(lunar_pole_icrf(2459898.5),
        [-0.018623297367304174,-0.3788936150168519,0.925252831011411], atol=2e-15,rtol=0)


def test_base_package_import_does_not_require_navigation_extra():
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('astropy', 'pint')):
            raise ImportError('Optional dependencies blocked by regression test')
sys.meta_path.insert(0, Block())
import qrc_pulsar_bench
from qrc_pulsar_bench.cislunar_forces import CislunarForceParameters
assert CislunarForceParameters().gm_moon > 0
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
