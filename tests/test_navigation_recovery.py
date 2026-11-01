from dataclasses import replace
import numpy as np
import pytest
from qrc_pulsar_bench.adaptive_control import (CausalOrbitFilter, MoonCartesianFlightModel,
    square_root_measurement, validate_covariance, propagate_known_commands)
from qrc_pulsar_bench.navigation_recovery import ManeuverAwareOrbitFilter, ReacquisitionSettings
from qrc_pulsar_bench.jacobi_control import Actuator


INITIAL=np.array([1e7,2e7,3e7,0.,0.,0.])
R=np.diag(np.r_[[30.]*3,[.002]*3]**2)


def model():
    return MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),
        lambda t:(np.array([1e8,0,0]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=1800.)


def nav():
    f=ManeuverAwareOrbitFilter(model(),INITIAL,R,500.)
    f.observe(INITIAL,R,0,60)
    return f


def test_joint_square_root_preserves_small_noise_in_ill_conditioned_prior():
    # P=LL' loses its 1e-6 differential variance at 1e20 scale. R is
    # also rounded away by P+R, but the factor representation retains both.
    root=np.array([[1e10,0.],[1e10,1e-3]])
    nis,correction,posterior=square_root_measurement(root,np.eye(2),np.array([1.,-1.]))
    # The factor condition is ~1e10; a few ppm is the double-precision
    # backward-error scale here, versus singular full-matrix arithmetic.
    assert nis==pytest.approx(4/(2+1e-6),rel=2e-6)
    assert np.all(np.isfinite(correction))
    p=posterior@posterior.T;validate_covariance(p)
    expected=np.array([[1+1e-6,1.],[1.,1+1e-6]])/(2+1e-6)
    np.testing.assert_allclose(p,expected,rtol=2e-6,atol=2e-6)


def test_joint_square_root_matches_well_conditioned_kalman_update():
    rng=np.random.default_rng(3401);root=rng.normal(size=(9,9));root+=3*np.eye(9)
    r=R;innovation=rng.normal(size=6);p=root@root.T;s=p[:6,:6]+r
    gain=np.linalg.solve(s,p[:6,:]).T
    nis,correction,posterior=square_root_measurement(root,r,innovation)
    np.testing.assert_allclose(correction,gain@innovation,atol=1e-12)
    assert nis==pytest.approx(innovation@np.linalg.solve(s,innovation),rel=1e-12)
    j=np.eye(9);j[:,:6]-=gain
    np.testing.assert_allclose(posterior@posterior.T,j@p@j.T+gain@r@gain.T,atol=1e-12)


def test_invalid_update_is_transactional(monkeypatch):
    f=nav();state=f.state.copy();root=f._root.copy()
    counters=(f.accepted,f.rejected,f.bias_clips,f.time_s,f.last_observation_s,f.last_processed_observation_s)
    def bad(*args):return np.full(9,np.nan),root,500.
    monkeypatch.setattr(f,'_predict_factor',bad)
    with pytest.raises(ValueError):f.observe(INITIAL,R,1800,1860)
    np.testing.assert_array_equal(state,f.state);np.testing.assert_array_equal(root,f._root)
    assert counters==(f.accepted,f.rejected,f.bias_clips,f.time_s,f.last_observation_s,f.last_processed_observation_s)


@pytest.mark.parametrize('jump',[.05,.2,.5,1.])
def test_real_jump_requires_three_distinct_consistent_fixes(jump):
    f=nav()
    for i,t in enumerate([1800.,3600.,5400.]):
        fix=INITIAL+np.array([jump*(t-1800),0,0,jump,0,0])
        result=f.observe(fix,R,t,t+60)
        assert result['reacquired']==(i==2)
        assert f.forecast(t+60)['navigation_trusted']==(i==2)
    assert f.reacquisitions==1 and len(f.events)==1
    assert f.events[0]['confirmations']==3
    np.testing.assert_allclose(f.state[:6],fix,atol=1e-6)
    assert f.last_observation_s==5400.


def test_single_bad_fix_does_not_replace_navigation():
    f=nav();bad=INITIAL.copy();bad[3]=.5
    assert not f.observe(bad,R,1800,1860)['accepted']
    assert f.candidate is not None and not f.navigation_trusted
    assert f.observe(INITIAL,R,3600,3660)['accepted']
    assert f.navigation_trusted and f.candidate is None and f.reacquisitions==0
    np.testing.assert_allclose(f.state[:6],INITIAL,atol=1e-8)


@pytest.mark.parametrize('position_bias,velocity_bias',[(0,.5),(150.,.006),(3000.,.5),(3000.,0.)])
def test_persistent_inconsistent_sensor_bias_cannot_confirm_jump(position_bias,velocity_bias):
    f=nav()
    bad=INITIAL+np.array([position_bias,0,0,velocity_bias,0,0])
    for t in np.arange(1800,18001,1800):f.observe(bad,R,t,t+60)
    assert f.reacquisitions==0


def test_replayed_and_future_fixes_cannot_confirm_candidate():
    f=nav();fix=INITIAL.copy();fix[3]=.5
    f.observe(fix,R,1800,1860);candidate=f.candidate;root=candidate._root.copy()
    for stamp,arrival in [(1800,2000),(1700,2000),(3600,3500)]:
        with pytest.raises(ValueError):f.observe(fix,R,stamp,arrival)
    assert f.confirmations==1 and f.candidate is candidate
    np.testing.assert_array_equal(root,candidate._root)


def test_long_gap_restarts_confirmation():
    f=nav()
    for t in [1800.,7200.]:
        fix=INITIAL+np.array([.5*(t-1800),0,0,.5,0,0])
        f.observe(fix,R,t,t+60)
    assert f.reacquisitions==0 and f.confirmations==1 and f.candidate_resets==1


def test_candidate_accounts_for_known_finite_burn_without_double_use():
    f=nav();state=INITIAL.copy();state[3]=.2
    f.observe(state,R,1800,1860)
    burn=Actuator().command([0,.02,0],2000.,500.)
    for t in [3600.,5400.]:
        fix,_,_=propagate_known_commands(model(),state,1800.,t,500.,[burn],as_of_s=t+60)
        f.observe(fix,R,t,t+60,[burn])
    assert f.reacquisitions==1
    np.testing.assert_allclose(f.state[:6],fix,atol=1e-6)
    assert f.mass==pytest.approx(500.-burn.mass_flow_kg_s*(burn.stop_s-burn.start_s))


def test_coordinated_false_trajectory_is_unidentifiable_from_one_sensor_stream():
    # This is a documented limit, not a desired security guarantee: coherent
    # spoofing has exactly the same inputs as a physical jump in this model.
    f=nav()
    for t in [1800.,3600.,5400.]:
        f.observe(INITIAL+np.array([.5*(t-1800),0,0,.5,0,0]),R,t,t+60)
    assert f.reacquisitions==1


@pytest.mark.parametrize('kwargs',[{'confirmation_fixes':2},{'confirmation_fixes':3.5},
    {'seed_covariance_multiplier':.5},{'minimum_jump_m_s':3.},{'maximum_track_age_s':float('nan')}])
def test_invalid_reacquisition_settings(kwargs):
    with pytest.raises(ValueError):ReacquisitionSettings(**kwargs)


def test_untrusted_navigation_cannot_command_or_rearm():
    from qrc_pulsar_bench.routine_supervisor import RoutineMPC
    from test_mpc_hardening import Model,Plan,estimate
    controller=RoutineMPC(Model(),Plan());e=estimate();e['navigation_trusted']=False
    assert not controller.command(0,e,500.).any() and controller.last_status==2
    with pytest.raises(ValueError,match='Trusted'):controller.rearm(1800,e,500.)
    assert not controller.issued


@pytest.mark.parametrize('bad',[None,[],0.])
def test_malformed_navigation_fails_closed_before_trust_flag_access(bad):
    from qrc_pulsar_bench.routine_supervisor import RoutineMPC
    from test_mpc_hardening import Model,Plan
    controller=RoutineMPC(Model(),Plan())
    assert not controller.command(0,bad,500.).any() and controller.last_status==1


def test_long_observation_gap_reacquires_without_a_valid_old_track():
    f=nav()
    for i,t in enumerate([21600.,23400.,25200.]):
        fix=INITIAL+np.array([.35*t,0,0,.35,0,0])
        result=f.observe(fix,R,t,t+60)
        assert result['reacquired']==(i==2)
    assert f.events[0]['reason']=='observation_gap'
    np.testing.assert_allclose(f.state[:6],fix,atol=1e-6)


def test_bad_first_returning_fix_does_not_lock_out_later_good_data():
    f=nav()
    bad=INITIAL+np.array([10000.,0,0,.5,0,0])
    f.observe(bad,R,21600.,21660.)
    for t in [23400.,25200.,27000.]:
        fix=INITIAL+np.array([.35*t,0,0,.35,0,0])
        f.observe(fix,R,t,t+60)
    assert f.reacquisitions==1 and f.candidate_resets==1
    assert f.events[0]['reason']=='observation_gap'
    np.testing.assert_allclose(f.state[:6],fix,atol=1e-6)


def test_bad_velocity_stream_after_gap_cannot_confirm_motion():
    f=nav();bad=INITIAL.copy();bad[3]=.6
    for t in np.arange(21600.,32401.,1800.):f.observe(bad,R,t,t+60)
    assert f.reacquisitions==0 and not f.navigation_trusted


def test_rejection_streak_does_not_masquerade_as_missing_observations():
    f=nav();bad=INITIAL+np.array([10000.,0,0,.6,0,0])
    for t in np.arange(1800.,21601.,1800.):f.observe(bad,R,t,t+60)
    assert not f.gap_reacquisition_pending and f.reacquisitions==0
