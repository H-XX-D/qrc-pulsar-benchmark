from dataclasses import replace
import numpy as np
import pytest
from qrc_pulsar_bench.adaptive_control import (MoonCartesianFlightModel,CausalOrbitFilter,FilterSettings,
    BiasAwareController,covariance_step,propagate_known_commands)
from qrc_pulsar_bench.jacobi_control import JacobiFlightModel,NavigationEstimate,Actuator,ControlSettings


def inner(t):
    return np.array([2e7+.001*t*t,-1e8+100*t,0.]),np.array([.002*t,100,0.]),np.array([.002,0,0.])


def free_model():
    return MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),inner,moon_radius_m=1.,earth_radius_m=1.,max_step_s=1000.)


def test_covariance_exponential_matches_integrated_random_walk():
    a=np.array([[0.,1,0],[0,0,1],[0,0,0]])
    q=np.diag([0.,0.,2.]);dt=3.
    phi,noise=covariance_step(a,q,dt)
    expected_phi=np.array([[1,dt,dt*dt/2],[0,1,dt],[0,0,1]])
    expected_q=2*np.array([[dt**5/20,dt**4/8,dt**3/6],[dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
    np.testing.assert_allclose(phi,expected_phi,atol=1e-12,rtol=0)
    np.testing.assert_allclose(noise,expected_q,atol=1e-11,rtol=0)


def test_filter_recovers_constant_acceleration_from_past_measurements():
    model=free_model();initial=np.array([1e6,2e6,3e6,0,0,0.])
    settings=FilterSettings(bias_random_walk_psd=0.,acceleration_noise_psd=0.)
    f=CausalOrbitFilter(model,initial,np.diag([100.]*3+[.01]*3),500.,settings=settings)
    bias=np.array([3e-8,-2e-8,1e-8]);measurement=np.diag([1.]*3+[1e-8]*3)
    for t in np.arange(0,86401,3600):
        true=initial+np.r_[.5*bias*t*t,bias*t]
        row=f.observe(true,measurement,t,t+600)
        assert row['accepted']
    np.testing.assert_allclose(f.state[6:],bias,atol=1e-10,rtol=0)
    np.testing.assert_allclose(f.state[:6],true,atol=.01,rtol=0)


def test_outlier_gate_and_delayed_forecast_do_not_consume_future_observations():
    model=free_model();initial=np.array([1e6,2e6,3e6,0,0,0.])
    f=CausalOrbitFilter(model,initial,np.diag([100.]*3+[.001]*3),500.)
    r=np.diag([1.]*3+[1e-6]*3)
    assert f.observe(initial,r,0,0)['accepted']
    before=f.state.copy();p=f.covariance.copy()
    forecast=f.forecast(1000.)
    np.testing.assert_array_equal(f.state,before);np.testing.assert_array_equal(f.covariance,p)
    assert f.time_s==0
    bad=initial.copy();bad[0]+=1e5
    assert not f.observe(bad,r,500.,1100.)['accepted']
    assert f.last_observation_s==0
    assert f.observe(initial,r,600.,1200.)['accepted']
    assert f.last_observation_s==600 and f.rejected==1
    with pytest.raises(ValueError):f.observe(initial,r,700.,650.)


def test_command_ledger_requires_knowledge_time_and_counts_mass_once():
    model=free_model();initial=np.array([1e6,0,0,0,0,0.])
    burn=Actuator().command([.01,0,0],100.,500.)
    with pytest.raises(ValueError,match='Future command'):
        propagate_known_commands(model,initial,0.,1000.,500.,[burn],as_of_s=0.)
    state,mass,_=propagate_known_commands(model,initial,0.,1000.,500.,[burn],as_of_s=1000.)
    assert mass==pytest.approx(500-burn.mass_flow_kg_s*(burn.stop_s-burn.start_s),abs=1e-9)
    assert state[3]==pytest.approx(.01,abs=1e-10)
    with pytest.raises(ValueError,match='Overlapping'):
        propagate_known_commands(model,initial,0.,1000.,500.,[burn,burn],as_of_s=1000.)


def test_maneuver_execution_uncertainty_increases_velocity_covariance():
    initial=np.array([1e6,0,0,0,0,0.]);model=free_model()
    burn=Actuator().command([.1,0,0],0.,500.)
    zero=FilterSettings(maneuver_fraction_sigma=0.,maneuver_pointing_sigma_rad=0.)
    a=CausalOrbitFilter(model,initial,np.eye(6),500.,settings=zero)
    b=CausalOrbitFilter(model,initial,np.eye(6),500.,settings=replace(zero,maneuver_fraction_sigma=.1))
    pa=a.forecast(1000.,[burn])['covariance'];pb=b.forecast(1000.,[burn])['covariance']
    assert pb[3,3]-pa[3,3]==pytest.approx(.01*.01,rel=1e-6)


def test_bias_feedforward_is_not_cancelled_by_reference_acceleration():
    model=free_model();initial=np.array([1e6,0,0,0,0,0.])
    c=BiasAwareController(model,lambda t:initial,100000.)
    bias=np.array([1e-7,-2e-7,3e-7]);c.set_bias(bias)
    np.testing.assert_allclose(c._damped_dv(NavigationEstimate(initial,0,0)),-bias*c.settings.decision_interval_s,atol=1e-15)
    phi,g=c._affine_transition(0.,100.)
    np.testing.assert_allclose(g,np.vstack([5000*np.eye(3),100*np.eye(3)]),atol=1e-9)


def test_matched_coordinate_controller_commands_agree():
    force=lambda r,v,t:np.zeros(3)
    cart=free_model();jac=JacobiFlightModel(force,inner,moon_radius_m=1.,earth_radius_m=1.,max_step_s=1000.)
    x=np.array([1e6,0,0,0,0,0.]);error=np.array([1000.,300.,-500.,.01,-.01,.005])
    settings=ControlSettings(decision_interval_s=1000.,planning_horizon_s=5000.,linear_step_s=500.,
        response_time_s=2000.,dv_cost_scale_m_s=1.,position_cost_scale_m=100.)
    burns=[]
    for model in (cart,jac):
        ref=lambda t:model.from_moon(x,t)
        c=BiasAwareController(model,ref,10000.,settings=settings)
        c.set_bias([1e-7,2e-7,-1e-7])
        decision=c.decide(NavigationEstimate(model.from_moon(x+error,0.),0,0),500.)
        assert decision.burn is not None
        burns.append(np.asarray(decision.burn.direction_icrf)*(decision.burn.stop_s-decision.burn.start_s))
    np.testing.assert_allclose(burns[0],burns[1],rtol=0,atol=1e-6)


def test_training_selector_rejects_held_out_data():
    from qrc_pulsar_bench.control_tuning import select
    with pytest.raises(ValueError,match='training scenarios only'):
        select([{'scenario':{'name':'test_a'}}],'mpc')
