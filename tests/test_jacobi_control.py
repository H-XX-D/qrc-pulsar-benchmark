"""Analytic coordinate/rocket controls and explicit safety failure scenarios."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from qrc_pulsar_bench.jacobi_control import (
    Actuator, NavigationEstimate, ControlSettings, JacobiFlightModel,
    JacobiPredictiveController, G0)


def moving_inner(t):
    return (np.array([2e7+.001*t*t, -1e8+100*t, 0.]),
            np.array([.002*t, 100., 0.]), np.array([.002, 0., 0.]))


def model(acceleration=lambda r,v,t:np.zeros(3), step=10.):
    return JacobiFlightModel(acceleration, moving_inner, moon_radius_m=100.,
                            earth_radius_m=100., max_step_s=step, rtol=1e-12)


def test_accelerated_jacobi_origin_preserves_free_physical_motion():
    m = model()
    x = np.array([1e6, 2e6, 3e6, -10., 20., 30.])
    q = m.from_moon(x, 0.)
    result = m.propagate(q, 0., 1000., 500.)
    for t in (0., 124., 567., 1000.):
        expected = x.copy(); expected[:3] += t*x[3:]
        np.testing.assert_allclose(m.to_moon(result.at(t)[:6], t), expected, atol=1e-6, rtol=0)


def test_velocity_dependent_force_reaches_jacobi_integrator():
    rate = .001
    m = model(lambda r,v,t:-rate*v)
    x = np.array([1e6, 2e6, 3e6, -10., 20., 30.])
    result = m.propagate(m.from_moon(x, 0), 0., 1000., 500.)
    expected = np.r_[x[:3]+x[3:]/rate*(1-np.exp(-1)), x[3:]*np.exp(-1)]
    np.testing.assert_allclose(m.to_moon(result.at(1000)[:6],1000),expected,atol=2e-6,rtol=0)


def test_finite_burn_matches_independent_rocket_equation_and_mass_budget():
    m = model(); initial = m.from_moon([1e6,0,0,0,0,0],0)
    actuator = Actuator(thrust_n=20., dry_mass_kg=10., reserve_kg=0., routine_burn_s=1000.)
    burn = actuator.command([10.,0,0],0.,500.)
    end = burn.stop_s+100.
    result = m.propagate(initial,0.,end,500.,burn)
    final = result.at(end); ve = actuator.exhaust_velocity
    mf = 500.-burn.mass_flow_kg_s*burn.stop_s
    displacement = ve*(burn.stop_s+mf/burn.mass_flow_kg_s*np.log(mf/500.))+10.*100.
    physical = m.to_moon(final[:6],end)
    np.testing.assert_allclose(physical,[1e6+displacement,0,0,10.,0,0],atol=2e-6,rtol=0)
    assert final[6] == pytest.approx(mf,abs=1e-10)


def test_reserve_and_thrust_duration_limits():
    actuator = Actuator(thrust_n=10.,dry_mass_kg=10.,reserve_kg=.5,routine_burn_s=1.,emergency_burn_s=1000.)
    mass = 10.51
    for emergency in (False,True):
        burn = actuator.command([1e9,0,0],0,mass,emergency=emergency)
        floor = 10 if emergency else 10.5
        assert mass-burn.mass_flow_kg_s*burn.stop_s >= floor-1e-12
        assert burn.stop_s <= (1000 if emergency else 1)+1e-9
    assert actuator.command([1,0,0],0,10.5) is None
    assert replace(actuator,thrust_n=0).command([1,0,0],0,mass,emergency=True) is None
    with pytest.raises(ValueError): actuator.capacity(9.)


def test_future_or_inconsistent_burn_rejected():
    m = model(); q = m.from_moon([1e6,0,0,0,0,0],0)
    burn = Actuator().command([.01,0,0],0,500)
    for altered in (replace(burn,available_at_s=1.),replace(burn,mass_start_kg=501.)):
        with pytest.raises(ValueError): m.propagate(q,0,1000,500,altered)


def safety_case(thrust=20., body='moon'):
    m = model(step=1.)
    x = np.array([600.,0,0,-10.,0,0])
    if body=='earth':
        r,v,_ = moving_inner(0)
        x = np.r_[-r+np.array([600.,0,0]),-v+np.array([-10.,0,0])]
    q = m.from_moon(x,0.)
    settings = ControlSettings(decision_interval_s=100.,planning_horizon_s=100.,
        linear_step_s=10.,moon_floor_m=100.,earth_floor_m=100.,minimum_burn_spacing_s=100.)
    controller = JacobiPredictiveController(m,lambda t:q,100.,settings=settings,
        actuator=Actuator(thrust_n=thrust,isp_s=500.,dry_mass_kg=5.,reserve_kg=0.,
                          emergency_burn_s=20.))
    return m,controller,NavigationEstimate(q,0,0)


@pytest.mark.parametrize('body',['moon','earth'])
def test_emergency_finite_burn_avoids_predicted_body_floor(body):
    m,c,estimate = safety_case(body=body)
    coast = m.propagate(estimate.state,0,100,10.,floors=(100,100))
    assert not coast.completed
    decision = c.decide(estimate,10.)
    assert decision.mode=='EMERGENCY_BURN'
    assert decision.burn is not None
    checked = m.propagate(estimate.state,0,100,10.,decision.burn,floors=(100,100))
    assert checked.completed and checked.min_clearance_m > 0


def test_no_thruster_does_not_claim_rescue():
    _,c,estimate = safety_case(thrust=0.)
    decision = c.decide(estimate,10.)
    assert decision.mode=='NO_FEASIBLE_EMERGENCY' and decision.burn is None


def test_below_clearance_does_not_claim_rescue():
    m,c,_ = safety_case()
    q = m.from_moon([190,0,0,10,0,0],0)
    decision = c.decide(NavigationEstimate(q,0,0),10.)
    assert decision.mode=='CLEARANCE_ALREADY_VIOLATED' and decision.burn is None


def test_noise_deadband_stale_observations_and_copy_boundary():
    m = model(); x = np.array([1e6,0,0,0,0,0])
    reference=lambda t:m.from_moon(x,t)
    c=JacobiPredictiveController(m,reference,86400.)
    q=reference(10000); estimate=NavigationEstimate(q,10000,0)
    q[0]+=1e6
    assert estimate.state[0]!=q[0]
    assert c.decide(estimate,500.).mode=='NAVIGATION_STALE'
    for offset in (-50.,50.,-75.,75.):
        noisy=reference(0)+np.array([offset,0,0,0,0,0])
        assert c.decide(NavigationEstimate(noisy,0,0),500.).burn is None
    with pytest.raises(ValueError):NavigationEstimate(reference(0),0,1)
    with pytest.raises(ValueError):NavigationEstimate([np.nan]*6,0,0)


def test_damped_feedback_analytic_nonoscillation_envelope():
    m=model(); x=np.array([1e6,0,0,0,0,0]); tau=1000.
    c=JacobiPredictiveController(m,lambda t:m.from_moon(x,t),10000.,
        settings=ControlSettings(response_time_s=tau,decision_interval_s=10.))
    # Independent continuous-acceleration closure verifies the stated design
    # equation, not the constrained finite-burn controller's global stability.
    def rhs(t,y):
        estimate=NavigationEstimate(m.from_moon(x+y,t),t,t)
        return np.r_[y[3:],c._damped_dv(estimate)/c.settings.decision_interval_s]
    times=np.linspace(0,8000,161)
    sol=solve_ivp(rhs,(0,8000),[100.,0,0,0,0,0],t_eval=times,rtol=1e-10,atol=1e-10)
    expected=100*(1+times/tau)*np.exp(-times/tau)
    np.testing.assert_allclose(sol.y[0],expected,atol=1e-6,rtol=0)
    assert np.all(sol.y[0]>0) and np.all(np.diff(sol.y[0])<0)


def test_predictive_burn_reduces_tracking_with_finite_execution():
    m=model(step=60.); x=np.array([1e6,0,0,0,0,0])
    reference=lambda t:m.from_moon(x,t)
    settings=ControlSettings(decision_interval_s=1000.,planning_horizon_s=5000.,linear_step_s=100.,
        response_time_s=2000.,minimum_burn_spacing_s=1000.,position_on_m=10.,position_off_m=2.,
        position_cost_scale_m=100.,dv_cost_scale_m_s=1.)
    c=JacobiPredictiveController(m,reference,10000.,settings=settings,
                                actuator=Actuator(thrust_n=2.,routine_burn_s=500.))
    state=reference(0)+[1000,0,0,0,0,0]
    estimate=NavigationEstimate(state,0,0)
    decision=c.decide(estimate,500.)
    assert decision.burn is not None
    assert len(decision.planned_times_s)==5
    executed=m.propagate(state,0,1000,500.,decision.burn)
    assert np.linalg.norm(executed.at(1000)[:3]-reference(1000)[:3])<1000
    c.confirm_execution(decision)
    assert c.last_burn_s==0


def test_sampled_clearance_is_not_mislabeled_continuous_certificate():
    m,c,estimate=safety_case()
    trial=m.propagate(estimate.state,0,100,10.,floors=(100,100))
    assert not trial.completed
    assert trial.stop_s == pytest.approx(40.,abs=1e-5)
    with pytest.raises(ValueError):trial.at(50.)


def test_optimizer_failure_uses_damped_candidate_and_reserve_is_preserved():
    m=model(step=100.); x=np.array([1e6,0,0,0,0,0])
    reference=lambda t:m.from_moon(x,t)
    settings=ControlSettings(decision_interval_s=1000.,planning_horizon_s=5000.,
        response_time_s=2000.,position_cost_scale_m=100.,dv_cost_scale_m_s=10.)
    c=JacobiPredictiveController(m,reference,10000.,settings=settings,
                                actuator=Actuator(thrust_n=2.,routine_burn_s=500.))
    def failed_planner(*args):raise RuntimeError('injected optimizer failure')
    c._plan=failed_planner
    estimate=NavigationEstimate(reference(0)+[1000,0,0,0,0,0],0,0)
    decision=c.decide(estimate,500.)
    assert decision.mode=='DAMPED_FALLBACK' and decision.burn is not None
    assert c.last_burn_s==-np.inf  # planning alone must not consume dwell time
    depleted=c.decide(estimate,c.actuator.dry_mass_kg+c.actuator.reserve_kg)
    assert depleted.mode=='ROUTINE_AUTHORITY_EXHAUSTED' and depleted.burn is None


def test_orbit_in_large_moving_frame_matches_independent_cartesian_control():
    gm=4.9028e12; omega=2*np.pi/(27.3*86400); scale=3.84e8
    def inner(t):
        r=scale*np.array([np.cos(omega*t),np.sin(omega*t),0.])
        v=scale*omega*np.array([-np.sin(omega*t),np.cos(omega*t),0.])
        return r,v,-omega**2*r
    force=lambda r,v,t:-gm*r/np.linalg.norm(r)**3
    m=JacobiFlightModel(force,inner)
    x=np.array([1e7,0,0,0,np.sqrt(gm/1e7),0]); end=5*86400.
    result=m.propagate(m.from_moon(x,0),0,end,500.)
    control=solve_ivp(lambda t,y:np.r_[y[3:],force(y[:3],y[3:],t)],(0,end),x,
        method='DOP853',rtol=2e-13,atol=np.r_[[1e-6]*3,[1e-10]*3],max_step=900.,dense_output=True)
    difference=[np.linalg.norm(m.to_moon(result.at(t)[:6],t)[:3]-control.sol(t)[:3])
                for t in np.linspace(0,end,101)]
    assert max(difference)<.2
