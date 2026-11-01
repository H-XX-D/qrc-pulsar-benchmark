from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from qrc_pulsar_bench.fpga_mpc import FixedQP,compile_core,float_solve,scaled_qp
from qrc_pulsar_bench.mpc_strategies import StrategyMPC,strategies
from qrc_pulsar_bench.resilient_mpc import GuardedMPC
from qrc_pulsar_bench.adaptive_control import CausalOrbitFilter,FilterSettings,MoonCartesianFlightModel,validate_covariance

class Plan:
    stop_s=1e6
    def at(self,t):return np.zeros(7)
class Model:
    def to_moon(self,x,t):return x
    def acceleration(self,r,v,t):return np.zeros(3)
    def propagate(self,*a,**kw):return SimpleNamespace(completed=True,min_clearance_m=1e6)
def estimate(t=0,age=0,sigma=10):
    return dict(state=np.array([100.,0,0,0,0,0,0,0,0]),
                covariance=np.diag([sigma*sigma]*3+[1e-6]*3+[1e-16]*3),time_s=t,last_observation_s=t-age)

@pytest.mark.parametrize('mutation', ['nan_state','negative_tiny_variance','asymmetry','future_fix','wrong_epoch','shape'])
def test_bad_navigation_inhibits_and_clears_plan(mutation):
    c=StrategyMPC(Model(),Plan(),strategies()['sparse_05']);e=estimate();c.warm[:]=.01
    if mutation=='nan_state':e['state'][2]=np.nan
    elif mutation=='negative_tiny_variance':e['covariance'][6,6]=-1e-20
    elif mutation=='asymmetry':e['covariance'][6,7]=1e-16
    elif mutation=='future_fix':e['last_observation_s']=1
    elif mutation=='wrong_epoch':e['time_s']=1
    else:e['state']=np.zeros(8)
    assert not c.command(0,e,500).any()
    assert c.last_status==1 and not c.warm.any() and c.last_schedule.size==0
    assert np.isfinite(c.command(1800,estimate(1800),500)).all()

def test_duplicate_command_is_not_executed_twice():
    c=StrategyMPC(Model(),Plan(),strategies()['sparse_05'])
    assert c.command(0,estimate(),500).any()
    assert not c.command(0,estimate(),500).any() and c.last_status==1

def test_guard_bounds_age_uncertainty_authority_and_health():
    c=GuardedMPC(Model(),Plan())
    dv=c.command(10000,estimate(10000,10000),500)
    assert c.mode=='degraded' and c.last_status==0 and 0<np.linalg.norm(dv)<=.03
    assert c.diagnostics[-1]['dv_limit_m_s']==.03
    for t,e in [(12000,estimate(12000,sigma=201)),(40000,estimate(40000,40000))]:
        assert not c.command(t,e,500).any() and c.last_status==2 and c.mode=='inhibited'
    assert not c.command(50000,estimate(50000),500,valid=False).any()

def test_cache_and_recent_diagnostics_have_fixed_bounds():
    c=GuardedMPC(Model(),Plan())
    for i in range(1000):
        c._linear(i,10);c.transition(i,10);c.diagnostics.append({'i':i})
    assert len(c.cache)<=12 and len(c.affine_cache)<=12 and len(c.diagnostics)==256

def test_float_solver_invalid_and_unconverged_outputs_are_zero():
    for h,g,w,b,t in [(np.diag([.5,-.1]),[0,0],[0,0],0,1e-5),
                      (np.eye(2),[0,np.nan],[0,0],0,1e-5),
                      (np.eye(2),[0,0],[0,0],1.1,1e-5),
                      (np.eye(2),[0,0],[0,0],0,-1)]:
        status,u,_=float_solve(h,np.array(g),np.array(w),b,t)
        assert status==1 and not u.any()
    status,u,_=float_solve(np.diag([.9,.00002]),np.array([.1,.0002]),np.zeros(2),0,1e-12)
    assert status==16 and not u.any()
    with pytest.raises(ValueError,match='overflow'):scaled_qp(np.eye(2)*1e200,np.ones(2))

def test_library_precision_mismatch_and_lost_convexity_fail(tmp_path):
    path=compile_core(tmp_path/'qp.so',24,16)
    with pytest.raises(ValueError,match='mismatch'):FixedQP(path,32,24)
    qp=FixedQP(path,24,16)
    status,u,_=qp.solve(np.diag([.5,1e-10]),np.zeros(2),np.zeros(2),0)
    assert status==1 and not u.any()

def free_model():
    return MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),
        lambda t:(np.array([1e8,0,0]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=100.)

def test_decaying_bias_mean_and_covariance_match_analytic_solution():
    tau=7200.;t=3600.;bias=1e-7
    settings=FilterSettings(bias_correlation_s=tau,bias_random_walk_psd=0.,acceleration_noise_psd=0.)
    f=CausalOrbitFilter(free_model(),[1e6,0,0,0,0,0],np.eye(6),500,settings=settings)
    f.state[6]=bias;before=f.state.copy();p=f.covariance.copy()
    result=f.forecast(t);x=result['state'];decay=np.exp(-t/tau)
    assert x[6]==pytest.approx(bias*decay)
    assert x[3]==pytest.approx(bias*tau*(1-decay),abs=1e-12)
    assert x[0]-1e6==pytest.approx(bias*tau*(t-tau*(1-decay)),abs=1e-8)
    assert result['covariance'][6,6]==pytest.approx(p[6,6]*decay**2,rel=1e-10,abs=1e-25)
    np.testing.assert_array_equal(f.state,before)

def test_replayed_measurement_cannot_reduce_covariance_twice():
    initial=np.array([1e6,0,0,0,0,0.]);f=CausalOrbitFilter(free_model(),initial,np.eye(6),500)
    f.observe(initial,np.eye(6),0,0);p=f.covariance.copy()
    with pytest.raises(ValueError,match='Duplicate'):f.observe(initial,np.eye(6),0,10)
    np.testing.assert_array_equal(f.covariance,p)
    invalid=np.eye(9);invalid[6,6]=-1e-20
    with pytest.raises(ValueError):validate_covariance(invalid)
