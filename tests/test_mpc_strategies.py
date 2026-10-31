from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from qrc_pulsar_bench.mpc_strategies import Strategy,StrategyMPC,strategies,select_strategies

class Plan:
    stop_s=100000.
    def at(self,t):return np.zeros(7)
class Model:
    def to_moon(self,x,t):return x
    def acceleration(self,r,v,t):return np.zeros(3)
    def propagate(self,*a,**kw):return SimpleNamespace(completed=True,min_clearance_m=100000.)
def estimate(error=0,bias=0):
    return dict(state=np.array([error,0,0,0,0,0,bias,0,0],float),covariance=np.eye(9)*1e-12)

def test_slew_rows_encode_previous_command_and_future_differences():
    spec=replace(Strategy(),slew_scale_m_s=.03);c=StrategyMPC(Model(),Plan(),spec)
    c.previous=np.array([.003,0,0])
    m,y,cap,active=c.extended_problem(0,np.zeros(6),500,np.zeros(4),np.zeros((4,3)))
    u=np.tile(c.previous/cap,4)
    np.testing.assert_allclose((m@u-y)[-12:],0,atol=1e-15)

def test_dwell_removes_future_decisions_and_blocks_early_burn():
    c=StrategyMPC(Model(),Plan(),strategies()['spaced_1h'])
    _,_,_,active=c.extended_problem(0,np.zeros(6),500,np.zeros(4),np.zeros((4,3)))
    np.testing.assert_array_equal(active,[0,1,2,6,7,8])
    c.last_burn=0
    assert not c.command(1800,estimate(20),500).any()
    assert c.diagnostics[-1]['reason']=='minimum_spacing'

def test_event_coasts_at_small_error_but_tighter_corridor_requests_burn():
    coast=StrategyMPC(Model(),Plan(),replace(Strategy(),trigger_m=60.))
    assert not coast.command(0,estimate(20),500).any()
    fire=StrategyMPC(Model(),Plan(),replace(Strategy(),trigger_m=10.))
    assert fire.command(0,estimate(20),500)[0]<0
    assert fire.last_status==0

def test_bias_prediction_uses_estimate_and_has_correct_acceleration_units():
    c=StrategyMPC(Model(),Plan(),replace(Strategy(),use_bias=True))
    c.bias=np.array([1e-7,0,0])
    m,y,cap,active=c.extended_problem(0,np.zeros(6),500,np.zeros(4),np.zeros((4,3)))
    assert y[0]==pytest.approx(-.5*1e-7*1800**2/100)
    assert y[3]==pytest.approx(-1e-7*1800/.01)

def test_sparse_reweight_penalizes_zero_groups_more_strongly():
    c=StrategyMPC(Model(),Plan(),strategies()['sparse_01'])
    z=np.zeros((4,3));p=z.copy();p[0]=[.05,0,0]
    m0,_,_,_=c.extended_problem(0,np.zeros(6),500,np.zeros(4),z)
    m1,_,_,_=c.extended_problem(0,np.zeros(6),500,np.zeros(4),p)
    assert m1[-12,0]<m0[-12,0]
    np.testing.assert_array_equal(m1[-9:,3:],m0[-9:,3:])

def test_stale_input_cannot_schedule_burn_and_clears_previous_slot():
    c=StrategyMPC(Model(),Plan(),strategies()['smooth_003']);c.previous[:]=.01
    assert not c.command(0,estimate(1000),500,valid=False).any()
    assert c.last_status==2 and not c.previous.any()

def test_selection_rejects_holdout_and_duplicate_training_rows():
    rows=[]
    for name in strategies():
        for case in ('P1_sep','P2_sep'):
            for seed in (4101,4102):
                rows.append(dict(case=case,seed=seed,configuration=name,arithmetic='float64',split='training',commands=[],
                    dense_steady_rms_m=20.,dense_steady_max_m=50.,dv_m_s=.4,burns=100,core_status_counts={'0':144},collision=False))
    assert select_strategies(rows)['primary'] is not None
    rows[0]['split']='fresh_evaluation'
    with pytest.raises(ValueError):select_strategies(rows)
    rows[0]['split']='training'
    with pytest.raises(ValueError):select_strategies(rows+[rows[0]])
