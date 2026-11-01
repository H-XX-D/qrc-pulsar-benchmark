from dataclasses import replace
import json
import numpy as np
import pytest
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,score_recovery,AnomalyExperiment
from qrc_pulsar_bench.resilience_benchmark import Scenario
from qrc_pulsar_bench.adaptive_control import MoonCartesianFlightModel
from qrc_pulsar_bench.jacobi_control import Actuator

def points():
    return [dict(time_s=float(t),position_error_m=20.,velocity_error_m_s=.001) for t in range(0,86401,300)]

def test_recovery_requires_position_velocity_and_sustained_final_hold():
    event=Anomaly('kick',43200,43200,kick_m_s=.01);p=points()
    p[150]['position_error_m']=500
    result=score_recovery(p,event,completed=True)
    assert result['response']=='recovered' and result['settling_after_onset_h']==pytest.approx(.5833333333)
    p[-2]['velocity_error_m_s']=.03
    assert not score_recovery(p,event,completed=True)['settled']
    assert not score_recovery(points(),event,completed=False)['settled']

def test_transient_end_and_persistent_regulation_have_distinct_scoring():
    event=Anomaly('step',43200,86400,acceleration_m_s2=1e-7)
    assert not score_recovery(points(),event,completed=True)['settled']
    assert score_recovery(points(),replace(event,persistent=True),completed=True)['response']=='contained'
    with pytest.raises(ValueError):Anomaly('bad',43201,50000)

def test_hidden_impulse_cannot_enter_controller_before_delayed_fix(monkeypatch):
    import qrc_pulsar_bench.anomaly_benchmark as module
    seen=[]
    class Spy:
        def __init__(self,*args,**kwargs):
            self.settings=type('Settings',(),dict(interval_s=1800.))();self.actuator=Actuator();self.last_status=0;self.mode='fresh'
        def command(self,t,estimate,mass):
            assert set(estimate)=={'state','covariance','mass','time_s','last_observation_s'}
            assert mass==estimate['mass']
            seen.append((t,estimate['state'].copy()));return np.array([.001,0,0])
    monkeypatch.setattr(module,'RoutineMPC',Spy)
    model=MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),lambda t:(np.array([1e8,0,0]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=1800.)
    initial=np.array([1e7,0,0,0,1.,0]);plan=model.propagate(initial,0,86400,500.)
    exp=AnomalyExperiment('.','.','.','.')
    monkeypatch.setattr(exp,'prepare',lambda s:(initial,model,plan,{}))
    scenario=Scenario('unit',1,'unused',1.,42,'unit',pulses=())
    none=Anomaly('none',43200,43200)
    baseline=exp.run_anomaly(scenario,none);baseline_seen=seen.copy();seen.clear()
    fault=exp.run_anomaly(scenario,replace(none,name='kick',kick_m_s=.005))
    assert baseline['metrics']['burns']==48 and baseline['metrics']['actual_burns']==48
    json.dumps(baseline,allow_nan=False);json.dumps(fault,allow_nan=False)
    for (t,a),(u,b) in zip(baseline_seen,seen):
        if t<=43200:np.testing.assert_array_equal(a,b)
    # The plant jump is visible in truth immediately, but not the delayed sensor.
    a=next(p for p in baseline['dense_samples'] if p['time_s']==43200)
    b=next(p for p in fault['dense_samples'] if p['time_s']==43200)
    assert a['position_error_m']==b['position_error_m']
    assert a['velocity_error_m_s']!=b['velocity_error_m_s']
    assert any(not np.array_equal(a,b) for (t,a),(u,b) in zip(baseline_seen,seen) if t>43200)
