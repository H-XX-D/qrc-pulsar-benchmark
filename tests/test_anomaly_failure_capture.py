import importlib
import json
from pathlib import Path
import numpy as np
from qrc_pulsar_bench.adaptive_control import CausalOrbitFilter,MoonCartesianFlightModel
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,AnomalyExperiment
from qrc_pulsar_bench.resilience_benchmark import Scenario


def test_numerical_exception_preserves_partial_evidence_without_resuming(monkeypatch):
    import qrc_pulsar_bench.anomaly_benchmark as module
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    capture=importlib.import_module('artemis_anomaly_continuation').capture_failure
    class BrokenFilter(CausalOrbitFilter):
        @property
        def covariance(self):
            p=super().covariance
            if getattr(self,'inject_failure',False):p[6,6]=-1e-20
            return p
        def observe(self,fix,covariance,measurement_s,available_s,commands=()):
            if available_s>=43200:
                self.inject_failure=True
                self._check_covariance(self.covariance)
            return super().observe(fix,covariance,measurement_s,available_s,commands)
    monkeypatch.setattr(module,'CausalOrbitFilter',BrokenFilter)
    model=MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),lambda t:(np.array([1e8,0,0]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=1800.)
    initial=np.array([1e7,0,0,0,1.,0]);plan=model.propagate(initial,0,86400,500.)
    experiment=AnomalyExperiment('.','.','.','.')
    monkeypatch.setattr(experiment,'prepare',lambda s:(initial,model,plan,{}))
    scenario=Scenario('unit',1,'unused',1.,42,'unit',pulses=());event=Anomaly('kick',43200,43200,kick_m_s=.005)
    try:experiment.run_anomaly(scenario,event)
    except ValueError as exc:row=capture(exc,scenario,event,'float64',False)
    else:raise AssertionError('Injected covariance failure was swallowed')
    assert row['failure']['time_s']==43200 and row['failure']['message']=='Negative variance'
    assert row['failure']['covariance_diagonal'][6]<0
    assert not row['metrics']['completed'] and row['metrics']['response']=='not_recovered'
    assert row['dense_samples'][-1]['time_s']==43200 and row['commands'][-1]['status']==512
    json.dumps(row,allow_nan=False)
