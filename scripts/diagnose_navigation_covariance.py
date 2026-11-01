#!/usr/bin/env python3
"""Replay the archived P2 jump with passive covariance instrumentation."""
import argparse
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
from qrc_pulsar_bench import anomaly_benchmark as benchmark
from qrc_pulsar_bench.resilience_benchmark import Scenario


def stats(p):
    scale=np.sqrt(np.maximum(np.abs(np.diag(p)),1e-300))
    normalized=p/np.outer(scale,scale)
    return dict(diagonal=np.diag(p).tolist(),correlation_eigen_min=float(np.linalg.eigvalsh((normalized+normalized.T)/2)[0]),
                symmetry=float(np.max(np.abs(normalized-normalized.T))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=args.archive/'source/qrc_pulsar_bench/adaptive_control.py'
    spec=importlib.util.spec_from_file_location('qrc_pulsar_bench._archived_filter',source)
    legacy=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    original_step=legacy.covariance_step;steps=[];predictions=[];bad_steps=[]
    def step(a,spectral,dt):
        phi,q=original_step(a,spectral,dt)
        item=dict(dt=dt,q=stats(q),phi_max=float(np.max(np.abs(phi))),a_eigen_max=float(np.max(np.abs(np.linalg.eigvals(a)))))
        steps.append(item)
        if len(steps)>50:steps.pop(0)
        if item['q']['correlation_eigen_min'] < -1e-10 or min(item['q']['diagonal'])<0:
            bad_steps.append(item)
            if len(bad_steps)==1:np.savez(args.output/'first_bad_process_noise.npz',a=a,spectral=spectral,dt=dt,phi=phi,q=q)
        return phi,q
    legacy.covariance_step=step
    class Instrumented(legacy.CausalOrbitFilter):
        def _predict(self,t,commands,as_of_s):
            state,p,mass=super()._predict(t,commands,as_of_s)
            item=dict(t=float(t),start=float(self.time_s),p=stats(p),prior=stats(self.covariance))
            predictions.append(item)
            if len(predictions)>50:predictions.pop(0)
            if item['p']['correlation_eigen_min'] < -1e-10 or min(item['p']['diagonal'])<0:
                path=args.output/'first_bad_prediction.npz'
                if not path.exists():np.savez(path,state=state,p=p,prior=self.covariance,t=t,start=self.time_s)
            self.diagnostic_prediction=(state.copy(),p.copy(),mass)
            return state,p,mass
        def observe(self,z,r,t,available_s,commands=()):
            try:return super().observe(z,r,t,available_s,commands)
            except Exception:
                state,p,mass=self.diagnostic_prediction
                np.savez(args.output/'failed_update.npz',predicted=state,p=p,r=r,z=z,posterior=self.covariance,state=self.state,t=t)
                raise
    benchmark.CausalOrbitFilter=Instrumented
    home=Path('/Users/hendrixx.')
    experiment=benchmark.AnomalyExperiment(home/'qrc-mpc-strategies-2026-10-31/data',home/'qrc-cislunar-data-2026-10-30',home/'qrc-force-followup-2026-10-31',home/'qrc-force-followup-2026-10-31/liblunar.so')
    scenario=Scenario('P2_jan20_7days',2,'2012-01-20T00:00:00',7.,11002,'covariance_diagnosis',pulses=())
    result={}
    try:
        row=experiment.run_anomaly(scenario,benchmark.Anomaly('kick_50cm_s',172800.,172800.,kick_m_s=.5))
        result['metrics']=row['metrics']
    except Exception as error:
        result.update(exception=str(error),traceback=traceback.format_exc())
    result.update(steps=steps,predictions=predictions,bad_process_steps=bad_steps,archived_source=str(source))
    (args.output/'diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','predictions','bad_process_steps')},indent=2),flush=True)
    print('Bad process steps',len(bad_steps),flush=True)


if __name__=='__main__':main()
