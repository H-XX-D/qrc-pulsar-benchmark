#!/usr/bin/env python3
"""Continue the frozen matrix, retaining numerical exceptions as failed cases.

The original numerical implementation is unchanged. Exception tracebacks supply
already-computed partial histories for postmortem scoring only. No state is
reinitialized and no propagation is continued after a failed numerical check.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time
import traceback
import numpy as np
from artemis_resilience_campaign import REPO,read,save,sha,archive_runners
from artemis_anomaly_campaign import jobs,key,initialize
from qrc_pulsar_bench.anomaly_benchmark import AnomalyExperiment,score_recovery

EXP=None;ROOT=None
def init(root,data,kernels,physics,native):
    global EXP,ROOT
    ROOT=root;EXP=AnomalyExperiment(data,kernels,physics,native)

def capture_failure(exc,scenario,event,bits,coast):
    frame=None;tb=exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name=='run_anomaly':frame=tb.tb_frame
        tb=tb.tb_next
    if frame is None:raise exc
    local=frame.f_locals
    if not local.get('dense') or local.get('t',0)<event.onset_s:raise exc
    points=local['dense'];commands=list(local['records']);nav=local['nav'];t=float(local['t'])
    text=traceback.format_exc()
    stage='navigation' if 'adaptive_control.py' in text else 'simulation'
    # 512 is a harness terminal-exception annotation, not an onboard mode.
    commands.append(dict(time_s=t,dv=[0.,0.,0.],burn_s=0.,actual_burn_s=0.,status=512,mode='simulation_exception'))
    counts={str(s):sum(c['status']==s for c in commands) for s in {c['status'] for c in commands}}
    error=dict(type=type(exc).__name__,message=str(exc),stage=stage,time_s=t,traceback=text,
        covariance_diagonal=[float(v) if np.isfinite(v) else None for v in np.diag(nav.covariance)])
    return dict(scenario=asdict(scenario),anomaly=asdict(event),direction_icrf=local['direction'].tolist(),
        arithmetic=bits,control='coast_after_event' if coast else 'routine_mpc',
        initial_state=local['initial'].tolist(),model=local['inputs'],nominal_min_clearance_m=local['plan'].min_clearance_m,
        failure=error,metrics=dict(**score_recovery(points,event,completed=False),completed=False,collision=False,
            min_clearance_m=float(local['minimum_clearance']),issued_dv_m_s=float(sum(np.linalg.norm(c['dv']) for c in commands if c['burn_s']>0)),
            delivered_dv_m_s=float(local['actual_dv']) if stage=='navigation' else None,propellant_kg=500.-float(local['mass']),
            burns=sum(bool(c['burn_s']>0) for c in commands),actual_burns=sum(bool(c['actual_burn_s']>0) for c in commands),
            status_counts=counts,fixes_rejected=nav.rejected,fixes_accepted=nav.accepted,
            first_recovery_required_h=next((c['time_s']/3600 for c in commands if c['status']==64),None)),
        cpu_s=time.process_time()-local['start'],samples=local['samples'],dense_samples=points,commands=commands)

def run(job):
    scenario,event,bits,coast=job
    try:
        return EXP.run_anomaly(scenario,event,library=None if bits=='float64' else ROOT/'mpc32.so',coast_after_event=coast)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        return capture_failure(exc,scenario,event,bits,coast)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();root=a.output
    archive_runners(root,('artemis_anomaly_continuation.py',))
    baseline=read(root/'results.json');sources=read(root/'source-sha256.json');inputs=read(root/'input-sha256.json')
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise ValueError('Numerical source changed')
    paths=[a.data/f'ARTEMIS_P{p}_DEF_2009022_2012022.bsp' for p in (1,2)]
    paths += [a.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [a.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
    if {str(p.resolve()):sha(p) for p in paths}!=inputs:raise ValueError('Physical inputs changed')
    header=dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(root/'protocol.json'),
        base_results_sha256=sha(root/'results.json'),runner_sha256=sha(Path(__file__)),
        reason='Retain numerical exceptions as failed recovery; unchanged dynamics/controller/protocol; preserve all earlier successful rows')
    out=root/'continuation.json';report=read(out) if out.exists() else dict(**header,results=[])
    if any(report[k]!=v for k,v in header.items()):raise ValueError('Continuation provenance mismatch')
    done={key(r) for r in baseline['results']+report['results']}
    pending=[j for j in jobs() if (j[0].name,j[1].name,j[2],'coast_after_event' if j[3] else 'routine_mpc') not in done]
    print('Retained',len(done),'remaining',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(root,a.data,a.kernels,a.physics,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run,j):j for j in pending}
        for f in as_completed(futures):
            row=f.result();json.dumps(row,allow_nan=False);report['results'].append(row);report['results'].sort(key=key);save(out,report)
            print(*key(row),row['metrics'],row.get('failure',{}).get('message'),flush=True)
    if len(baseline['results'])+len(report['results'])!=len(jobs()):raise ValueError('Incomplete combined matrix')
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise ValueError('Source changed during continuation')

if __name__=='__main__':main()
