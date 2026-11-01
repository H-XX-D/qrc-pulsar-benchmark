#!/usr/bin/env python3
"""Freeze unseen plant/sensor faults and test automatic recovery without retuning."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import shutil
from artemis_resilience_campaign import sha,read,save,REPO,archive_runners
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,AnomalyExperiment
from qrc_pulsar_bench.resilience_benchmark import Scenario,DAY
from qrc_pulsar_bench.fpga_mpc import compile_core

EXP=None;ROOT=None
def cases():
    geometries=[Scenario('P1_jan15_3days',1,'2012-01-15T00:00:00',3.,11001,'anomaly_evaluation',pulses=()),
                Scenario('P2_jan20_7days',2,'2012-01-20T00:00:00',7.,11002,'anomaly_evaluation',pulses=())]
    result=[]
    for scenario in geometries:
        onset=DAY if scenario.probe==1 else 2*DAY;stop=scenario.days*DAY
        events=[Anomaly('none',onset,onset),
            Anomaly('kick_5mm_s',onset,onset,kick_m_s=.005),
            Anomaly('kick_5cm_s',onset,onset,kick_m_s=.05),
            Anomaly('kick_50cm_s',onset,onset,kick_m_s=.5),
            Anomaly('acceleration_pulse',onset,onset+6*3600,acceleration_m_s2=2e-6),
            Anomaly('persistent_force_error',onset,stop,acceleration_m_s2=3e-7,persistent=True),
            Anomaly('half_thrust_24h',onset,onset+DAY,thrust_gain=.5),
            Anomaly('navigation_bias_6h',onset,onset+6*3600,sensor_position_bias_m=150.,sensor_velocity_bias_m_s=.006),
            Anomaly('combined_fault',onset,onset+12*3600,kick_m_s=.02,thrust_gain=.75,outage_s=6*3600),
            Anomaly('permanent_thrust_failure',onset,stop,thrust_gain=0.,persistent=True)]
        result.extend((scenario,event) for event in events)
    return result

def key(row):return row['scenario']['name'],row['anomaly']['name'],row['arithmetic'],row['control']
def initialize(root,data,kernels,physics,native):
    global EXP,ROOT
    ROOT=root;EXP=AnomalyExperiment(data,kernels,physics,native)
def run(job):
    scenario,event,bits,coast=job
    return EXP.run_anomaly(scenario,event,library=None if bits=='float64' else ROOT/'mpc32.so',coast_after_event=coast)
def jobs():
    return [(s,e,b,c) for s,e in cases() for b,c in
            ([('float64',False),('32',False)] if e.name=='none' else [('float64',False),('32',False),('float64',True)])]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    archive_runners(root,('artemis_resilience_campaign.py','artemis_anomaly_campaign.py'))
    protocol=dict(cases=[dict(scenario=asdict(s),anomaly=asdict(e)) for s,e in cases()],
        controller='Frozen RoutineMPC sparse_05; no automatic rearm, controller retuning or fault metadata',
        recovery='After transient ends (or persistent onset): <=100m position and <=0.02m/s velocity for >=6h continuously through end; sampled every300s',
        comparisons='Identical initial conditions/noise per geometry: no anomaly, closed-loop anomaly float/32, corrections disabled from event onset float',
        reference='Single reconstructed ARTEMIS SPK initial state and nominal model. Later vehicle motion/fixes synthetic. Require reference clearance >=50km.',
        boundaries='Impulse plant only, then delayed pre-event fix. Force discontinuities at exact integration boundaries. Bias tied to measurement epoch.',
        actuator='Fault gain multiplies actual thrust and flow; navigation knows only issued commands. Controller receives predicted mass, never truth mass.',
        scoring='Report all cases including latched recovery, failures, actual clearance, commanded/delivered delta-v and propellant. No global survival guarantee.')
    protocol=json.loads(json.dumps(protocol))
    pp=root/'protocol.json'
    if pp.exists() and read(pp)!=protocol:raise ValueError('Frozen protocol changed')
    if not pp.exists():save(pp,protocol)
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+[REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    sources={str(f.relative_to(REPO)):sha(f) for f in files};sf=root/'source-sha256.json'
    if sf.exists() and read(sf)!=sources:raise ValueError('Frozen numerical sources changed')
    if not sf.exists():
        save(sf,sources)
        for f in files:
            target=root/'source'/f.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    paths=[a.data/f'ARTEMIS_P{p}_DEF_2009022_2012022.bsp' for p in (1,2)]
    paths += [a.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [a.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
    inputs={str(p.resolve()):sha(p) for p in paths};ip=root/'input-sha256.json'
    if ip.exists() and read(ip)!=inputs:raise ValueError('Inputs changed')
    save(ip,inputs);compile_core(root/'mpc32.so',32,24)
    header=dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(pp),runner_sha256=sha(Path(__file__)))
    out=root/'results.json';report=read(out) if out.exists() else dict(**header,results=[])
    if any(report[k]!=v for k,v in header.items()):raise ValueError('Resume provenance mismatch')
    done={key(r) for r in report['results']}
    pending=[j for j in jobs() if (j[0].name,j[1].name,j[2],'coast_after_event' if j[3] else 'routine_mpc') not in done]
    print('Frozen jobs',len(jobs()),'remaining',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.data,a.kernels,a.physics,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run,j):j for j in pending}
        try:
            for f in as_completed(futures):
                row=f.result();json.dumps(row,allow_nan=False)
                report['results'].append(row);report['results'].sort(key=key);save(out,report)
                print(*key(row),row['metrics'],flush=True)
        except BaseException:
            for future in futures:future.cancel()
            # Preserve the original error immediately; context exit may still
            # wait for the few jobs already running, but not the queued matrix.
            import traceback
            traceback.print_exc()
            raise
    if len(report['results'])!=len(jobs()) or len({key(r) for r in report['results']})!=len(jobs()):raise ValueError('Incomplete matrix')
    if sources!={str(f.relative_to(REPO)):sha(f) for f in files}:raise ValueError('Sources changed during execution')

if __name__=='__main__':main()
