#!/usr/bin/env python3
"""Freeze and evaluate square-root navigation and causal jump reacquisition."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import shutil
import traceback
from artemis_resilience_campaign import REPO,sha,read,save
from artemis_anomaly_continuation import capture_failure
from qrc_pulsar_bench.adaptive_control import CausalOrbitFilter,FilterSettings
from qrc_pulsar_bench.navigation_recovery import ManeuverAwareOrbitFilter,ReacquisitionSettings
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,AnomalyExperiment
from qrc_pulsar_bench.resilience_benchmark import Scenario,DAY
from qrc_pulsar_bench.fpga_mpc import compile_core

EXP=None;ROOT=None


def jobs():
    result=[]
    for phase,geometries in [
        ('regression',[(1,'2012-01-15T00:00:00',3.,11001),(2,'2012-01-20T00:00:00',7.,11002)]),
        ('fresh',[(1,'2012-01-16T00:00:00',5.,12001),(2,'2012-01-19T00:00:00',7.,12002)])]:
        for probe,utc,days,seed in geometries:
            s=Scenario(f'P{probe}_{phase}',probe,utc,days,seed,phase,pulses=())
            onset=DAY if probe==1 else 2*DAY
            events=[Anomaly('none',onset,onset)]
            if phase=='regression':
                events += [Anomaly('kick_50cm_s',onset,onset,kick_m_s=.5),
                    Anomaly('position_velocity_bias_6h',onset,onset+21600,sensor_position_bias_m=150,sensor_velocity_bias_m_s=.006),
                    Anomaly('velocity_bias_6h',onset,onset+21600,sensor_velocity_bias_m_s=.5),
                    Anomaly('one_bad_fix',onset,onset+1800,sensor_position_bias_m=3000,sensor_velocity_bias_m_s=.5)]
            else:
                events += [Anomaly(f'kick_{m}m_s',onset,onset,kick_m_s=m) for m in (.2,.75,1.25)]
                events += [Anomaly('kick_with_outage',onset,onset+10800,kick_m_s=.2,outage_s=10800),
                    Anomaly('velocity_bias_6h',onset,onset+21600,sensor_velocity_bias_m_s=.75),
                    Anomaly('position_velocity_bias_6h',onset,onset+21600,sensor_position_bias_m=250,sensor_velocity_bias_m_s=.01)]
            for event in events:
                result.extend((s,event,bits,'maneuver_aware') for bits in ('float64','32'))
            if phase=='regression' and probe==2:
                result.append((s,Anomaly('kick_50cm_s',onset,onset,kick_m_s=.5),'float64','square_root_only'))
    # New noise/direction instances declared after the V2 observation-gap miss.
    # They use eligible known geometries, not new historical flight datasets.
    for probe,utc,days,seed in [(1,'2012-01-15T00:00:00',3.,13001),(2,'2012-01-20T00:00:00',7.,13002)]:
        scenario=Scenario(f'P{probe}_gap_holdout',probe,utc,days,seed,'gap_holdout',pulses=())
        onset=DAY if probe==1 else 2*DAY
        events=[Anomaly('none',onset,onset),
            Anomaly('kick_with_6h_outage',onset,onset+21600,kick_m_s=.35,outage_s=21600),
            Anomaly('bad_velocity_after_outage',onset,onset+32400,sensor_velocity_bias_m_s=.6,outage_s=10800)]
        result.extend((scenario,event,bits,'maneuver_aware') for event in events for bits in ('float64','32'))
    return result


def key(row):return row['scenario']['name'],row['anomaly']['name'],row['arithmetic'],row['navigation']


def initialize(root,data,kernels,physics,native):
    global EXP,ROOT
    ROOT=root;EXP=AnomalyExperiment(data,kernels,physics,native)


def run(job):
    scenario,event,bits,navigation=job
    try:
        row=EXP.run_anomaly(scenario,event,library=None if bits=='float64' else ROOT/'mpc32.so',
            filter_factory=ManeuverAwareOrbitFilter if navigation=='maneuver_aware' else CausalOrbitFilter)
    except Exception as error:
        try:row=capture_failure(error,scenario,event,bits,False)
        except Exception:
            row=dict(scenario=asdict(scenario),anomaly=asdict(event),arithmetic=bits,
                failure=dict(message=str(error),traceback=traceback.format_exc()),
                metrics=dict(completed=False,response='not_run',navigation_reacquisitions=0))
    row['navigation']=navigation
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--phase',choices=('all','regression','fresh'),default='all')
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    protocol=dict(jobs=[dict(scenario=asdict(s),anomaly=asdict(e),arithmetic=b,navigation=n) for s,e,b,n in jobs()],
        filter=asdict(FilterSettings()),reacquisition=asdict(ReacquisitionSettings()),
        scoring='Original100m/0.02m/s final6h corridor; separately report navigation recovery and authority latches.',
        causality='One real SPK initial; later spacecraft motion/fixes synthetic. Fault metadata is plant/scorer only.',
        comparison='Square-root-only ablation replays failed P2 geometry. Regression and fresh instances predeclared.',
        limits='No routine authority extension/rearm. Three fixes do not establish physical origin against coherent sensor spoofing.')
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))
    files += [REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    files += [REPO/'scripts'/n for n in ('artemis_navigation_campaign.py','artemis_anomaly_continuation.py','artemis_resilience_campaign.py')]
    files += list((REPO/'tests').glob('test*navigation*py'))+[REPO/'tests/test_anomaly_failure_capture.py']
    sources={str(f.relative_to(REPO)):sha(f) for f in files}
    paths=[a.data/f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp' for probe in (1,2)]
    paths += [a.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [a.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
    inputs={str(path.resolve()):sha(path) for path in paths}
    for name,value in [('protocol.json',protocol),('source-sha256.json',sources),('input-sha256.json',inputs)]:
        value=json.loads(json.dumps(value))
        if (root/name).exists() and read(root/name)!=value:raise ValueError(f'Frozen {name} changed')
        save(root/name,value)
    for f in files:
        target=root/'source'/f.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(f,target)
        if sha(target)!=sha(f):raise ValueError('Archive changed')
    compile_core(root/'mpc32.so',32,24)
    header=dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(root/'protocol.json'))
    output=root/'results.json';report=read(output) if output.exists() else dict(**header,results=[])
    if any(report[k]!=v for k,v in header.items()):raise ValueError('Resume provenance changed')
    done={key(row) for row in report['results']}
    pending=[j for j in jobs() if (a.phase=='all' or j[0].split==a.phase)
             and (j[0].name,j[1].name,j[2],j[3]) not in done]
    print('Frozen',len(jobs()),'pending',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,
            initargs=(root,a.data,a.kernels,a.physics,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run,job) for job in pending]):
            row=future.result();json.dumps(row,allow_nan=False)
            report['results'].append(row);report['results'].sort(key=key);save(output,report)
            print(*key(row),json.dumps(row['metrics']),row.get('failure',{}).get('message',''),flush=True)
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise ValueError('Source changed during campaign')


if __name__=='__main__':main()
