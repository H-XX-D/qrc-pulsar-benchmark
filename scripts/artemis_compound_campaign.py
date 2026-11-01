#!/usr/bin/env python3
"""Freeze20 new single/concurrent/sequential fault schedules and execute comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import shutil
import traceback
from artemis_resilience_campaign import REPO,read,save,sha
from qrc_pulsar_bench.compound_benchmark import CompoundExperiment
from qrc_pulsar_bench.compound_faults import FaultCase,suite
from qrc_pulsar_bench.resilience_benchmark import Scenario
from qrc_pulsar_bench.fpga_mpc import compile_core

EXP=None;ROOT=None


def jobs():
    result=[]
    for probe,utc,seed in [(1,'2012-01-15T00:00:00',14001),(2,'2012-01-20T00:00:00',14002)]:
        scenario=Scenario(f'P{probe}_compound',probe,utc,3.,seed,'compound_holdout',pulses=())
        clean=FaultCase('none','baseline','No faults',())
        result.extend((scenario,clean,b,False) for b in ('float64','32'))
        result.extend((scenario,case,b,c) for case in suite()
                      for b,c in [('float64',False),('32',False),('float64',True)])
    return result


def key(row):return row['scenario']['name'],row['case']['name'],row['arithmetic'],row['control']


def initialize(root,data,kernels,physics,native):
    global ROOT,EXP
    ROOT=root;EXP=CompoundExperiment(data,kernels,physics,native)


def run(job):
    scenario,case,bits,coast=job
    stem=f'{scenario.name}-{case.name}-{bits}-'+('coast' if coast else 'mpc')
    folder=ROOT/'journals';folder.mkdir(exist_ok=True)
    attempt=len(list(folder.glob(stem+'.attempt-*.jsonl')))+1
    journal=folder/f'{stem}.attempt-{attempt}.jsonl'
    try:
        return EXP.run(scenario,case,library=None if bits=='float64' else ROOT/'mpc32.so',
                       coast_after_event=coast,journal_path=journal)
    except Exception as error:
        return dict(scenario=asdict(scenario),case=asdict(case),arithmetic=bits,
            control='coast_after_event' if coast else 'routine_mpc',
            metrics=dict(completed=False,response='not_run'),
            failure=dict(message=str(error),traceback=traceback.format_exc(),stage='setup_or_harness'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','kernels','physics','native'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    protocol=dict(jobs=[dict(scenario=asdict(s),case=asdict(c),arithmetic=b,coast=k) for s,c,b,k in jobs()],
        schedules='20 new schedules:10 individual,5 simultaneous combinations,5 sequences; independent fixed event directions.',
        flight='Frozen square-root+ManeuverAwareOrbitFilter V3, NavigationIngress, RoutineMPC. No rearm/retuning or fault metadata.',
        physical='Single real ARTEMIS initial. Degree60 lunar gravity and existing forces. Later motion/fixes synthetic.',
        causality='Fault timing/direction/identity only plant/sensor/scorer; communications sees link availability. No truth mass in control.',
        propulsion='Exact segment boundaries, vector sum of overlapping propulsion, summed mass flow. Leaks consume propellant.',
        telemetry='Append-only local outbox and simulated deduplicating receiver. Ground blackout independent from sensor dropout. No external messages.',
        comparisons='Matched clean controls, float/32 MPC and coast after first event. Coasting is a scored counterfactual.',
        scoring='Report all cases; legacy100m/.02m/s final6h trajectory criterion; separately200m/.02m/s final6h trusted navigation, clearance, fuel and telemetry.',
        limits='Three days per run. No power/thermal/attitude plant, real sensor observables, actual radio or synthesized FPGA. No universal survival/optimality claim.')
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))
    files += [REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    files += [REPO/'scripts'/n for n in ('artemis_compound_campaign.py','artemis_resilience_campaign.py')]
    files += [REPO/'tests/test_compound_failures.py',REPO/'tests/test_navigation_recovery.py']
    sources={str(f.relative_to(REPO)):sha(f) for f in files}
    paths=[args.data/f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp' for probe in (1,2)]
    paths += [args.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [args.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[args.native]
    inputs={str(path.resolve()):sha(path) for path in paths}
    for name,value in [('protocol.json',protocol),('source-sha256.json',sources),('input-sha256.json',inputs)]:
        value=json.loads(json.dumps(value))
        if (root/name).exists() and read(root/name)!=value:raise ValueError(f'Frozen {name} changed')
        save(root/name,value)
    for file in files:
        target=root/'source'/file.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(file,target)
        if sha(file)!=sha(target):raise ValueError('Source archive changed')
    compile_core(root/'mpc32.so',32,24)
    header=dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(root/'protocol.json'))
    out=root/'results.json';report=read(out) if out.exists() else dict(**header,results=[])
    if any(report[k]!=v for k,v in header.items()):raise ValueError('Resume provenance changed')
    done={key(row) for row in report['results']}
    pending=[j for j in jobs() if (j[0].name,j[1].name,j[2],'coast_after_event' if j[3] else 'routine_mpc') not in done]
    print('Frozen',len(jobs()),'pending',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(root,args.data,args.kernels,args.physics,args.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run,job) for job in pending]):
            row=future.result();json.dumps(row,allow_nan=False)
            report['results'].append(row);report['results'].sort(key=key);save(out,report)
            print(*key(row),json.dumps(row['metrics']),row.get('failure',{}).get('message',''),flush=True)
    if len(report['results'])!=len(jobs()):raise ValueError('Incomplete matrix')
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise ValueError('Sources changed during execution')


if __name__=='__main__':main()
