#!/usr/bin/env python3
"""Frozen regression and fresh holdout tests of five layered autonomy repairs."""
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
from qrc_pulsar_bench.compound_faults import Fault,FaultCase,suite,HOUR
from qrc_pulsar_bench.resilience_benchmark import Scenario
from qrc_pulsar_bench.fpga_mpc import compile_core

EXP=None;ROOT=None


def fresh():
    def f(kind,a,b,v=0):return Fault(kind,a*HOUR,b*HOUR,v)
    return [
        FaultCase('21_noisy_outage_then_kick','holdout','New noise/outage timing followed by .25m/s kick',
            (f('sensor_dropout',12,16),f('sensor_noise',16,22,5),f('impulse',28,28,.25))),
        FaultCase('22_second_kick_in_blackout','holdout','Noisy fixes followed by two kicks during ground blackout',
            (f('sensor_noise',12,20,4),f('impulse',24,24,.3),f('impulse',36,36,.15),f('ground_blackout',18,54))),
        FaultCase('23_long_stuck_thruster','holdout','Twelve-hour .15N leak with six-hour navigation outage',
            (f('uncommanded_thrust',18,30,.15),f('sensor_dropout',18,24),f('ground_blackout',12,60))),
        FaultCase('24_failed_isolation','holdout','Stuck thrust with unavailable isolation hardware',
            (f('uncommanded_thrust',12,18,.2),f('sensor_dropout',12,15),f('ground_blackout',12,60))),
        FaultCase('25_ground_authority','holdout','Queued ground commands during blackout and replayed navigation',
            (f('ground_blackout',12,60),f('sensor_replay',30,36))),
        FaultCase('26_beyond_routine_envelope','holdout','.65m/s kick during ground blackout',
            (f('impulse',24,24,.65),f('ground_blackout',12,60))),
    ]


def ground_schedule():
    return [dict(sequence=1,issued_s=6*HOUR,delivery_s=6*HOUR,expires_s=7*HOUR,action='hold_mission'),
        dict(sequence=2,issued_s=8*HOUR,delivery_s=8*HOUR,expires_s=9*HOUR,action='resume_mission',bad_signature=True),
        dict(sequence=1,issued_s=9*HOUR,delivery_s=9*HOUR,expires_s=10*HOUR,action='resume_mission'),
        dict(sequence=2,issued_s=24*HOUR,delivery_s=24*HOUR,expires_s=36*HOUR,action='resume_mission'),
        dict(sequence=3,issued_s=61*HOUR,delivery_s=61*HOUR,expires_s=65*HOUR,action='resume_mission')]


def jobs(development=False):
    result=[]
    for probe,utc in ((1,'2012-01-15T00:00:00'),(2,'2012-01-20T00:00:00')):
        regression=Scenario(f'P{probe}_compound',probe,utc,3.,14000+probe,'regression',pulses=())
        cases=[FaultCase('none','baseline','No faults',()),*suite()]
        if development:cases=[c for c in cases if c.name in ('none','15_stuck_thrust_blackout','17_force_bias_then_actuator','18_bad_sensor_return_then_kick')]
        result.extend((regression,case,bits,True,True,[]) for case in cases for bits in (('float64',) if development else ('float64','32')))
        if not development:
            scenario=Scenario(f'P{probe}_fresh',probe,utc,3.,15000+probe,'holdout',pulses=())
            for case in fresh():
                commands=ground_schedule() if case.name=='25_ground_authority' else []
                result.extend((scenario,case,bits,True,case.name!='24_failed_isolation',commands) for bits in ('float64','32'))
                # Same ground authority is meaningful only for the layered stack.
                if not commands:result.append((scenario,case,'float64',False,False,[]))
    return result


def key(row):return row['scenario']['name'],row['case']['name'],row['arithmetic'],row['control']


def initialize(root,data,kernels,physics,native):
    global ROOT,EXP
    ROOT=root;EXP=CompoundExperiment(data,kernels,physics,native)


def run(job):
    scenario,case,bits,layered,isolation,commands=job
    control='layered_mpc' if layered else 'routine_mpc'
    stem=f'{scenario.name}-{case.name}-{bits}-{control}'
    folder=ROOT/'journals';folder.mkdir(exist_ok=True)
    attempt=len(list(folder.glob(stem+'.attempt-*.jsonl')))+1
    try:
        return EXP.run(scenario,case,library=None if bits=='float64' else ROOT/'mpc32.so',layered=layered,
            isolation_available=isolation,ground_schedule=commands,journal_path=folder/f'{stem}.attempt-{attempt}.jsonl')
    except Exception as error:
        return dict(scenario=asdict(scenario),case=asdict(case),arithmetic=bits,control=control,
            metrics=dict(completed=False,response='not_run'),failure=dict(message=str(error),traceback=traceback.format_exc()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','kernels','physics','native'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--development',action='store_true');parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    work=jobs(args.development)
    protocol=dict(jobs=[dict(scenario=asdict(s),case=asdict(c),arithmetic=b,layered=l,isolation_available=i,ground_commands=g) for s,c,b,l,i,g in work],
        mode='development' if args.development else 'frozen_regression_and_holdout',
        physical='Single real ARTEMIS initial state; subsequent vehicle truth and direct-state fixes simulated. Existing force model.',
        sensors='NEW assumed independent thrust and mass-flow telemetry every60s, noise1mN and1e-7kg/s. No fault labels enter flight interfaces.',
        isolation='Primary shutoff acts at second corroborated unexpected-thrust sample. Backup enabled after two closed/quiet reports. Failed isolation explicitly tested.',
        planner='Coast first if13 initial uncertainty points at6 sigma pass6h clearance/reserve screen; otherwise bounded finite-burn candidates minimize effort.1m/s rolling shared budget. Diagnostic screen,not certified reachability.',
        ground='Preprovisioned HMAC test authority; held/replayed/expired/invalid commands tested. No external transmission. Ground cannot bypass local interlocks.',
        limits='Host simulation. OnlyQP runs integer C++;no FPGA synthesis,real radio,attitude/power/thermal model,calibrated uncertainty guarantee or universal survival claim.')
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+list((REPO/'tests').glob('test_*.py'))
    files+=list((REPO/'hardware').glob('*.cpp'))+list((REPO/'hardware').glob('*.h'))
    files += [REPO/'scripts'/n for n in ('artemis_layered_campaign.py','artemis_resilience_campaign.py')]
    sources={str(f.relative_to(REPO)):sha(f) for f in files}
    paths=[args.data/f'ARTEMIS_P{p}_DEF_2009022_2012022.bsp' for p in (1,2)]
    paths += [args.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [args.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[args.native]
    inputs={str(p.resolve()):sha(p) for p in paths}
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
    done={key(r) for r in report['results']}
    pending=[j for j in work if (j[0].name,j[1].name,j[2],'layered_mpc' if j[3] else 'routine_mpc') not in done]
    print('Frozen',len(work),'pending',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
            initargs=(root,args.data,args.kernels,args.physics,args.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run,job) for job in pending]):
            row=future.result();json.dumps(row,allow_nan=False)
            report['results'].append(row);report['results'].sort(key=key);save(out,report)
            print(*key(row),json.dumps(row['metrics']),row.get('failure',{}).get('message',''),flush=True)
    for name,digest in sources.items():
        if sha(REPO/name)!=digest:raise ValueError('Sources changed during execution')


if __name__=='__main__':main()
