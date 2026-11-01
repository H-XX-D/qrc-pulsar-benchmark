#!/usr/bin/env python3
"""Frozen development, arithmetic, fresh and thirty-day outage-control tests."""
import argparse
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
from qrc_pulsar_bench.resilience_benchmark import Scenario,ResilienceExperiment
from qrc_pulsar_bench.resilient_mpc import profiles
from qrc_pulsar_bench.fpga_mpc import compile_core

REPO=Path(__file__).resolve().parents[1]
EXPERIMENT=None
ROOT=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temporary.replace(p)
def archive_runners(root,names):
    for name in names:
        source=REPO/'scripts'/name;dest=root/'source'/'scripts'/name
        if dest.exists() and sha(dest)!=sha(source):raise ValueError('Archived runner changed; use its frozen copy or a new output directory')
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
def scenarios():
    development=[Scenario(f'P{p}_jan10_{s}',p,'2012-01-10T00:00:00',3.,s,'development',((36.,42.),)) for p in (1,2) for s in (8101,8102)]
    fresh=[]
    for day in (15,18):
        for p in (1,2):
            for seed in (9201,9202):
                days=3. if day==15 else 7.
                outages=((36.,42.),) if days==3 else ((36.,44.),(108.,120.))
                pulses=((24.,36. if p==1 else 60.,3e-7),) if days==3 else ((24.,36. if p==1 else 60.,3e-7),(96.,132. if p==1 else 108.,2e-7))
                fresh.append(Scenario(f'P{p}_jan{day}_{seed}',p,f'2012-01-{day}T00:00:00',days,seed,'fresh',outages,pulses,
                                      bias_m_s2=8e-8 if days==3 else 1.2e-7,thrust_sigma=.03 if days==3 else .06))
    endurance=[Scenario(f'P{p}_30days',p,'2012-01-15T00:00:00',30.,9300+p,'endurance',
                        tuple((36.+168*i,42.+168*i) for i in range(4)),
                        tuple((24.+168*i,(36. if i%2==0 else 60.)+168*i,3e-7) for i in range(4))) for p in (1,2)]
    return development,fresh,endurance
def initialize(root,data,kernels,physics,native):
    global EXPERIMENT,ROOT
    ROOT=root;EXPERIMENT=ResilienceExperiment(data,kernels,physics,native)
def run(job):
    scenario,name,bits=job
    return EXPERIMENT.run(scenario,name,library=None if bits=='float64' else ROOT/'mpc32.so')
def key(row):return row['scenario']['name'],row['profile'],row['arithmetic']
def summarize(rows):
    import numpy as np
    return dict(cases=len(rows),mean_rms_m=float(np.mean([r['metrics']['rms_m'] for r in rows])),
                worst_rms_m=max(r['metrics']['rms_m'] for r in rows),worst_peak_m=max(r['metrics']['peak_m'] for r in rows),
                mean_dv_m_s=float(np.mean([r['metrics']['dv_m_s'] for r in rows])),mean_burns=float(np.mean([r['metrics']['burns'] for r in rows])),
                failed_cases=sum(bool(not r['metrics']['completed'] or r['metrics']['numerical_faults'] or r['metrics']['rms_m']>100 or r['metrics']['peak_m']>500) for r in rows),
                tight_cases=sum(r['metrics']['completed'] and not r['metrics']['numerical_faults'] and r['metrics']['rms_m']<=50 and r['metrics']['peak_m']<=300 for r in rows))
def select(rows):
    development,_,_=scenarios()
    expected={(s.name,n,'float64') for s in development for n in profiles()}
    if len(rows)!=len(expected) or {key(r) for r in rows}!=expected:raise ValueError('Incomplete development grid')
    if any(r['scenario']['split']!='development' for r in rows):raise ValueError('Selection cannot read holdout')
    table={n:summarize([r for r in rows if r['profile']==n]) for n in profiles()}
    qualified=[n for n in table if table[n]['failed_cases']==0]
    base=table['legacy']
    chosen=min(qualified,key=lambda n:table[n]['worst_rms_m']/50+table[n]['mean_dv_m_s']/base['mean_dv_m_s']+table[n]['mean_burns']/base['mean_burns']) if qualified else None
    return dict(selected=chosen,table=table,rule='Among all-case100mRMS/500mpeak qualified profiles, minimize worstRMS/50 + meanDV/baselineDV + meanBurns/baselineBurns')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('develop','precision','fresh','endurance'))
    for name in ('output','data','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    archive_runners(root,('artemis_resilience_campaign.py',))
    development,fresh,endurance=scenarios()
    protocol=json.loads(json.dumps(dict(profiles={n:dict(filter=asdict(v['filter']),guard=None if v['guard'] is None else asdict(v['guard']),bias=v['bias']) for n,v in profiles().items()},
        development=[asdict(s) for s in development],fresh=[asdict(s) for s in fresh],endurance=[asdict(s) for s in endurance],
        selection='Development-only:100mRMS/500mpeak/no numericalfault/full completion; minimize worstRMS/50+DV/baselineDV+burns/baselineburns',
        precision='32/24 on four development cases:<=1% RMS/DV,<=10mpeak,<=1burndifference,no numerical faults',
        classification='Synthetic controlled vehicles, one real reconstructed initial state. January10 is reused development data; January15/18 and new seeds are fresh. No future spacecraft state is read.',
        scoring='5minute samples; steady excludes first12h; fuel and burns cover full arc; report all failed cases, no holdout tuning')))
    pp=root/'protocol.json'
    if pp.exists() and read(pp)!=protocol:raise ValueError('Protocol changed')
    if not pp.exists():save(pp,protocol)
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+[REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    sources={str(f.relative_to(REPO)):sha(f) for f in files};sp=root/'source-sha256.json'
    if sp.exists() and read(sp)!=sources:raise ValueError('Numerical source changed')
    if not sp.exists():
        save(sp,sources)
        for f in files:
            dest=root/'source'/f.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    paths=[a.data/f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp' for probe in (1,2)]
    paths += [a.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [a.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
    inputs={str(p.resolve()):sha(p) for p in paths};ip=root/'input-sha256.json'
    if ip.exists() and read(ip)!=inputs:raise ValueError('Input changed')
    save(ip,inputs);compile_core(root/'mpc32.so',32,24)
    selection=None
    if a.phase=='develop':jobs=[(s,n,'float64') for s in development for n in profiles()]
    else:
        training=read(root/'develop.json');selection=training['selection'];name=selection['selected']
        if name is None:raise ValueError('No qualified development setting')
        if a.phase=='precision':jobs=[(s,name,'32') for s in development]
        else:
            precision=read(root/'precision.json')
            if not precision['arithmetic_passed'] or precision['development_sha256']!=sha(root/'develop.json'):raise ValueError('Arithmetic qualification required')
            jobs=[(s,name,b) for s in (fresh if a.phase=='fresh' else endurance) for b in ('float64','32')]
    output=root/f'{a.phase}.json'
    report=read(output) if output.exists() else dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(pp),runner_sha256=sha(Path(__file__)),results=[])
    if report['source_sha256']!=sources or report['input_sha256']!=inputs or report['protocol_sha256']!=sha(pp) or report['runner_sha256']!=sha(Path(__file__)):raise ValueError('Resume provenance mismatch')
    done={key(r) for r in report['results']};todo=[j for j in jobs if (j[0].name,j[1],j[2]) not in done]
    print('PHASE',a.phase,'jobs',len(todo),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.data,a.kernels,a.physics,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(run,todo):
            report['results'].append(r);save(output,report);print(*key(r),r['metrics'],flush=True)
    if len(report['results'])!=len(jobs) or {key(r) for r in report['results']}!={(s.name,n,b) for s,n,b in jobs}:raise ValueError('Incomplete/duplicate matrix')
    if a.phase=='develop':report['selection']=select(report['results']);print('SELECTION',report['selection'],flush=True)
    if a.phase=='precision':
        refs={r['scenario']['name']:r for r in training['results'] if r['profile']==name};checks=[]
        for r in report['results']:
            m=r['metrics'];ref=refs[r['scenario']['name']]['metrics']
            dr=abs(m['rms_m']/ref['rms_m']-1);dv=abs(m['dv_m_s']/ref['dv_m_s']-1);db=abs(m['burns']-ref['burns']);dp=abs(m['peak_m']-ref['peak_m'])
            checks.append(dict(case=r['scenario']['name'],rms_relative_change=dr,dv_relative_change=dv,burn_count_change=db,peak_change_m=dp,
                passed=dr<=.01 and dv<=.01 and db<=1 and dp<=10 and not m['numerical_faults'] and m['completed']))
        report.update(checks=checks,arithmetic_passed=all(c['passed'] for c in checks),development_sha256=sha(root/'develop.json'))
        print('ARITHMETIC',report['arithmetic_passed'],checks,flush=True)
    save(output,report)
    if sources!={str(f.relative_to(REPO)):sha(f) for f in files}:raise ValueError('Source changed during run')

if __name__=='__main__':main()
