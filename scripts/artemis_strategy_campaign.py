#!/usr/bin/env python3
"""Frozen multiobjective search for tight tracking, low delta-v and fewer burns."""
import argparse
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import urllib.request
from qrc_pulsar_bench.artemis_pid_benchmark import PIDExperiment
from qrc_pulsar_bench.artemis_replay import load_arc,make_model,DAY
from qrc_pulsar_bench.predictive_mpc import PredictiveMPC
from qrc_pulsar_bench.mpc_strategies import StrategyMPC,strategies,select_strategies,score_run
from qrc_pulsar_bench.fpga_mpc import compile_core

REPO=Path(__file__).resolve().parents[1]
EXPERIMENT=None;ROOT=None
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def fresh_cases():
    return [dict(name=f'P{probe}_jan{day:02}',probe=probe,utc=f'2012-01-{day:02}T00:00:00',split='fresh_evaluation') for day in (1,10) for probe in (1,2)]
def initialize(root,base,kernels,physics,native,phase):
    global EXPERIMENT,ROOT
    ROOT=root;EXPERIMENT=PIDExperiment(base/'data',kernels,physics,native,{})
    if phase=='evaluate':
        for case in fresh_cases():
            _,states,metadata=load_arc(case,root/'data',kernels)
            model,inputs=make_model(metadata['epoch_jd_tdb'],3*DAY,physics,kernels/'de440s.bsp',kernels/'gm_de440.tpc',native)
            initial=states[0].copy();plan=model.propagate(initial,0,3*DAY,500.)
            if not plan.completed:raise RuntimeError('Reference propagation failed')
            EXPERIMENT.cache[case['name']]=(initial,model,plan,metadata,inputs)
def run(job):
    case,seed,name,arithmetic=job;spec=strategies()[name]
    def factory(model,plan):
        args=dict(library=None if arithmetic=='float64' else ROOT/f'mpc{arithmetic}.so',bits=32 if arithmetic=='float64' else int(arithmetic))
        return PredictiveMPC(model,plan,spec.mpc,**args) if name=='baseline' else StrategyMPC(model,plan,spec,**args)
    row=EXPERIMENT.run(case,seed,name,arithmetic=arithmetic,faults=seed%100==2,guidance_factory=factory,score_step_s=300.)
    row['strategy']=asdict(spec);row['metrics']=score_run(row);return row
def select_precision(root):
    training=read(root/'train.json');precision=read(root/'precision.json');name=training['selection']['primary']
    if precision['source_sha256']!=training['source_sha256'] or precision['protocol_sha256']!=training['protocol_sha256'] or precision['input_sha256']!=training['input_sha256']:raise ValueError('Precision provenance mismatch')
    refs={(r['case'],r['seed']):r for r in training['results'] if r['configuration']==name}
    expected={(c,s,b) for c in ('P1_sep','P2_sep') for s in (4101,4102) for b in ('24','32')};rows=precision['results']
    if len(rows)!=8 or {(r['case'],r['seed'],r['arithmetic']) for r in rows}!=expected:raise ValueError('Incomplete precision matrix')
    checks=[]
    for r in rows:
        if r['configuration']!=name or r['split']!='training':raise ValueError('Training-only precision selection')
        x=refs[(r['case'],r['seed'])];m=r['metrics'];ref=x['metrics']
        dr=abs(m['rms_m']/ref['rms_m']-1);dv=abs(m['dv_m_s']/ref['dv_m_s']-1);db=abs(m['burns']-ref['burns'])
        passed=dr<=.01 and dv<=.01 and db<=1 and abs(m['peak_m']-ref['peak_m'])<=10 and not m['collision'] and not m['solver_or_screen_faults']
        checks.append(dict(case=r['case'],seed=r['seed'],bits=int(r['arithmetic']),rms_relative_change=dr,dv_relative_change=dv,burn_count_change=db,passed=bool(passed)))
    passing=[b for b in (24,32) if all(r['passed'] for r in checks if r['bits']==b)]
    choice=dict(selected_bits=min(passing) if passing else None,configuration=name,checks=checks,training_sha256=sha(root/'train.json'),precision_sha256=sha(root/'precision.json'))
    save(root/'precision-selection.json',choice);return choice
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','pilot','train','precision','evaluate'])
    for name in ('output','base','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    protocol=dict(classification='Synthetic controlled spacecraft on real definitive ARTEMIS initial geometry',settings={k:asdict(v) for k,v in strategies().items()},
        training_cases=['P1_sep','P2_sep'],training_seeds=[4101,4102],fresh_cases=fresh_cases(),fresh_seeds=[8101,8102],score_step_s=300,
        feasible='All four training cases: five-minute sampled steady RMS<=100m, steady peak<=500m, no collision/solver/clearance fault; first12h excluded from steady scoring',
        pareto='Nondominated feasible candidates on worst RMS, mean delta-v and mean burn count',
        selection='Report tightest, least fuel, fewest burns; primary minimizes worstRMS/50+meanDV/baselineDV+meanBurns/baselineBurns among RMS<=50m,peak<=300m if any, otherwise feasible set',
        precision='Primary only on September: <=1% relative RMS/DV change, <=10m steady peak change, <=1 burn-count change per case, no faults; choose lowest passing24/32width',
        evaluation='Selected distinct candidates and baseline float64, plus primary selected width on8fresh January cases; no holdout tuning',
        assumptions='Same500kg,.2N generic vehicle and synthetic fixes/noise/faults as prior campaign. No PIDfallback or future observed spacecraft data in guidance. Device unspecified.')
    pp=root/'protocol.json'
    if pp.exists() and read(pp)!=protocol:raise ValueError('Frozen protocol changed')
    if not pp.exists():save(pp,protocol)
    if a.phase=='prepare':
        data=root/'data';data.mkdir(exist_ok=True);manifest=[]
        for probe in (1,2):
            name=f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp';src=a.base/'data'/name;dest=data/name
            if not dest.exists():shutil.copy2(src,dest)
            if sha(src)!=sha(dest):raise ValueError('SPK mismatch')
            manifest.append(dict(file=str(dest),source=str(src),sha256=sha(dest)))
        def fetch(item):
            probe,day=item;name=f'{probe}_l1_state_201201{day:02}_v02.cdf';dest=data/name
            url=f'https://spdf.gsfc.nasa.gov/pub/data/themis/{probe}/l1/state/2012/{name}'
            if not dest.exists():
                part=dest.with_suffix('.partial')
                with urllib.request.urlopen(url,timeout=60) as response,part.open('wb') as f:shutil.copyfileobj(response,f)
                part.replace(dest)
            return dict(file=str(dest),url=url,sha256=sha(dest))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:manifest+=list(pool.map(fetch,[(p,d) for p in ('thb','thc') for d in (*range(1,5),*range(10,14))]))
        save(root/'download-manifest.json',manifest);print('16 definitive January CDFs and2SPKs verified');return
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+[REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    hashes={str(p.relative_to(REPO)):sha(p) for p in files}
    frozen=root/'source-sha256.json'
    if a.phase!='pilot':
        if frozen.exists() and read(frozen)!=hashes:raise ValueError('Numerical sources changed')
        if not frozen.exists():
            save(frozen,hashes)
            for p in files:
                dest=root/'source'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    inputs=read(a.base/'pid-input-sha256.json')
    for item in read(root/'download-manifest.json'):inputs[item['file']]=item['sha256']
    for name,h in inputs.items():
        if sha(Path(name))!=h:raise ValueError('Physical input changed')
    save(root/'input-sha256.json',inputs)
    for b in (24,32):compile_core(root/f'mpc{b}.so',b,b-8)
    if a.phase=='pilot':jobs=[('P1_sep',4901,n,'float64') for n in ('smooth_003','spaced_2h','event_60m','sparse_01')]
    elif a.phase=='train':jobs=[(c,s,n,'float64') for c in ('P1_sep','P2_sep') for s in (4101,4102) for n in strategies()]
    else:
        training=read(root/'train.json');selection=training['selection'];name=selection['primary']
        if training['source_sha256']!=hashes or training['protocol_sha256']!=sha(pp):raise ValueError('Training provenance mismatch')
        if name is None:raise ValueError('No candidate qualified')
        if a.phase=='precision':jobs=[(c,s,name,str(b)) for c in ('P1_sep','P2_sep') for s in (4101,4102) for b in (24,32)]
        else:
            choice=read(root/'precision-selection.json');bits=choice['selected_bits']
            if bits is None:raise ValueError('No fixed-point width qualified')
            if choice['training_sha256']!=sha(root/'train.json'):raise ValueError('Selection changed')
            configs=[(n,'float64') for n in dict.fromkeys(['baseline',*selection['selected']])]+[(name,str(bits))]
            jobs=[(c['name'],s,n,b) for c in fresh_cases() for s in (8101,8102) for n,b in configs]
    path=root/f'{a.phase}.json';report=read(path) if path.exists() else dict(source_sha256=hashes,protocol_sha256=sha(pp),input_sha256=inputs,runner_sha256=sha(Path(__file__)),results=[])
    if report['source_sha256']!=hashes or report['protocol_sha256']!=sha(pp) or report['input_sha256']!=inputs:raise ValueError('Resume provenance mismatch')
    done={(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in report['results']};jobs=[j for j in jobs if j not in done]
    print('PHASE',a.phase,'jobs',len(jobs),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.base,a.kernels,a.physics,a.native,a.phase),mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(run,jobs):
            report['results'].append(row);save(path,report)
            print(row['case'],row['seed'],row['configuration'],row['arithmetic'],row['metrics'],flush=True)
    if a.phase=='train':report['selection']=select_strategies(report['results']);save(path,report);print('SELECTION',report['selection'],flush=True)
    if a.phase=='precision':print('PRECISION',select_precision(root),flush=True)
    if hashes!={str(p.relative_to(REPO)):sha(p) for p in files}:raise ValueError('Sources changed during run')

if __name__=='__main__':main()
