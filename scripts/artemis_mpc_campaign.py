#!/usr/bin/env python3
"""MPC replaces PID; identical plant, navigation and draws preserve comparison."""
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
from qrc_pulsar_bench.predictive_mpc import PredictiveMPC,MPCSettings
from qrc_pulsar_bench.fpga_mpc import compile_core

REPO=Path(__file__).resolve().parents[1]
EXPERIMENT=None
ROOT=None
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');temp.replace(p)
def configurations():
    return {f'mpc_{n//2}h_dv{w:g}':MPCSettings(horizon_steps=n,dv_scale_m_s=w) for n in (4,12) for w in (.02,.05,.1)}
def initialize(root,base,kernels,physics,native,phase):
    global EXPERIMENT,ROOT
    ROOT=root;EXPERIMENT=PIDExperiment(base/'data',kernels,physics,native,{})
    if phase=='evaluate':
        for probe in (1,2):
            case=dict(name=f'P{probe}_dec',probe=probe,utc='2011-12-01T00:00:00',split='fresh_evaluation')
            _,states,metadata=load_arc(case,root/'data',kernels)
            model,inputs=make_model(metadata['epoch_jd_tdb'],3*DAY,physics,kernels/'de440s.bsp',kernels/'gm_de440.tpc',native)
            initial=states[0].copy();plan=model.propagate(initial,0,3*DAY,500.)
            if not plan.completed:raise RuntimeError('Reference propagation failed')
            EXPERIMENT.cache[case['name']]=(initial,model,plan,metadata,inputs)
def run(job):
    case,seed,config,arithmetic=job
    factory=None
    if config!='tau1h_i0.0':
        settings=configurations()[config]
        factory=lambda model,plan:PredictiveMPC(model,plan,settings,
            library=None if arithmetic=='float64' else ROOT/f'mpc{arithmetic}.so',bits=32 if arithmetic=='float64' else int(arithmetic))
    return EXPERIMENT.run(case,seed,config,arithmetic=arithmetic,faults=seed%100==2,guidance_factory=factory)
def select(rows):
    expected={(c,s) for c in ('P1_sep','P2_sep') for s in (4101,4102)};summary=[]
    for config in configurations():
        group=[r for r in rows if r['configuration']==config]
        if len(group)!=4 or {(r['case'],r['seed']) for r in group}!=expected or any(r['split']!='training' or r['arithmetic']!='float64' for r in group):raise ValueError('Complete training only')
        feasible=all(not r['collision'] and r['steady_rms_m'] is not None and r['steady_rms_m']<=100 and r['max_m']<=10000 and not any(r['core_status_counts'].get(str(k),0) for k in (1,4,16,32)) for r in group)
        summary.append(dict(configuration=config,feasible=feasible,mean_dv_m_s=sum(r['dv_m_s'] for r in group)/4,worst_steady_rms_m=max(r['steady_rms_m'] for r in group)))
    allowed=[r for r in summary if r['feasible']]
    return dict(selected=None if not allowed else min(allowed,key=lambda r:r['mean_dv_m_s'])['configuration'],candidates=summary)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','pilot','train','precision','evaluate'])
    for name in ('output','base','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    protocol=dict(classification='Synthetic controlled spacecraft initialized from definitive ARTEMIS; no future observed spacecraft states in guidance',settings={k:asdict(v) for k,v in configurations().items()},
        training_cases=['P1_sep','P2_sep'],training_seeds=[4101,4102],evaluation_cases=['P1_dec','P2_dec'],evaluation_seeds=[7101,7102],
        selection='minimum mean DV among all four training runs with steady RMS<=100m, max error<=10000m, no collision/solver/guard fault',
        precision='24/32bit on selected September cases: <=1% RMS/DV relative change, <=10m max error change, no solver/range/guard fault; choose lowest passing width',
        comparison='Retained one-hour PD plus feedforward baseline, same plant/sensors/seeds. MPC has two QPs of256fixediterations, no PID fallback, no extra bias feedforward.',
        scope='QP core only for FPGA; host navigation, force linearization, QP formation and nonlinear first-burn screen')
    pp=root/'protocol.json'
    if pp.exists() and read(pp)!=protocol:raise ValueError('Protocol changed; use new campaign directory')
    if not pp.exists():save(pp,protocol)
    if a.phase=='prepare':
        data=root/'data';data.mkdir(exist_ok=True);manifest=[]
        for probe in (1,2):
            name=f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp';source=a.base/'data'/name;dest=data/name
            if not dest.exists():shutil.copy2(source,dest)
            assert sha(source)==sha(dest)
            manifest.append(dict(file=str(dest),sha256=sha(dest),source=str(source)))
        for probe in ('thb','thc'):
            for day in range(1,5):
                name=f'{probe}_l1_state_201112{day:02}_v02.cdf';dest=data/name;url=f'https://spdf.gsfc.nasa.gov/pub/data/themis/{probe}/l1/state/2011/{name}'
                if not dest.exists():
                    temporary=dest.with_suffix('.partial')
                    with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as f:shutil.copyfileobj(response,f)
                    temporary.replace(dest)
                manifest.append(dict(file=str(dest),url=url,sha256=sha(dest)))
        save(root/'download-manifest.json',manifest);print('Eight December V02 files and two definitive SPKs verified');return
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+[REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    hashes={str(p.relative_to(REPO)):sha(p) for p in files}
    if a.phase!='pilot':
        frozen=root/'source-sha256.json'
        if frozen.exists() and read(frozen)!=hashes:raise ValueError('Numerical sources changed')
        if not frozen.exists():
            save(frozen,hashes)
            for p in files:
                dest=root/'source'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    inputs=read(a.base/'pid-input-sha256.json')
    for item in read(root/'download-manifest.json'):inputs[item['file']]=item['sha256']
    for name,digest in inputs.items():
        if sha(Path(name))!=digest:raise ValueError('Physical input changed: '+name)
    save(root/'input-sha256.json',inputs)
    for bits in (24,32):compile_core(root/f'mpc{bits}.so',bits,bits-8)
    if a.phase=='pilot':jobs=[('P1_sep',4901,'mpc_2h_dv0.05','float64')]
    elif a.phase=='train':jobs=[(c,s,n,'float64') for c in ('P1_sep','P2_sep') for s in (4101,4102) for n in configurations()]
    else:
        name=read(root/'train.json')['selection']['selected']
        if name is None:raise ValueError('No training candidate qualified')
        if a.phase=='precision':jobs=[(c,s,name,str(b)) for c in ('P1_sep','P2_sep') for s in (4101,4102) for b in (24,32)]
        else:
            choice=read(root/'precision-selection.json');bits=choice['selected_bits']
            if bits is None:raise ValueError('No fixed-point width qualified')
            jobs=[(c,s,n,b) for c in ('P1_dec','P2_dec') for s in (7101,7102) for n,b in ((name,'float64'),(name,str(bits)),('tau1h_i0.0','float64'))]
    path=root/f'{a.phase}.json';report=read(path) if path.exists() else dict(source_sha256=hashes,protocol_sha256=sha(pp),runner_sha256=sha(Path(__file__)),input_sha256=inputs,results=[])
    if report['source_sha256']!=hashes or report['protocol_sha256']!=sha(pp) or report['input_sha256']!=inputs:raise ValueError('Resume receipt mismatch')
    done={(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in report['results']};jobs=[j for j in jobs if j not in done]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.base,a.kernels,a.physics,a.native,a.phase),mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(run,jobs):
            report['results'].append(r);save(path,report)
            print(r['case'],r['seed'],r['configuration'],r['arithmetic'],'RMS',r['steady_rms_m'],'DV',r['dv_m_s'],'status',r['core_status_counts'],flush=True)
    if a.phase=='train':report['selection']=select(report['results']);save(path,report);print('SELECTION',report['selection'],flush=True)
    if hashes!={str(p.relative_to(REPO)):sha(p) for p in files}:raise ValueError('Source changed during execution')

if __name__=='__main__':main()
