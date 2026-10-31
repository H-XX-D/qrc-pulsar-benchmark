#!/usr/bin/env python3
"""Fresh validation after the first, fuel-prioritized controller failed October.

The revised choice minimizes WORST SEPTEMBER TRAINING RMS. November results
cannot select the gain or arithmetic width. The original October run is retained.
"""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from qrc_pulsar_bench.artemis_pid_benchmark import PIDExperiment
from qrc_pulsar_bench.artemis_replay import load_arc,make_model,DAY


EXPERIMENT=None
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def initialize(root,kernels,physics,native,phase):
    global EXPERIMENT
    EXPERIMENT=PIDExperiment(root/'data',kernels,physics,native,{b:root/f'pid{b}.so' for b in (24,32)})
    if phase=='fresh':
        for case in read(root/'pid-accuracy-protocol.json')['fresh_cases']:
            # Composition of the existing model and runner. The only spacecraft
            # datum entering plan/control is states[0], exactly as in training.
            _,states,metadata=load_arc(case,root/'fresh-data',kernels)
            model,inputs=make_model(metadata['epoch_jd_tdb'],3*DAY,physics,kernels/'de440s.bsp',kernels/'gm_de440.tpc',native)
            initial=states[0].copy();plan=model.propagate(initial,0,3*DAY,500.)
            if not plan.completed:raise RuntimeError('Invalid fresh reference')
            EXPERIMENT.cache[case['name']]=(initial,model,plan,metadata,inputs)


def run(job):
    case,seed,config,arithmetic=job
    return EXPERIMENT.run(case,seed,config,arithmetic=arithmetic,faults=seed%100==2)


def select_precision(root):
    protocol=read(root/'pid-accuracy-protocol.json');train=read(root/'pid-train.json')
    name=protocol['chosen']['configuration'];refs={(r['case'],r['seed']):r for r in train['results'] if r['configuration']==name}
    rows=read(root/'pid-accuracy-precision.json')['results'];expected={(c,s,b) for c in ('P1_sep','P2_sep') for s in (4101,4102) for b in ('24','32')}
    if len(rows)!=8 or {(r['case'],r['seed'],r['arithmetic']) for r in rows}!=expected:raise ValueError('Precision training incomplete')
    checks=[]
    for row in rows:
        ref=refs[(row['case'],row['seed'])]
        dr=abs(row['steady_rms_m']/ref['steady_rms_m']-1);dv=abs(row['dv_m_s']/ref['dv_m_s']-1);dm=abs(row['max_m']-ref['max_m'])
        passed=(dr<=.01 and dv<=.01 and dm<=10 and not row['collision'] and not any(row['core_status_counts'].get(str(k),0) for k in (1,4)))
        checks.append(dict(case=row['case'],seed=row['seed'],bits=int(row['arithmetic']),rms_relative_change=dr,dv_relative_change=dv,max_error_change_m=dm,passed=bool(passed)))
    passing=[bits for bits in (24,32) if all(r['passed'] for r in checks if r['bits']==bits)]
    result=dict(selected_bits=min(passing) if passing else None,configuration=name,checks=checks)
    save(root/'pid-accuracy-precision-selection.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['precision','fresh']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kernels',type=Path,required=True);p.add_argument('--physics',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    root=a.output;protocol=read(root/'pid-accuracy-protocol.json');training=read(root/'pid-train.json')
    if sha(root/'pid-train.json')!=protocol['selection_data_sha256']:raise ValueError('Original training receipt changed')
    repo=Path(__file__).resolve().parents[1]
    if any(sha(repo/n)!=s for n,s in training['source_sha256'].items()):raise ValueError('Controller implementation changed')
    chosen=min([r for r in training['selection']['candidates'] if r['feasible']],key=lambda r:r['worst_steady_rms_m'])
    if chosen!=protocol['chosen']:raise ValueError('Gain was not selected from training')
    name=chosen['configuration']
    if a.phase=='precision':jobs=[(c,s,name,b) for c in ('P1_sep','P2_sep') for s in (4101,4102) for b in ('24','32')]
    else:
        selection=read(root/'pid-accuracy-precision-selection.json');bits=selection['selected_bits']
        if bits is None:raise ValueError('No arithmetic format qualified')
        jobs=[(c['name'],s,name,b) for c in protocol['fresh_cases'] for s in protocol['fresh_seeds'] for b in ('float64',str(bits))]
    output=root/f'pid-accuracy-{a.phase}.json';report=read(output) if output.exists() else dict(source_sha256=training['source_sha256'],protocol_sha256=sha(root/'pid-accuracy-protocol.json'),runner_sha256=sha(Path(__file__)),results=[])
    done={(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in report['results']};jobs=[j for j in jobs if j not in done]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.kernels,a.physics,a.native,a.phase),mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(run,jobs):
            report['results'].append(r);save(output,report)
            print(r['case'],r['seed'],r['configuration'],r['arithmetic'],'steady',r['steady_rms_m'],'DV',r['dv_m_s'],'max',r['max_m'],'status',r['core_status_counts'],'reject',r['fixes_rejected'],flush=True)
    if a.phase=='precision':print('PRECISION',select_precision(root),flush=True)
    if any(sha(repo/n)!=s for n,s in training['source_sha256'].items()):raise ValueError('Controller changed during execution')


if __name__=='__main__':main()
