#!/usr/bin/env python3
"""Freeze and test bounded routine MPC after the seven-day tracking failure."""
import argparse
import concurrent.futures
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import shutil
import json
from artemis_resilience_campaign import sha,read,save,REPO,archive_runners
from qrc_pulsar_bench.resilience_benchmark import Scenario,ResilienceExperiment
from qrc_pulsar_bench.routine_supervisor import RoutineMPC,RoutineLimits
from qrc_pulsar_bench.fpga_mpc import compile_core

EXP=None;ROOT=None
def cases():
    regressions=[Scenario('P1_jan15_regression',1,'2012-01-15T00:00:00',3.,9201,'regression',((36.,42.),)),
        Scenario('P1_jan18_failure_replay',1,'2012-01-18T00:00:00',7.,9202,'regression',((36.,44.),(108.,120.)),((24.,36.,3e-7),(96.,132.,2e-7)),bias_m_s2=1.2e-7,thrust_sigma=.06)]
    endurance=[Scenario(f'P{p}_jan20_30days',p,'2012-01-20T00:00:00',30.,9400+p,'fresh_endurance',
        ((36.,42.),(204.,210.),(372.,384.),(540.,546.)),
        ((24.,36.,3e-7),(192.,216.,3e-7),(360.,396.,2e-7),(528.,540.,3e-7))) for p in (1,2)]
    return regressions,endurance
def initialize(root,data,kernels,physics,native):
    global EXP,ROOT
    ROOT=root;EXP=ResilienceExperiment(data,kernels,physics,native)
def run(job):
    scenario,bits=job
    result=EXP.run(scenario,'legacy',library=None if bits=='float64' else ROOT/'mpc32.so',guidance_factory=RoutineMPC)
    result['profile']='supervised_sparse_05';result['routine_limits']=asdict(RoutineLimits())
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('regression','endurance'))
    for name in ('output','previous','data','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    archive_runners(root,('artemis_resilience_campaign.py','artemis_supervised_campaign.py'))
    regressions,endurance=cases()
    protocol=json.loads(json.dumps(dict(routine_limits=asdict(RoutineLimits()),regressions=[asdict(s) for s in regressions],endurance=[asdict(s) for s in endurance],
        reason='Prior fresh seven-day failure spent >20m/s and exceeded local MPC validity. Preserve failed evidence; add estimated-error latch and rolling issued-DV budget. No claim that command inhibition recovers the mission.',
        previous_fresh_path=str(a.previous/'fresh.json'),previous_fresh_sha256=sha(a.previous/'fresh.json'),
        precision='Fixed32/24. Replays are diagnostics, not holdout. New Jan20 geometries/seeds reserved for30day endurance after supervisor limits frozen.',
        scoring='Five-minute truth samples; steady excludes first12h. Report all limits, fault/inhibit modes and full-arc errors.')))
    pp=root/'protocol.json'
    if pp.exists() and read(pp)!=protocol:raise ValueError('Frozen protocol changed')
    if not pp.exists():save(pp,protocol)
    files=list((REPO/'qrc_pulsar_bench').glob('*.py'))+[REPO/'hardware'/n for n in ('orbit_mpc_core.cpp','orbit_mpc_core.h')]
    sources={str(p.relative_to(REPO)):sha(p) for p in files};sf=root/'source-sha256.json'
    if sf.exists() and read(sf)!=sources:raise ValueError('Numerical sources changed')
    if not sf.exists():
        save(sf,sources)
        for f in files:
            dest=root/'source'/f.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    paths=[a.data/f'ARTEMIS_P{probe}_DEF_2009022_2012022.bsp' for probe in (1,2)]
    paths += [a.kernels/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]
    paths += [a.physics/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
    inputs={str(p.resolve()):sha(p) for p in paths}
    if inputs!=read(a.previous/'input-sha256.json'):raise ValueError('Actual input arguments differ from the previous frozen inputs')
    save(root/'input-sha256.json',inputs);compile_core(root/'mpc32.so',32,24)
    jobs=[(s,b) for s in (regressions if a.phase=='regression' else endurance) for b in ('float64','32')]
    out=root/f'{a.phase}.json'
    result=read(out) if out.exists() else dict(source_sha256=sources,input_sha256=inputs,protocol_sha256=sha(pp),runner_sha256=sha(Path(__file__)),results=[])
    if result['source_sha256']!=sources or result['input_sha256']!=inputs or result['protocol_sha256']!=sha(pp) or result['runner_sha256']!=sha(Path(__file__)):raise ValueError('Resume provenance changed')
    done={(r['scenario']['name'],r['arithmetic']) for r in result['results']}
    print('PHASE',a.phase,'jobs',len(jobs)-len(done),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(root,a.data,a.kernels,a.physics,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(run,[j for j in jobs if (j[0].name,j[1]) not in done]):
            result['results'].append(r);save(out,result);print(r['scenario']['name'],r['arithmetic'],r['metrics'],flush=True)
    if len(result['results'])!=len(jobs):raise ValueError('Incomplete matrix')
    if sources!={str(p.relative_to(REPO)):sha(p) for p in files}:raise ValueError('Source changed during run')

if __name__=='__main__':main()
