#!/usr/bin/env python3
"""Run after python -m qrc_pulsar_bench.artemis_download OUTPUT.

Invoke from the repository with PYTHONPATH=. or install the package first.
The controller benchmark does not use future recorded spacecraft states.
"""
import argparse,concurrent.futures,hashlib,json,multiprocessing,pathlib
from dataclasses import asdict
from qrc_pulsar_bench.artemis_pid_benchmark import PIDExperiment,candidates,select
from qrc_pulsar_bench.fpga_pid import compile_core
ROOT=None
REPO=pathlib.Path(__file__).resolve().parents[1]
K=None;P=None
EXPERIMENT=None
def init(root,kernels,physics,native):
 global EXPERIMENT
 EXPERIMENT=PIDExperiment(root/'data',kernels,physics,native,{b:root/f'pid{b}.so' for b in (16,24,32)})
def run(job):
 case,seed,config,arithmetic,faults=job
 return EXPERIMENT.run(case,seed,config,arithmetic=arithmetic,faults=faults)
def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def main():
 global ROOT,K,P
 parser=argparse.ArgumentParser(description='Frozen ARTEMIS-geometry PID experiment; synthetic control observations.')
 parser.add_argument('phase',choices=['pilot','train','precision','evaluate'])
 parser.add_argument('--workers',type=int,default=3)
 parser.add_argument('--output',type=pathlib.Path,required=True)
 parser.add_argument('--kernels',type=pathlib.Path,required=True)
 parser.add_argument('--physics',type=pathlib.Path,required=True)
 parser.add_argument('--native',type=pathlib.Path,required=True)
 a=parser.parse_args();ROOT=a.output;K=a.kernels;P=a.physics;ROOT.mkdir(parents=True,exist_ok=True)
 names=['qrc_pulsar_bench/artemis_replay.py','qrc_pulsar_bench/artemis_pid_benchmark.py','qrc_pulsar_bench/fpga_pid.py','hardware/orbit_pid_core.cpp','hardware/orbit_pid_core.h','qrc_pulsar_bench/adaptive_control.py','qrc_pulsar_bench/jacobi_control.py']
 hashes={n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in names}
 input_paths=sorted((ROOT/'data').glob('*'))+[K/n for n in ('de440s.bsp','gm_de440.tpc','naif0012.tls')]+[P/n for n in ('gggrx_0900c_sha.tab','moon_pa_de421_1900-2050.bpc','moon_080317.tf')]+[a.native]
 input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths if p.is_file()}
 input_receipt=ROOT/'pid-input-sha256.json'
 if input_receipt.exists() and json.load(open(input_receipt))!=input_hashes:raise ValueError('Physical inputs changed between phases')
 if not input_receipt.exists():save(input_receipt,input_hashes)
 protocol=dict(classification='Simulated control on definitive ARTEMIS initial geometry; no later historical state in guidance',settings={n:asdict(s) for n,s in candidates().items()},training_cases=['P1_sep','P2_sep'],training_seeds=[4101,4102],evaluation_cases=['P1_oct','P2_oct'],evaluation_seeds=[5101,5102],selection='Training only; all steady RMS<=1000m, maxsampleerror<=10000m, no collision; choose minimum meanDV; if none qualify flag failure',precision='Compare 16.8/24.16/32.24 integer arithmetic with float64 on training; require <=1% RMS and DV changes and <=10m change in max error, no range faults',spacecraft='assumed500kg, .2N omnidirectional equivalent thruster,220sIsp; not ARTEMIS spin-constrained actuation',sensor='synthetic30m/.002m/s per-axis direct-state fixes,60s delayed,30min cadence',faults='seed ending2:6hour outage starting36h and corruptfix48h; allcases8e-8 constantbias and3e-7 transverse pulse24to36h,3percentthrust,.005radpointing')
 pp=ROOT/'pid-protocol.json'
 if pp.exists():assert json.load(open(pp))==protocol
 else:save(pp,protocol)
 for b in (16,24,32):
  lib=ROOT/f'pid{b}.so'
  # Rebuild from the checked core, avoiding a stale host library after edits.
  compile_core(lib,b,b-8)
 jobs=[]
 if a.phase=='pilot':jobs=[('P1_sep',4900,'tau3h_i0.1','float64',True)]
 elif a.phase=='train':jobs=[(case,seed,cfg,'float64',seed==4102) for case in ['P1_sep','P2_sep'] for seed in [4101,4102] for cfg in candidates()]
 else:
  training=json.load(open(ROOT/'pid-train.json'));assert training['source_sha256']==hashes
  if a.phase=='evaluate' and not (ROOT/'pid-precision-selection.json').exists():raise ValueError('Freeze arithmetic selection from training before evaluation')
  chosen=training['selection']['selected']
  cases=['P1_sep','P2_sep'] if a.phase=='precision' else ['P1_oct','P2_oct']
  seeds=[4101,4102] if a.phase=='precision' else [5101,5102]
  arithmetic=['16','24','32'] if a.phase=='precision' else ['float64','16','24','32']
  jobs=[(case,seed,chosen,ar,seed%100==2) for case in cases for seed in seeds for ar in arithmetic]
 output=ROOT/f'pid-{a.phase}.json'
 report=json.load(open(output)) if output.exists() else dict(source_sha256=hashes,protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),results=[])
 assert report['source_sha256']==hashes
 report['input_sha256']=input_hashes
 report['runner_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
 done={(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in report['results']};jobs=[j for j in jobs if j[:4] not in done]
 print('phase',a.phase,'jobs',len(jobs),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(ROOT,K,P,a.native),mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run,jobs):
   report['results'].append(r);save(output,report)
   print(r['case'],r['seed'],r['configuration'],r['arithmetic'],'steady',r['steady_rms_m'],'DV',r['dv_m_s'],'max',r['max_m'],'faults',r['core_status_counts'],'reject',r['fixes_rejected'],flush=True)
 if a.phase=='train':report['selection']=select(report['results']);save(output,report);print('SELECTION',report['selection'],flush=True)
 assert hashes=={n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in names}
if __name__=='__main__':main()
