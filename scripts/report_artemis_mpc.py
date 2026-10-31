#!/usr/bin/env python3
"""Training-only precision selection and paired, frozen MPC/PID evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_receipt(report,training,protocol):
    if report['source_sha256']!=training['source_sha256'] or report['protocol_sha256']!=protocol or report['input_sha256']!=training['input_sha256']:
        raise ValueError('Source, protocol or input mismatch')
def precision_selection(root):
    training=read(root/'train.json');precision=read(root/'precision.json');protocol=sha(root/'protocol.json')
    check_receipt(precision,training,protocol)
    name=training['selection']['selected'];expected={(c,s) for c in ('P1_sep','P2_sep') for s in (4101,4102)}
    refs={(r['case'],r['seed']):r for r in training['results'] if r['configuration']==name}
    if set(refs)!=expected:raise ValueError('Incomplete training baseline')
    rows=precision['results']
    if len(rows)!=8 or {(r['case'],r['seed'],r['arithmetic']) for r in rows}!={(c,s,b) for c,s in expected for b in ('24','32')}:raise ValueError('Eight precision training runs required')
    checks=[]
    for row in rows:
        if row['configuration']!=name or row['split']!='training':raise ValueError('Only selected training setting can choose width')
        baseline=refs[(row['case'],row['seed'])]
        dr=abs(row['steady_rms_m']/baseline['steady_rms_m']-1);dv=abs(row['dv_m_s']/baseline['dv_m_s']-1);dm=abs(row['max_m']-baseline['max_m'])
        faults=sum(row['core_status_counts'].get(str(k),0) for k in (1,4,16,32))
        checks.append(dict(case=row['case'],seed=row['seed'],bits=int(row['arithmetic']),rms_relative_change=dr,dv_relative_change=dv,max_error_change_m=dm,faults=faults,
            passed=bool(dr<=.01 and dv<=.01 and dm<=10 and not row['collision'] and faults==0)))
    passing=[b for b in (24,32) if all(c['passed'] for c in checks if c['bits']==b)]
    return dict(selected_bits=min(passing) if passing else None,configuration=name,checks=checks,
                training_receipt_sha256=sha(root/'train.json'),precision_receipt_sha256=sha(root/'precision.json'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--select-precision',action='store_true');p.add_argument('--output',type=Path,default=Path('docs'));a=p.parse_args();root=a.campaign
    selection=precision_selection(root)
    if a.select_precision:
        path=root/'precision-selection.json'
        if path.exists() and read(path)!=selection:raise ValueError('Selection changed')
        path.write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps(selection,indent=2));return
    if read(root/'precision-selection.json')!=selection:raise ValueError('Precision was not frozen')
    train=read(root/'train.json');evaluate=read(root/'evaluate.json');check_receipt(evaluate,train,sha(root/'protocol.json'))
    name=train['selection']['selected'];bits=str(selection['selected_bits']);rows=evaluate['results']
    expected={(c,s,n,b) for c in ('P1_dec','P2_dec') for s in (7101,7102) for n,b in ((name,'float64'),(name,bits),('tau1h_i0.0','float64'))}
    if len(rows)!=12 or {(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in rows}!=expected:raise ValueError('Incomplete December comparison')
    if any(r['split']!='fresh_evaluation' for r in rows):raise ValueError('Wrong evaluation split')
    group={'pid_float64':[r for r in rows if r['configuration']=='tau1h_i0.0'],
           'mpc_float64':[r for r in rows if r['configuration']==name and r['arithmetic']=='float64'],
           'mpc_fixed':[r for r in rows if r['configuration']==name and r['arithmetic']==bits]}
    stats={k:dict(mean_steady_rms_m=float(np.mean([r['steady_rms_m'] for r in v])),mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in v])),
        worst_sampled_error_m=max(r['max_m'] for r in v),mean_online_cpu_s=float(np.mean([r['online_cpu_s'] for r in v])),
        mean_guidance_cpu_s=float(np.mean([r['guidance_cpu_s'] for r in v])),mean_burns=float(np.mean([r['burns'] for r in v])),
        faults=sum(sum(r['core_status_counts'].get(str(j),0) for j in (1,4,16,32)) for r in v)) for k,v in group.items()}
    summary=dict(classification='Matched synthetic control on real ARTEMIS initial geometry, not historical flight control validation or FPGA synthesis',
        training_selection=train['selection'],precision_selection=selection,aggregate=stats,
        rms_reduction_fraction=1-stats['mpc_float64']['mean_steady_rms_m']/stats['pid_float64']['mean_steady_rms_m'],
        dv_reduction_fraction=1-stats['mpc_float64']['mean_dv_m_s']/stats['pid_float64']['mean_dv_m_s'],
        evaluation=[{k:v for k,v in r.items() if k not in ('samples','commands','guidance_diagnostics')} for r in rows],
        source_sha256=train['source_sha256'],receipts={n:dict(path=str(root/n),sha256=sha(root/n)) for n in ('protocol.json','train.json','precision.json','precision-selection.json','evaluate.json','download-manifest.json','input-sha256.json')})
    if (root/'pid-baseline-fairness.json').exists():
        summary['baseline_fairness_audit']=read(root/'pid-baseline-fairness.json')
        summary['receipts']['pid-baseline-fairness.json']=dict(path=str(root/'pid-baseline-fairness.json'),sha256=sha(root/'pid-baseline-fairness.json'))
    if (root/'command-direction-diagnostics.json').exists():
        summary['command_direction_diagnostics']=read(root/'command-direction-diagnostics.json')
        summary['receipts']['command-direction-diagnostics.json']=dict(path=str(root/'command-direction-diagnostics.json'),sha256=sha(root/'command-direction-diagnostics.json'))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'artemis-mpc-results-2026-10-31.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    pairs=[('P1_dec',7101),('P1_dec',7102),('P2_dec',7101),('P2_dec',7102)]
    for i,(key,label,color) in enumerate([('pid_float64','Tuned PID baseline','#a94c4c'),('mpc_float64','Predictive MPC, float64','#222222'),('mpc_fixed',f'Predictive MPC, {bits} bit','#186da4')]):
        rr=[next(r for r in group[key] if (r['case'],r['seed'])==pair) for pair in pairs]
        for ax,metric in zip(axes,('steady_rms_m','dv_m_s','online_cpu_s')):ax.bar(np.arange(4)+(i-1)*.24,[r[metric] for r in rr],width=.24,color=color,label=label)
    for ax,label,title in zip(axes,('Steady tracking RMS (m)','Commanded delta-v (m/s)','Host online CPU (s / 3 days)'),('Tracking','Correction cost','Host computation, including navigation')):
        ax.set(ylabel=label,title=title);ax.set_xticks(range(4),['P1','P1 outage','P2','P2 outage']);ax.legend(fontsize=7)
    fig.suptitle('Frozen MPC vs tuned PID on fresh December ARTEMIS geometry — synthetic control')
    fig.savefig(a.output/'artemis-mpc-results-2026-10-31.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('aggregate','rms_reduction_fraction','dv_reduction_fraction')},indent=2))

if __name__=='__main__':main()
