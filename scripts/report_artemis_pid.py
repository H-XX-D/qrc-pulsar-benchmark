#!/usr/bin/env python3
"""Select arithmetic precision on TRAINING cases, then report frozen evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def precision_selection(train,precision):
    if train['source_sha256']!=precision['source_sha256'] or train['protocol_sha256']!=precision['protocol_sha256']:
        raise ValueError('Source or protocol changed')
    selected=train['selection']['selected']
    cases={(c,s) for c in ('P1_sep','P2_sep') for s in (4101,4102)}
    reference={(r['case'],r['seed']):r for r in train['results'] if r['configuration']==selected}
    if set(reference)!=cases:raise ValueError('Training reference incomplete')
    rows=precision['results']
    expected={(c,s,str(b)) for c,s in cases for b in (16,24,32)}
    if len(rows)!=12 or {(r['case'],r['seed'],r['arithmetic']) for r in rows}!=expected:
        raise ValueError('Precision selection requires twelve training runs only')
    comparisons=[]
    for r in rows:
        if r['split']!='training' or r['configuration']!=selected:raise ValueError('Held-out data cannot select precision')
        ref=reference[(r['case'],r['seed'])]
        rms=abs(r['steady_rms_m']/ref['steady_rms_m']-1)
        dv=abs(r['dv_m_s']/ref['dv_m_s']-1)
        mx=abs(r['max_m']-ref['max_m'])
        faults=sum(r['core_status_counts'].get(str(k),0) for k in (1,4))
        comparisons.append(dict(case=r['case'],seed=r['seed'],bits=int(r['arithmetic']),
            rms_relative_change=rms,dv_relative_change=dv,max_error_change_m=mx,
            passed=bool(rms<=.01 and dv<=.01 and mx<=10 and not r['collision'] and faults==0)))
    passing=[bits for bits in (16,24,32) if all(r['passed'] for r in comparisons if r['bits']==bits)]
    return dict(selected_bits=min(passing) if passing else None,settings=selected,
                rule='On all four training scenarios: <=1% relative RMS and DV change, <=10m max-error change, no collision or arithmetic range faults.',
                comparisons=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path)
    p.add_argument('--select-precision',action='store_true');p.add_argument('--output',type=Path,default=Path('docs'))
    a=p.parse_args();base=a.campaign;train=read(base/'pid-train.json');precision=read(base/'pid-precision.json')
    choice=precision_selection(train,precision)
    selected_path=base/'pid-precision-selection.json'
    if a.select_precision:
        if selected_path.exists() and read(selected_path)!=choice:raise ValueError('Precision selection changed')
        selected_path.write_text(json.dumps(choice,indent=2)+'\n');print(json.dumps(choice,indent=2));return
    if not selected_path.exists() or read(selected_path)!=choice:raise ValueError('Select precision before evaluation')
    evaluated=read(base/'pid-evaluate.json')
    if train['source_sha256']!=evaluated['source_sha256'] or train['protocol_sha256']!=evaluated['protocol_sha256']:
        raise ValueError('Evaluation implementation/protocol differs from training')
    rows=evaluated['results'];expected={(c,s,b) for c in ('P1_oct','P2_oct') for s in (5101,5102) for b in ('float64','16','24','32')}
    if len(rows)!=16 or {(r['case'],r['seed'],r['arithmetic']) for r in rows}!=expected:raise ValueError('Incomplete evaluation')
    if any(r['configuration']!=train['selection']['selected'] or r['split']!='evaluation' for r in rows):raise ValueError('Evaluation settings changed')
    compact=lambda r:{k:v for k,v in r.items() if k not in ('samples','commands')}
    aggregate={b:dict(mean_steady_rms_m=float(np.mean([r['steady_rms_m'] for r in rows if r['arithmetic']==b])),
                      mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in rows if r['arithmetic']==b])),
                      worst_sampled_error_m=max(r['max_m'] for r in rows if r['arithmetic']==b),
                      arithmetic_range_faults=sum(r['core_status_counts'].get('4',0) for r in rows if r['arithmetic']==b))
               for b in ('float64','16','24','32')}
    summary=dict(classification='Definitive navigation-product replay plus separate synthetic control tests on ARTEMIS initial geometries; no FPGA device results',
        source_sha256=train['source_sha256'],protocol_sha256=train['protocol_sha256'],
        precision_selection=choice,training_selection=train['selection'],
        evaluation=sorted([compact(r) for r in rows],key=lambda r:(r['case'],r['seed'],r['arithmetic'])),aggregate=aggregate,
        replay=read(base/'replay-event-audit.json'),reference_discontinuities=read(base/'reference-jump-refinement.json'),
        receipts={n:dict(path=str(base/n),sha256=sha(base/n)) for n in ('download-manifest.json','pid-train.json','pid-precision.json','pid-evaluate.json','pid-protocol.json','pid-input-sha256.json','replay-summary.json','replay-event-audit.json','reference-jump-refinement.json')})
    fresh_path=base/'pid-accuracy-fresh.json'
    if fresh_path.exists():
        fresh=read(fresh_path);comparison=read(base/'pid-accuracy-comparison.json')
        precision2=read(base/'pid-accuracy-precision-selection.json');protocol2=read(base/'pid-accuracy-protocol.json')
        chosen=protocol2['chosen']['configuration'];bits=str(precision2['selected_bits'])
        expected={(c,s,b) for c in ('P1_nov','P2_nov') for s in (6101,6102) for b in ('float64',bits)}
        rr=fresh['results'];old=comparison['results']
        if len(rr)!=8 or {(r['case'],r['seed'],r['arithmetic']) for r in rr}!=expected:raise ValueError('Fresh evaluation incomplete')
        if len(old)!=4 or {(r['case'],r['seed']) for r in old}!={(c,s) for c in ('P1_nov','P2_nov') for s in (6101,6102)}:raise ValueError('Paired comparator incomplete')
        if any(r['configuration']!=chosen for r in rr):raise ValueError('Fresh gain changed')
        for report in (fresh,comparison):
            if report['source_sha256']!=train['source_sha256'] or report['protocol_sha256']!=sha(base/'pid-accuracy-protocol.json'):raise ValueError('Fresh source/protocol mismatch')
        grouped={'original_gain_float64':old,'revised_gain_float64':[r for r in rr if r['arithmetic']=='float64'],
                 'revised_gain_fixed':[r for r in rr if r['arithmetic']==bits]}
        stats={name:dict(mean_steady_rms_m=float(np.mean([r['steady_rms_m'] for r in group])),mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in group])),
                         maximum_sampled_error_m=max(r['max_m'] for r in group),mean_burns=float(np.mean([r['burns'] for r in group]))) for name,group in grouped.items()}
        summary['refinement']=dict(reason=protocol2['reason'],selected_configuration=chosen,selected_bits=int(bits),
            precision_training=precision2,evaluation=[compact(r) for r in rr],paired_original=[compact(r) for r in old],aggregate=stats,
            rms_reduction_fraction=1-stats['revised_gain_float64']['mean_steady_rms_m']/stats['original_gain_float64']['mean_steady_rms_m'],
            dv_reduction_fraction=1-stats['revised_gain_float64']['mean_dv_m_s']/stats['original_gain_float64']['mean_dv_m_s'])
        for n in ('pid-accuracy-protocol.json','pid-accuracy-precision.json','pid-accuracy-precision-selection.json','pid-accuracy-fresh.json','pid-accuracy-comparison.json','fresh-download-manifest.json'):
            summary['receipts'][n]=dict(path=str(base/n),sha256=sha(base/n))
        summary['fresh_replay']=[read(base/'replay-fresh'/f'P{i}_nov.json') for i in (1,2)]
    a.output.mkdir(parents=True,exist_ok=True);destination=a.output/'artemis-fpga-pid-results-2026-10-31.json'
    destination.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for case in ('P1_feb','P2_feb','P1_sep','P2_sep','P1_oct','P2_oct'):
        arr=np.load(base/'replay'/f'{case}-cram0.016.npz')
        axes[0,0].semilogy(arr['time_s'][1:]/86400,arr['error_m'][1:],label=case.replace('_',' '))
    axes[0,0].set(xlabel='Days from recorded initial state',ylabel='Position error (m)',title='Real orbit-product replay, including events')
    axes[0,0].legend(ncol=2,fontsize=8)
    candidates=train['selection']['candidates']
    for strength,color in [(0.,'#146a9c'),(.1,'#d17d13'),(.3,'#777777')]:
        rr=[r for r in candidates if r['configuration'].endswith(f'i{strength:.1f}')]
        axes[0,1].scatter([r['mean_dv_m_s'] for r in rr],[r['worst_steady_rms_m'] for r in rr],color=color,label=f'Integral strength {strength:g}')
    chosen=next(r for r in candidates if r['configuration']==train['selection']['selected'])
    axes[0,1].scatter(chosen['mean_dv_m_s'],chosen['worst_steady_rms_m'],marker='*',s=200,c='black')
    axes[0,1].set(yscale='log',xlabel='Mean commanded delta-v (m/s)',ylabel='Worst training steady RMS (m)',title='Training: gain selection')
    axes[0,1].legend(fontsize=8)
    pairs=[('P1_oct',5101),('P1_oct',5102),('P2_oct',5101),('P2_oct',5102)]
    for i,(ar,color) in enumerate([('float64','#111111'),('16','#b03c3c'),('24','#146a9c'),('32','#d17d13')]):
        rr=[next(r for r in rows if (r['case'],r['seed'],r['arithmetic'])==(c,s,ar)) for c,s in pairs]
        x=np.arange(4)+(i-1.5)*.18
        for ax,key in [(axes[1,0],'steady_rms_m'),(axes[1,1],'dv_m_s')]:ax.bar(x,[r[key] for r in rr],width=.18,color=color,label=ar if ar=='float64' else ar+' bit')
    for ax in axes[1]:
        ax.set_xticks(range(4),['P1 ordinary','P1 outage','P2 ordinary','P2 outage']);ax.legend(fontsize=8)
    axes[1,0].set(yscale='log',ylabel='Steady tracking RMS (m)',title='Held-out synthetic control: precision effect')
    axes[1,1].set(ylabel='Commanded delta-v over 3 days (m/s)',title='Held-out synthetic control: correction cost')
    fig.suptitle('ARTEMIS replay and portable PID arithmetic — distinct validation questions')
    fig.savefig(a.output/'artemis-fpga-pid-results-2026-10-31.png',dpi=180)
    plt.close(fig)
    if 'refinement' in summary:
        fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        pairs=[('P1_nov',6101),('P1_nov',6102),('P2_nov',6101),('P2_nov',6102)]
        refine=summary['refinement'];groups=[('Original six-hour gain',refine['paired_original'],'#b03c3c'),
            ('Revised one-hour, float64',[r for r in refine['evaluation'] if r['arithmetic']=='float64'],'#111111'),
            ('Revised one-hour, 24 bit',[r for r in refine['evaluation'] if r['arithmetic']==str(refine['selected_bits'])],'#146a9c')]
        for i,(label,group,color) in enumerate(groups):
            rr=[next(r for r in group if (r['case'],r['seed'])==pair) for pair in pairs]
            x=np.arange(4)+(i-1)*.23
            for axes,key in zip(ax,('steady_rms_m','dv_m_s')):axes.bar(x,[r[key] for r in rr],width=.23,label=label,color=color)
        for axes in ax:axes.set_xticks(range(4),['P1 ordinary','P1 outage','P2 ordinary','P2 outage']);axes.legend(fontsize=8)
        ax[0].set(yscale='log',ylabel='Steady RMS (m)',title='Fresh November scenarios: tracking')
        ax[1].set(ylabel='Commanded delta-v over 3 days (m/s)',title='Same cases, actuator and noise draws')
        fig.suptitle('Revised gain selected from September training; November results did not tune it')
        fig.savefig(a.output/'artemis-fpga-pid-refinement-2026-10-31.png',dpi=180);plt.close(fig)
    print(destination)


if __name__=='__main__':main()
