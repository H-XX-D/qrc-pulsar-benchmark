#!/usr/bin/env python3
"""Report the entire strategy search and the frozen January comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(rows):
    return dict(cases=len(rows),mean_rms_m=float(np.mean([r['metrics']['rms_m'] for r in rows])),
        worst_rms_m=max(r['metrics']['rms_m'] for r in rows),worst_peak_m=max(r['metrics']['peak_m'] for r in rows),
        mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in rows])),mean_burns=float(np.mean([r['burns'] for r in rows])),
        mean_direction_changes=float(np.mean([r['metrics']['direction_changes'] for r in rows])),
        mean_adjacent_reversals=float(np.mean([r['metrics']['adjacent_reversals'] for r in rows])),
        mean_online_cpu_s=float(np.mean([r['online_cpu_s'] for r in rows])),
        bad_cases=sum(r['metrics']['rms_m']>100 or r['metrics']['peak_m']>500 or r['metrics']['solver_or_screen_faults']>0 or r['collision'] for r in rows),
        tight_cases=sum(r['metrics']['rms_m']<=50 and r['metrics']['peak_m']<=300 and not r['metrics']['solver_or_screen_faults'] and not r['collision'] for r in rows))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,default=Path('docs'));p.add_argument('--training-only',action='store_true');a=p.parse_args();root=a.campaign
    training=read(root/'train.json');selection=training['selection'];protocol=read(root/'protocol.json')
    summary=dict(classification='Synthetic controlled vehicle on definitive ARTEMIS initial geometry; best among tested strategies, not a global optimum',selection=selection,
        protocol=protocol,source_sha256=training['source_sha256'],receipts={n:dict(path=str(root/n),sha256=sha(root/n)) for n in ('protocol.json','train.json','input-sha256.json','download-manifest.json')})
    if not a.training_only:
        choice=read(root/'precision-selection.json');evaluation=read(root/'evaluate.json')
        if evaluation['source_sha256']!=training['source_sha256'] or evaluation['protocol_sha256']!=training['protocol_sha256'] or evaluation['input_sha256']!=training['input_sha256']:raise ValueError('Evaluation provenance mismatch')
        if choice['training_sha256']!=sha(root/'train.json') or choice['precision_sha256']!=sha(root/'precision.json'):raise ValueError('Selection receipt changed')
        selected=list(dict.fromkeys(['baseline',*selection['selected']]))
        configs=[(n,'float64') for n in selected]+[(selection['primary'],str(choice['selected_bits']))]
        expected={(c['name'],s,n,b) for c in protocol['fresh_cases'] for s in protocol['fresh_seeds'] for n,b in configs}
        rows=evaluation['results']
        if len(rows)!=len(expected) or {(r['case'],r['seed'],r['configuration'],r['arithmetic']) for r in rows}!=expected or any(r['split']!='fresh_evaluation' for r in rows):raise ValueError('Fresh comparison incomplete')
        stats={f'{name}/{bits}':summarize([r for r in rows if r['configuration']==name and r['arithmetic']==bits]) for name,bits in configs}
        base=stats['baseline/float64'];primary=stats[selection['primary']+'/float64']
        summary.update(precision_selection=choice,evaluation=[{k:v for k,v in r.items() if k not in ('samples','dense_samples','commands','guidance_diagnostics')} for r in rows],aggregate=stats,
            primary_changes=dict(rms_fraction=primary['mean_rms_m']/base['mean_rms_m']-1,dv_fraction=primary['mean_dv_m_s']/base['mean_dv_m_s']-1,
                                 burns_fraction=primary['mean_burns']/base['mean_burns']-1,reversals_fraction=primary['mean_direction_changes']/base['mean_direction_changes']-1))
        for n in ('precision.json','precision-selection.json','evaluate.json'):summary['receipts'][n]=dict(path=str(root/n),sha256=sha(root/n))
    a.output.mkdir(parents=True,exist_ok=True);suffix='training' if a.training_only else 'results'
    (a.output/f'artemis-strategies-{suffix}-2026-10-31.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for c in selection['candidates']:
        color='#176fa0' if c['feasible'] else '#bd5353';marker='o' if c['feasible'] else 'x'
        axes[0].scatter(c['mean_burns'],c['worst_rms_m'],s=45,color=color,marker=marker)
        axes[0].annotate(c['configuration'],(c['mean_burns'],c['worst_rms_m']),fontsize=6,xytext=(3,3),textcoords='offset points')
    axes[0].axhline(50,color='grey',linestyle=':',linewidth=1);axes[0].axhline(100,color='grey',linestyle='--',linewidth=1)
    axes[0].set(yscale='log',xlabel='Mean burns per three-day training case',ylabel='Worst training steady RMS (m)',title='All training strategies, including failures')
    if not a.training_only:
        colors=plt.get_cmap('tab10')
        for i,name in enumerate(selected):
            d=stats[name+'/float64'];axes[1].scatter(d['mean_burns'],d['mean_rms_m'],s=100+300*d['mean_dv_m_s'],color=colors(i),label=f"{name}: {d['mean_dv_m_s']:.3f} m/s")
            axes[1].annotate(name,(d['mean_burns'],d['mean_rms_m']),fontsize=7,xytext=(3,5),textcoords='offset points')
        axes[1].set(xlabel='Mean burns per three-day January case',ylabel='Mean steady RMS (m)',title='Fresh holdout: bubble size reflects delta-v')
        axes[1].legend(fontsize=8,loc='best')
    else:
        for c in selection['candidates']:
            if c['feasible']:axes[1].scatter(c['mean_burns'],c['mean_dv_m_s'],label=c['configuration'])
        axes[1].set(xlabel='Mean burns',ylabel='Mean delta-v (m/s)',title='Feasible training tradeoffs');axes[1].legend(fontsize=6)
    fig.suptitle('Tight tracking, fuel use and correction frequency are separate objectives')
    fig.savefig(a.output/f'artemis-strategies-{suffix}-2026-10-31.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('selection','aggregate','primary_changes') if k in summary},indent=2))

if __name__=='__main__':main()
