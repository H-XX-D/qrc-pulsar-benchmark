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
        mean_full_arc_rms_m=float(np.mean([np.sqrt(np.mean([s['error_m']**2 for s in r['dense_samples']])) for r in rows])),
        worst_full_arc_peak_m=max(s['error_m'] for r in rows for s in r['dense_samples']),
        worst_rms_m=max(r['metrics']['rms_m'] for r in rows),worst_peak_m=max(r['metrics']['peak_m'] for r in rows),
        mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in rows])),mean_burns=float(np.mean([r['burns'] for r in rows])),
        mean_direction_changes=float(np.mean([r['metrics']['direction_changes'] for r in rows])),
        mean_adjacent_reversals=float(np.mean([r['metrics']['adjacent_reversals'] for r in rows])),
        mean_online_cpu_s=float(np.mean([r['online_cpu_s'] for r in rows])),
        bad_cases=sum(r['metrics']['rms_m']>100 or r['metrics']['peak_m']>500 or r['metrics']['solver_or_screen_faults']>0 or r['collision'] for r in rows),
        tight_cases=sum(r['metrics']['rms_m']<=50 and r['metrics']['peak_m']<=300 and not r['metrics']['solver_or_screen_faults'] and not r['collision'] for r in rows))
def failed(row):
    m=row['metrics']
    return m['rms_m'] is None or m['rms_m']>100 or m['peak_m']>500 or m['solver_or_screen_faults'] or m['collision']
def failure_detail(row):
    return {k:row[k] for k in ('case','seed','configuration','arithmetic','metrics','fixes_accepted','fixes_rejected','bias_clips','core_status_counts')}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,default=Path('docs'));p.add_argument('--training-only',action='store_true');a=p.parse_args();root=a.campaign
    training=read(root/'train.json');selection=training['selection'];protocol=read(root/'protocol.json')
    summary=dict(classification='Synthetic controlled vehicle on definitive ARTEMIS initial geometry; best among tested strategies, not a global optimum',selection=selection,
        protocol=protocol,source_sha256=training['source_sha256'],receipts={n:dict(path=str(root/n),sha256=sha(root/n)) for n in ('protocol.json','train.json','input-sha256.json','download-manifest.json')},
        training_aggregate={name:summarize([r for r in training['results'] if r['configuration']==name]) for name in protocol['settings']},
        training_failures=[failure_detail(r) for r in training['results'] if failed(r)])
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
        baseline_rows={(r['case'],r['seed']):r for r in rows if r['configuration']=='baseline' and r['arithmetic']=='float64'}
        primary_rows=[r for r in rows if r['configuration']==selection['primary'] and r['arithmetic']=='float64']
        paired={key:sum(r['metrics'][metric]<baseline_rows[(r['case'],r['seed'])]['metrics'][metric] for r in primary_rows)
                for key,metric in (('lower_rms_cases','rms_m'),('lower_dv_cases','dv_m_s'),('fewer_burn_cases','burns'))}
        primary_index={(r['case'],r['seed']):r for r in primary_rows}
        integer_rows=[r for r in rows if r['configuration']==selection['primary'] and r['arithmetic']==str(choice['selected_bits'])]
        precision_holdout={key:max(abs(r['metrics'][metric]/primary_index[(r['case'],r['seed'])]['metrics'][metric]-1) for r in integer_rows)
                           for key,metric in (('max_rms_relative_change','rms_m'),('max_dv_relative_change','dv_m_s'))}
        precision_holdout['max_burn_count_change']=max(abs(r['burns']-primary_index[(r['case'],r['seed'])]['burns']) for r in integer_rows)
        summary.update(precision_selection=choice,evaluation=[{k:v for k,v in r.items() if k not in ('samples','dense_samples','commands','guidance_diagnostics')} for r in rows],aggregate=stats,
            evaluation_failures=[failure_detail(r) for r in rows if failed(r)],
            primary_paired_comparison=paired,precision_holdout=precision_holdout,
            primary_changes=dict(rms_fraction=primary['mean_rms_m']/base['mean_rms_m']-1,dv_fraction=primary['mean_dv_m_s']/base['mean_dv_m_s']-1,
                                 burns_fraction=primary['mean_burns']/base['mean_burns']-1,reversals_fraction=primary['mean_direction_changes']/base['mean_direction_changes']-1))
        qualified=[name for name in selected if name!='baseline' and stats[name+'/float64']['bad_cases']==0]
        development=min(qualified,key=lambda name:(stats[name+'/float64']['worst_rms_m'],stats[name+'/float64']['mean_dv_m_s'])) if qualified else None
        summary['assessment']=dict(training_primary_passes_fresh_acceptance=primary['bad_cases']==0,
            all_fresh_tight_cases_achieved=any(d['tight_cases']==d['cases'] for d in stats.values()),
            qualified_float_finalists=qualified,development_candidate=development,
            interpretation='Development preference follows the held-out comparison; no retuning occurred and further independent confirmation is needed.')
        for n in ('precision.json','precision-selection.json','evaluate.json'):summary['receipts'][n]=dict(path=str(root/n),sha256=sha(root/n))
        extra_path=root/'sparse05-hardware-qualification.json'
        if extra_path.exists() and 'checks' in read(extra_path):
            extra=read(extra_path)
            if extra['source_sha256']!=training['source_sha256'] or extra['training_sha256']!=sha(root/'train.json') or extra['evaluation_sha256']!=sha(root/'evaluate.json'):raise ValueError('Additional arithmetic provenance mismatch')
            reference_keys={(r['case'],r['seed']) for d in (training,evaluation) for r in d['results'] if r['configuration']==extra['configuration'] and r['arithmetic']=='float64'}
            if len(extra['results'])!=12 or {(r['case'],r['seed']) for r in extra['results']}!=reference_keys:raise ValueError('Additional arithmetic matrix incomplete')
            summary['additional_arithmetic_qualification']=dict(classification=extra['classification'],configuration=extra['configuration'],
                bits=extra['bits'],fraction=extra['fraction'],arithmetic_passed=extra['arithmetic_passed'],checks=extra['checks'],
                reused_january_aggregate=summarize([r for r in extra['results'] if r['split']=='fresh_evaluation']))
            summary['receipts'][extra_path.name]=dict(path=str(extra_path),sha256=sha(extra_path))
    a.output.mkdir(parents=True,exist_ok=True);suffix='training' if a.training_only else 'results'
    (a.output/f'artemis-strategies-{suffix}-2026-10-31.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if not a.training_only:
        if development:
            d=stats[development+'/float64']
            assessment_text=f"**The January result favors {development} for further development.** Relative to baseline: mean RMS change {100*(d['mean_rms_m']/base['mean_rms_m']-1):+.2f}%, mean delta-v change {100*(d['mean_dv_m_s']/base['mean_dv_m_s']-1):+.2f}% and mean burn-count change {100*(d['mean_burns']/base['mean_burns']-1):+.2f}%. It passed the broader acceptance limits in all {d['cases']} cases."
        else:assessment_text='No new finalist passed the broader acceptance limits in every January case.'
        assessment_text+=f" The training-primary {selection['primary']} failed those limits in {primary['bad_cases']} of {primary['cases']} cases."
        if not summary['assessment']['all_fresh_tight_cases_achieved']:assessment_text+=' No tested policy met the tighter 50 m RMS / 300 m peak envelope in every January case.'
        assessment_text+=' Parameters remain frozen; this development preference interprets the held-out results and is not an independently confirmed deployment choice.'
        lines=['# Executed ARTEMIS MPC strategy comparison', '',
               'The frozen September search selected **'+selection['primary']+'** as the balanced and least-fuel qualified setting, and **'+selection['choices']['tightest']+'** for tightest worst-case tracking. Selection preceded the January evaluation.', '',
               assessment_text, '',
               'These are three-day synthetic controlled vehicles initialized from definitive ARTEMIS P1/P2 states. The reference is propagated from the initial state; later observed spacecraft states do not enter guidance. Tracking is scored every five minutes after the first 12 hours. Delta-v and burns cover the full three days.', '',
               '## Fresh January evaluation', '',
               'Each row covers all eight cases, including four with a six-hour navigation outage and a corrupted fix. “Tight” requires each case to meet 50 m steady RMS and 300 m sampled steady peak. These are sampled tracking bounds, not continuous-time safety guarantees.', '',
               '| Policy / arithmetic | Mean RMS (m) | Worst RMS (m) | Worst peak (m) | Mean delta-v (m/s) | Mean burns | Mean >90° changes | Mean host CPU (s) | Tight cases |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,d in stats.items():
            lines.append(f"| {name} | {d['mean_rms_m']:.3f} | {d['worst_rms_m']:.3f} | {d['worst_peak_m']:.3f} | {d['mean_dv_m_s']:.6f} | {d['mean_burns']:.2f} | {d['mean_direction_changes']:.2f} | {d['mean_online_cpu_s']:.2f} | {d['tight_cases']}/{d['cases']} |")
        lines+=['', 'Including the initial recovery, mean full-three-day RMS is '+
                '; '.join(f"{name}: {d['mean_full_arc_rms_m']:.3f} m" for name,d in stats.items())+'. '+
                f"The largest full-arc sampled error across these runs is {max(d['worst_full_arc_peak_m'] for d in stats.values()):.3f} m. These additional figures include the deliberately perturbed starting state."]
        lines+=['', 'Direction changes count the commanded directions of successive burns separated by more than 90 degrees; pointing-error perturbations are not included in this count. It is not an oscillation or stability proof. Host CPU includes navigation and guidance on this machine, and excludes the independent simulated plant. It is not FPGA latency.', '',
                '![All strategy tradeoffs](artemis-strategies-results-2026-10-31.png)', '',
                '![Every fresh tracking case](artemis-strategies-tracking-2026-10-31.png)', '',
                '## Full training search', '',
                '| Setting | Qualified | Worst RMS (m) | Worst peak (m) | Mean delta-v (m/s) | Mean burns |',
                '|---|---|---:|---:|---:|---:|']
        for d in selection['candidates']:
            lines.append(f"| {d['configuration']} | {'yes' if d['feasible'] else 'no'} | {d['worst_rms_m']:.3f} | {d['worst_peak_m']:.3f} | {d['mean_dv_m_s']:.6f} | {d['mean_burns']:.2f} |")
        lines+=['', 'Qualification requires all four September cases to meet 100 m steady RMS and 500 m sampled steady peak, with no collision, solver, arithmetic or first-burn screen failure. The primary and fewest-burn choices use the tighter 50 m / 300 m envelope when available. The 100 m trigger lies on the training Pareto set and uses fewer burns, but missed the tighter RMS envelope; it was not a frozen January finalist.', '',
                'Four failed settings rejected 38–59 navigation fixes in the difficult P2 outage case, versus one rejected corrupted fix for successful settings. The rejection gate can leave the estimate stale, which inhibits corrections. These records expose a coupled estimation/control failure; isolating its initiating cause requires an estimator-ablation experiment.', '',
                '## Arithmetic and scope', '',
                f"Selected integer width: **{choice['selected_bits']} total bits / {choice['selected_bits']-8} fractional bits**, for the QP core only. The 24-bit variant failed the predeclared 1% tracking-change gate in one training case; no January result was used to select precision.", '',
                'The primary uses a two-hour horizon with four 30-minute slots, a 60 m predicted coast trigger, sparse weight 0.1, a 0.001 m/s minimum requested correction, and six fixed reweighting/refinement solves of 256 QP iterations when optimization is required. Navigation, force propagation, QP assembly, the trigger and burn screening remain host operations. No FPGA device, synthesis result or measured resource/power advantage is claimed.', '',
                'The experiment optimizes corrections about a fixed planned orbit. It establishes neither a globally minimum-fuel transfer trajectory nor a separate benefit from shape-sphere coordinates. Three-day scenarios do not validate month-long behavior, real attitude/thruster allocation or emergency mission recovery.', '',
                '[Protocol and method](artemis-mpc-strategies.md) · [Machine-readable results and receipts](artemis-strategies-results-2026-10-31.json)', '']
        if 'additional_arithmetic_qualification' in summary:
            x=summary['additional_arithmetic_qualification'];checks=x['checks']
            lines+=['## Additional arithmetic qualification', '',
                    f"After examining the held-out results, the already frozen {x['configuration']} finalist was checked with {x['bits']}/{x['fraction']} integer arithmetic on all 12 September and January cases. All arithmetic gates passed: **{x['arithmetic_passed']}**. Maximum relative RMS change: {100*max(c['rms_relative_change'] for c in checks):.6f}%; delta-v change: {100*max(c['dv_relative_change'] for c in checks):.6f}%; largest burn-count difference: {max(c['burn_count_change'] for c in checks)}. These are reused cases and an arithmetic comparison, not additional independent control-validation evidence.", '']
        (a.output/'artemis-strategies-results-2026-10-31.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for i,c in enumerate(selection['candidates']):
        color='#176fa0' if c['feasible'] else '#bd5353';marker='o' if c['feasible'] else 'x'
        axes[0].scatter(c['worst_rms_m'],i,s=45,color=color,marker=marker)
    axes[0].set_yticks(range(len(selection['candidates'])),[c['configuration'] for c in selection['candidates']],fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(50,color='grey',linestyle=':',linewidth=1,label='Tight RMS limit')
    axes[0].axvline(100,color='grey',linestyle='--',linewidth=1,label='Qualification RMS limit')
    axes[0].set(xscale='log',xlabel='Worst training steady RMS (m)',title='All 15 strategies; red crosses fail qualification')
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
    if not a.training_only:
        fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,constrained_layout=True)
        for column,case in enumerate(protocol['fresh_cases']):
            for row_index,seed in enumerate(protocol['fresh_seeds']):
                axis=axes[row_index,column]
                for i,name in enumerate(selected):
                    r=next(r for r in rows if r['case']==case['name'] and r['seed']==seed and r['configuration']==name and r['arithmetic']=='float64')
                    points=[s for s in r['dense_samples'] if s['time_s']>=43200]
                    axis.plot([s['time_s']/86400 for s in points],
                              [s['error_m'] for s in points],label=name,color=colors(i),linewidth=1)
                if seed%100==2:axis.axvspan(1.5,1.75,color='grey',alpha=.2,label='Navigation outage')
                axis.set(xlim=(.5,3),ylim=(0,None),title=f"{case['name']} / seed {seed}",xlabel='Days',ylabel='Position error (m)')
                axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        fig.suptitle('All eight fresh cases: five-minute truth scoring after initial recovery')
        fig.savefig(a.output/'artemis-strategies-tracking-2026-10-31.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('selection','aggregate','primary_changes') if k in summary},indent=2))

if __name__=='__main__':main()
