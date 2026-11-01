#!/usr/bin/env python3
"""Audit both frozen revisions, including the failure that prompted supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from artemis_resilience_campaign import read,save,sha,REPO,scenarios,select
from artemis_supervised_campaign import cases

def check_row(row):
    m=row['metrics'];points=row['dense_samples'];steady=[p['error_m'] for p in points if p['time_s']>=43200]
    assert np.isclose(m['rms_m'],np.sqrt(np.mean(np.square(steady))),rtol=1e-13)
    assert m['peak_m']==max(steady)
    all_errors=[p['error_m'] for p in points]
    assert np.isclose(m['full_rms_m'],np.sqrt(np.mean(np.square(all_errors))),rtol=1e-13)
    assert m['full_peak_m']==max(all_errors)
    assert m['completed']==(not m['collision'] and points[-1]['time_s']==row['scenario']['days']*86400)
    fired=[c for c in row['commands'] if c['burn_s']>0]
    assert len(fired)==m['burns']
    assert np.isclose(m['dv_m_s'],sum(np.linalg.norm(c['dv']) for c in fired),rtol=1e-8,atol=1e-8)
    assert m['direction_changes']==sum(np.dot(a['dv'],b['dv'])<0 for a,b in zip(fired[:-1],fired[1:]))
    assert m['status_counts']=={str(s):sum(c['status']==s for c in row['commands']) for s in {c['status'] for c in row['commands']}}
    assert m['max_cache_sizes'][0]<=12 and m['max_cache_sizes'][1]<=12 and m['max_cache_sizes'][2]<=256
    if row['profile']=='supervised_sparse_05':
        peak=max((sum(np.linalg.norm(c['dv']) for c in fired if end['time_s']-86400<c['time_s']<=end['time_s']) for end in fired),default=0.)
        assert peak<=row['routine_limits']['issued_dv_per_day_m_s']+1e-10
        assert all(not np.linalg.norm(c['dv']) for c in row['commands'] if c['status'])
        return float(peak)

def aggregate(rows):
    return dict(cases=len(rows),mean_rms_m=float(np.mean([r['metrics']['rms_m'] for r in rows])),
        worst_rms_m=max(r['metrics']['rms_m'] for r in rows),worst_peak_m=max(r['metrics']['peak_m'] for r in rows),
        mean_dv_m_s=float(np.mean([r['metrics']['dv_m_s'] for r in rows])),mean_burns=float(np.mean([r['metrics']['burns'] for r in rows])),
        tracking_pass_cases=sum(bool(r['metrics']['completed'] and r['metrics']['rms_m']<=100 and r['metrics']['peak_m']<=500) for r in rows),
        screen_or_numeric_fault_cases=sum(bool(r['metrics']['numerical_faults']) for r in rows),
        recovery_cases=sum(r['metrics']['status_counts'].get('64',0)>0 for r in rows))

def diagnostics(row):
    """Report reference feasibility and recovery events without assigning causes."""
    recovery=[c['time_s']/86400 for c in row['commands'] if c['status']==64]
    screens=[c['time_s']/86400 for c in row['commands'] if c['status']==32]
    return dict(nominal_min_clearance_m=row.get('nominal_min_clearance_m'),
        nominal_below_controller_moon_floor=(row['nominal_min_clearance_m']<50000
            if 'nominal_min_clearance_m' in row else None),
        first_recovery_required_day=recovery[0] if recovery else None,
        first_clearance_rejection_day=screens[0] if screens else None,
        maximum_fix_age_h=max(p['age_s'] for p in row['samples'])/3600,
        maximum_navigation_error_m=max(p['navigation_error_m'] for p in row['samples']))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('earlier',type=Path);p.add_argument('supervised',type=Path);a=p.parse_args()
    roots={'earlier':a.earlier,'supervised':a.supervised};data={};receipts={};budget_peaks={}
    dev,fresh,_=scenarios();regression,endurance=cases()
    expected={('earlier','develop'):{(s.name,n,'float64') for s in dev for n in ('legacy','decay_only','guarded','decay_guarded')},
              ('earlier','precision'):{(s.name,'legacy','32') for s in dev},
              ('earlier','fresh'):{(s.name,'legacy',b) for s in fresh for b in ('float64','32')},
              ('supervised','regression'):{(s.name,'supervised_sparse_05',b) for s in regression for b in ('float64','32')},
              ('supervised','endurance'):{(s.name,'supervised_sparse_05',b) for s in endurance for b in ('float64','32')}}
    for label,root in roots.items():
        sources=read(root/'source-sha256.json');inputs=read(root/'input-sha256.json')
        for name,digest in sources.items():
            assert sha(root/'source'/name)==digest,name
            if label=='supervised':assert sha(REPO/name)==digest,name
        for name,digest in inputs.items():assert sha(Path(name))==digest,name
        for (group,phase),wanted in expected.items():
            if group!=label:continue
            report=read(root/f'{phase}.json');rows=report['results']
            assert report['source_sha256']==sources and report['input_sha256']==inputs
            assert report['protocol_sha256']==sha(root/'protocol.json')
            runner='artemis_resilience_campaign.py' if label=='earlier' else 'artemis_supervised_campaign.py'
            assert report['runner_sha256']==sha(root/'source'/'scripts'/runner)
            assert len(rows)==len(wanted) and {(r['scenario']['name'],r['profile'],r['arithmetic']) for r in rows}==wanted
            for row in rows:
                peak=check_row(row)
                if peak is not None:budget_peaks[row['scenario']['name']+'/'+row['arithmetic']]=peak
            data[label+'/'+phase]=report
            receipts[label+'/'+phase]=dict(path=str(root/f'{phase}.json'),sha256=sha(root/f'{phase}.json'))
    assert select(data['earlier/develop']['results'])==data['earlier/develop']['selection']
    assert data['earlier/precision']['arithmetic_passed']
    assert read(a.supervised/'protocol.json')['previous_fresh_sha256']==sha(a.earlier/'fresh.json')
    assert '362 passed' in (a.earlier/'pytest.log').read_text()
    assert '366 passed' in (a.supervised/'pytest.log').read_text()
    for name in ('native-fault24.log','native-fault32.log'):
        log=(a.earlier/name).read_text();assert 'cases=600' in log and 'runtime error:' not in log and 'AddressSanitizer' not in log
    assert 'Configuration checks passed' in (a.earlier/'hls-config-check.log').read_text()
    wheels=list((a.supervised/'wheels').glob('*.whl'));assert len(wheels)==1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in ('routine_supervisor','resilience_benchmark','resilient_mpc','predictive_mpc','fpga_mpc','adaptive_control'):
            path=f'qrc_pulsar_bench/{name}.py';assert hashlib.sha256(wheel.read(path)).hexdigest()==sources[path]
    stats={}
    for phase in ('earlier/fresh','supervised/regression','supervised/endurance'):
        for days in sorted({r['scenario']['days'] for r in data[phase]['results']}):
            for bits in ('float64','32'):
                rows=[r for r in data[phase]['results'] if r['scenario']['days']==days and r['arithmetic']==bits]
                stats[f'{phase}/{days:g}days/{bits}']=aggregate(rows)
    count=sum(len(d['results']) for d in data.values());days=sum(r['scenario']['days'] for d in data.values() for r in d['results'])
    verification=dict(status='evidence_verified',run_counts={k:len(d['results']) for k,d in data.items()},total_runs=count,total_simulated_days=days,
        current_source_files=len(sources),input_files=len(inputs),tests_passed=366,native_sanitized_fault_cases=1200,
        sources_and_snapshots_verified=True,complete_matrices=True,recomputed_metrics=True,rolling_issued_budget_peaks=budget_peaks,
        hardware_synthesis_performed=False,wheel_sha256=sha(wheels[0]),receipts=receipts)
    save(a.supervised/'verification.json',verification)
    save(a.supervised/'selected-controller.json',dict(controller='RoutineMPC',strategy='sparse_05',routine_limits=read(a.supervised/'protocol.json')['routine_limits'],
        arithmetic=dict(bits=32,fraction=24),hardware_device=None,classification='Research candidate with bounded routine authority; recovery planner and flight validation remain absent',
        implementation_partition=dict(portable_integer_core='Bounded QP solve',host='Navigation, forces, QP formation, nonlinear screen and routine supervisor'),
        remaining_limitations=['Reference viability against keep-out constraints', 'Navigation reacquisition after rejected fixes',
            'Validated recovery trajectories', 'Radiation/reset state integrity', 'Synthesis, RTL validation and hardware timing/power'],
        aggregates=stats,source_sha256=sources,receipts=receipts))
    compact=dict(classification='Synthetic controlled vehicles on real initial ARTEMIS geometry; no future spacecraft states read',
        revision_reason=read(a.supervised/'protocol.json')['reason'],development_selection=data['earlier/develop']['selection'],aggregate=stats,verification=verification,
        diagnostics={k:[dict(case=r['scenario']['name'],arithmetic=r['arithmetic'],**diagnostics(r)) for r in d['results']] for k,d in data.items()},
        results={k:[{n:v for n,v in r.items() if n not in ('samples','dense_samples','commands')} for r in d['results']] for k,d in data.items()})
    dest=REPO/'docs';save(dest/'artemis-hardening-results-2026-10-31.json',compact)
    lines=['# MPC hardening and extended mission results','',
        'The first revision hardened solver/navigation interfaces and compared four outage policies. It retained the original sparse MPC. Seven-day fresh tests then exposed runaway routine correction demand and numerical range failures. The second revision adds an explicit tracking-error envelope and rolling issued-delta-v budget. The failing evidence remains included.','',
        'The second revision replays a benign and a failed case, then uses new January 20 initial geometries and seeds for two 30-day missions, each in float64 and 32/24 integer arithmetic. Limits were frozen before these endurance runs. Replays are diagnostics, not fresh evidence. The earlier proposed endurance phase was replaced by this supervised revision after the fresh failure.','',
        'All spacecraft motion and sensor fixes after the single reconstructed ARTEMIS initial state are synthetic. The initial orbit product may itself incorporate later historical tracking. Later planetary ephemerides supply forces; later spacecraft states never enter guidance.','',
        '## Development alternatives','', '| Profile | Worst RMS (m) | Worst peak (m) | Mean delta-v (m/s) | Mean burns | Failed cases |','|---|---:|---:|---:|---:|---:|']
    for name,s in data['earlier/develop']['selection']['table'].items():
        lines.append(f"| {name} | {s['worst_rms_m']:.3f} | {s['worst_peak_m']:.3f} | {s['mean_dv_m_s']:.5f} | {s['mean_burns']:.2f} | {s['failed_cases']}/{s['cases']} |")
    for phase in ('earlier/fresh','supervised/regression','supervised/endurance'):
        lines+=['','## '+phase,'','| Case | Arithmetic | Days | Steady RMS (m) | Steady peak (m) | Delta-v (m/s) | Burns | Recovery-required updates | Solver/screen faults |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in data[phase]['results']:
            s=r['scenario'];m=r['metrics'];lines.append(f"| {s['name']} | {r['arithmetic']} | {s['days']:g} | {m['rms_m']:.3f} | {m['peak_m']:.3f} | {m['dv_m_s']:.5f} | {m['burns']} | {m['status_counts'].get('64',0)} | {m['numerical_faults']} |")
    lines+=['','Steady metrics use five-minute samples after the first 12 hours. Full-arc errors are retained in JSON. Delta-v and burns cover the complete simulation. Tracking limits are 100 m steady RMS / 500 m sampled peak; tighter targets are 50 m / 300 m. Completion does not mean the mission remained within those limits.','',
        'The failed seven-day P1 reference has a sampled minimum lunar clearance of 37.8 km, below the controller\'s 50 km first-burn clearance floor. The replay first rejects a burn at day 1.8333. Returning navigation fixes are repeatedly rejected before the estimated tracking error triggers the recovery latch at day 2.5625. These observations identify reference-feasibility and navigation-reacquisition limitations; they do not isolate a single causal explanation for the failure. Routine inhibition reduces expenditure but does not constitute fuel-efficient mission completion.','',
        'The new 30-day runs both meet the 100 m RMS / 500 m peak tracking limits in both arithmetic formats. P2 also meets the tighter 50 m / 300 m limits without a screen or numerical rejection. P1 does not pass the no-rejection gate: its nominal reference reaches 45.9 km lunar clearance and it rejects first burns on days 26.125 and 27.25. Neither run enters recovery-required or budget-wait mode. Successful tracking of an inadmissible reference is not mission feasibility.','',
        'Status 32 is a first-burn clearance rejection, not a floating-point arithmetic failure. Status 64 means the estimated state left the routine local-control envelope and the controller latched a recovery-required state. Status 128 waits for rolling issued-delta-v headroom. Zero output is not a certified safe trajectory or an emergency maneuver. The operator or a separately validated recovery planner must provide a viable plan before explicit rearming.','',
        f"Verification: {count} runs / {days:g} simulated days, 366 tests, 1,200 native sanitizer fault/range cases, current source and frozen-snapshot hashes, complete case matrices and recomputed metrics. The integer QP core is host-tested; no FPGA synthesis, RTL timing or power measurements were performed.",'',
        '![Thirty-day supervised tracking](artemis-hardening-tracking-2026-10-31.png)','']
    (dest/'artemis-hardening-results-2026-10-31.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for i,scenario in enumerate(endurance):
        for bits in ('float64','32'):
            r=next(r for r in data['supervised/endurance']['results'] if r['scenario']['name']==scenario.name and r['arithmetic']==bits)
            points=[p for p in r['dense_samples'] if p['time_s']>=43200]
            axes[0,i].plot([p['time_s']/86400 for p in points],[max(p['error_m'],.001) for p in points],label=bits,linewidth=.8,linestyle='-' if bits=='float64' else '--')
            fired=[c for c in r['commands'] if c['burn_s']>0]
            spent=np.cumsum([np.linalg.norm(c['dv']) for c in fired])
            axes[1,i].step([0]+[c['time_s']/86400 for c in fired]+[scenario.days],
                [0]+list(spent)+[float(spent[-1]) if len(spent) else 0],where='post',label=bits,linestyle='-' if bits=='float64' else '--')
            if bits=='float64':
                recovery=diagnostics(r)['first_recovery_required_day']
                if recovery is not None:
                    for ax in axes[:,i]:ax.axvline(recovery,color='firebrick',linestyle=':',label='Recovery required')
        for left,right in scenario.outages:axes[0,i].axvspan(left/24,right/24,color='grey',alpha=.2)
        axes[0,i].axhline(500,color='black',linestyle='--',linewidth=.7,label='500 m peak limit')
        axes[0,i].set(title=scenario.name,xlabel='Days',ylabel='Position error (m)',yscale='log');axes[0,i].legend()
        axes[1,i].set(xlabel='Days',ylabel='Cumulative issued delta-v (m/s)');axes[1,i].legend()
    fig.suptitle('Thirty-day supervised MPC: tracking and correction demand')
    fig.savefig(dest/'artemis-hardening-tracking-2026-10-31.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(aggregate=stats,verification=verification),indent=2))

if __name__=='__main__':main()
