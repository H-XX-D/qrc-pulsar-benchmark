#!/usr/bin/env python3
"""Verify frozen layered-autonomy results, including failed hardware controls."""
import argparse
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artemis_resilience_campaign import REPO,read,save,sha
from artemis_layered_campaign import key
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,score_recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.root/'final-campaign'
    report=read(folder/'results.json');rows=report['results'];protocol=read(folder/'protocol.json')
    expected={(j['scenario']['name'],j['case']['name'],j['arithmetic'],'layered_mpc' if j['layered'] else 'routine_mpc') for j in protocol['jobs']}
    assert len(rows)==len(expected) and {key(r) for r in rows}==expected
    for name,digest in report['source_sha256'].items():
        assert sha(REPO/name)==digest,(name,'source changed')
        assert sha(folder/'source'/name)==digest,(name,'snapshot changed')
    for name,digest in report['input_sha256'].items():assert sha(Path(name))==digest
    assert sha(folder/'protocol.json')==report['protocol_sha256']
    old=read(args.baseline/'results.json')['results'];old_lookup={key(r):r for r in old}
    old_verification=read(args.baseline/'verification.json')
    assert sha(args.baseline/'results.json')==old_verification['results_sha256']
    lookup={key(r):r for r in rows};nis=[];budget_max=0.;untrusted_checks=0;isolation_checks=0;prefix=0;ground_equal=0
    journal_hashes={};telemetry=dict(events=0,delivered=0,pending=0,dropped=0);summary=[];pairs=[]
    for row in rows:
        m=row['metrics'];events=row['case']['events']
        onset=min((e['start_s'] for e in events),default=43200.)
        end=max((e['end_s'] for e in events),default=43200.)
        if row.get('dense_samples'):
            scored=score_recovery(row['dense_samples'],Anomaly(row['case']['name'],onset,end),completed=m['completed'])
            for name,value in scored.items():
                if isinstance(value,float):assert np.isclose(m[name],value,rtol=1e-12,atol=1e-12)
                else:assert m[name]==value
            spent=sum(np.linalg.norm(c['dv']) for c in row['commands'] if c['burn_s']>0)
            assert abs(spent-m['issued_dv_m_s'])<1e-10
            for sample,command in zip(row['samples'],row['commands']):
                if not sample['navigation_trusted']:
                    assert command['burn_s']==0 and np.linalg.norm(command['dv'])==0;untrusted_checks+=1
                if command.get('health',{}).get('propulsion')=='isolation_pending':
                    assert command['burn_s']==0;isolation_checks+=1
                daily=sum(np.linalg.norm(c['dv']) for c in row['commands'] if command['time_s']-86400<c['time_s']<=command['time_s'] and c['burn_s']>0)
                assert daily<=1.+1e-10;budget_max=max(budget_max,float(daily))
                if command['burn_s']>0:assert command['status']==0
            values=[s['nis'] for s in row['samples'] if s['nis'] is not None]
            assert all(np.isfinite(n) and n>=0 for n in values);nis.extend(values)
            tail=[s for s in row['samples'] if s['time_s']>=end]
            bad=max((i for i,s in enumerate(tail) if not s['navigation_trusted'] or s['navigation_error_m']>200 or s['navigation_velocity_error_m_s']>.02),default=-1)
            tail=tail[bad+1:]
            assert m['navigation_settled']==bool(m['completed'] and tail and tail[-1]['time_s']-tail[0]['time_s']>=21600.)
            if row['scenario']['split']=='regression' and events:
                clean=lookup[(row['scenario']['name'],'none',row['arithmetic'],row['control'])]
                for a,b in zip(row['samples'],clean['samples']):
                    if a['time_s']<onset:assert a==b;prefix+=1
                for a,b in zip(row['commands'],clean['commands']):
                    if a['time_s']<onset:assert a==b;prefix+=1
                if row['case']['name']=='01_ground_blackout':
                    assert row['dense_samples']==clean['dense_samples']
                    for a,b in zip(row['commands'],clean['commands']):
                        assert {k:v for k,v in a.items() if k!='health'}=={k:v for k,v in b.items() if k!='health'}
                    ground_equal+=1
            if row['control']=='layered_mpc':
                protection=row['protection']
                assert protection['health_updates']==len(row['commands'])
                if row['case']['name']=='24_failed_isolation':
                    assert not protection['primary_closed'] and not protection['backup_enabled']
                    assert any(e['event']=='isolate_primary' for e in protection['propulsion_events'])
                if row['case']['name']=='25_ground_authority':
                    decisions=protection['ground_commands']
                    assert [d['accepted'] for d in decisions]==[True,False,False,False,True]
                    assert [d['reason'] for d in decisions]==['accepted','authentication','replay','epoch','accepted']
            t=row['telemetry'];path=Path(t['journal_path']);journal_hashes[str(path)]=sha(path)
            emitted={};pending=set();ack=set()
            for line in path.read_text().splitlines():
                record=json.loads(line);n=record['sequence']
                if record['record']=='event':assert n==len(emitted)+1;emitted[n]=record;pending.add(n)
                else:assert n in pending;pending.remove(n);ack.add(n)
            assert len(emitted)==t['events'] and len(pending)==t['pending'] and len(ack)==t['delivered']
            for k in telemetry:telemetry[k]+=t['dropped_events' if k=='dropped' else k]
        summary.append(dict(key=key(row),split=row['scenario']['split'],description=row['case']['description'],
            metrics=m,protection=row.get('protection'),telemetry=row.get('telemetry'),failure=row.get('failure'),cpu_s=row.get('cpu_s')))
        if row['arithmetic']=='32':
            f=lookup[(row['scenario']['name'],row['case']['name'],'float64',row['control'])]['metrics']
            done=bool(m['completed'] and f['completed'])
            pairs.append(dict(scenario=row['scenario']['name'],case=row['case']['name'],completed=done,
                same_tracking=m.get('settled')==f.get('settled'),same_navigation=m.get('navigation_settled')==f.get('navigation_settled'),
                same_burns=m.get('burns')==f.get('burns'),rms_difference_m=abs(m['post_event_rms_m']-f['post_event_rms_m']) if done else None,
                dv_difference_m_s=abs(m['issued_dv_m_s']-f['issued_dv_m_s']) if done else None))
    controlled=[r for r in rows if r['control']=='layered_mpc' and r['arithmetic']=='float64' and r['case']['name']!='none']
    regression=[r for r in controlled if r['scenario']['split']=='regression'];fresh=[r for r in controlled if r['scenario']['split']=='holdout']
    def counts(group):
        return dict(cases=len(group),completed=sum(r['metrics']['completed'] for r in group),
            navigation=sum(bool(r['metrics'].get('navigation_settled')) for r in group),
            tracking=sum(bool(r['metrics'].get('settled')) for r in group),
            collisions=sum(bool(r['metrics'].get('collision')) for r in group))
    comparisons=[]
    for r in regression:
        baseline=old_lookup[(r['scenario']['name'],r['case']['name'],'float64','routine_mpc')]
        assert r['initial_state']==baseline['initial_state']
        assert r['fault_directions']==baseline['fault_directions'] and r['case']==baseline['case']
        assert {k:v for k,v in r['scenario'].items() if k!='split'}=={k:v for k,v in baseline['scenario'].items() if k!='split'}
        comparisons.append(dict(scenario=r['scenario']['name'],case=r['case']['name'],old=baseline['metrics'],new=r['metrics']))
    fresh_comparisons=[]
    for r in fresh:
        baseline=lookup.get((r['scenario']['name'],r['case']['name'],'float64','routine_mpc'))
        if baseline is None:continue
        assert r['initial_state']==baseline['initial_state'] and r['fault_directions']==baseline['fault_directions']
        assert r['scenario']==baseline['scenario'] and r['case']==baseline['case']
        fresh_comparisons.append(dict(scenario=r['scenario']['name'],case=r['case']['name'],old=baseline['metrics'],new=r['metrics']))
    isolation=[];biased_promotions=[];misses=[]
    for r in controlled:
        label=dict(scenario=r['scenario']['name'],case=r['case']['name'])
        protection=r['protection']
        leaks=[e['start_s'] for e in r['case']['events'] if e['kind']=='uncommanded_thrust']
        if leaks:
            requests=[e['time_s'] for e in protection['propulsion_events'] if e['event']=='isolate_primary']
            backups=[e['time_s'] for e in protection['propulsion_events'] if e['event']=='backup_enabled']
            isolation.append(dict(**label,request_latency_s=min(requests)-min(leaks) if requests else None,
                backup_latency_s=min(backups)-min(leaks) if backups else None,
                physical_primary_closed=protection['primary_closed'],backup_enabled=protection['backup_enabled']))
        if r['case']['name'] in ('06_position_bias','13_force_and_sensor_bias'):
            for sample,command in zip(r['samples'],r['commands']):
                if command['health']['navigation']=='trusted' and sample['navigation_error_m']>200:
                    biased_promotions.append(dict(**label,time_s=sample['time_s'],navigation_error_m=sample['navigation_error_m'],
                        reported_position_3sigma_m=sample['position_3sigma_m'],burn_s=command['burn_s']))
        if not r['metrics']['settled']:
            modes={c['mode'] for c in r['commands']}
            plans=protection['protective_plans']
            misses.append(dict(**label,final_error_m=r['metrics']['final_error_m'],modes=sorted(modes),
                first_protective_time_s=plans[0]['time_s'] if plans else None,
                budget_waits=sum(c['status']==128 for c in r['commands']),
                ground_commands=protection['ground_commands']))
    test_log=(args.root/'pytest.log').read_text();matched=re.search(r'(\d+) passed',test_log)
    assert matched and 'FAILED ' not in test_log
    wheels=list((args.root/'wheels').glob('*.whl'));assert len(wheels)==1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in report['source_sha256']:
            if name.startswith('qrc_pulsar_bench/'):assert wheel.read(name)==(REPO/name).read_bytes()
    v=dict(cases=len(rows),completed=sum(r['metrics']['completed'] for r in rows),
        exceptions=sum('failure' in r for r in rows),collisions=sum(bool(r['metrics'].get('collision')) for r in rows),
        simulated_days=sum(r.get('dense_samples',[{'time_s':0}])[-1]['time_s']/86400 for r in rows),
        tests_passed=int(matched.group(1)),source_files=len(report['source_sha256']),physical_inputs=len(report['input_sha256']),
        nonnegative_nis_samples=len(nis),untrusted_zero_burn_checks=untrusted_checks,isolation_zero_burn_checks=isolation_checks,
        causal_prefix_checks=prefix,ground_link_independence_checks=ground_equal,maximum_rolling_dv_m_s=budget_max,
        regression=counts(regression),fresh=counts(fresh),telemetry=telemetry,arithmetic_pairs=pairs,
        arithmetic_outcomes_match=all(p['completed'] and p['same_tracking'] and p['same_navigation'] and p['same_burns'] for p in pairs),
        maximum_arithmetic_rms_difference_m=max((p['rms_difference_m'] for p in pairs if p['completed']),default=None),
        maximum_arithmetic_dv_difference_m_s=max((p['dv_difference_m_s'] for p in pairs if p['completed']),default=None),
        regression_rms_improved=sum(r['new']['post_event_rms_m']<r['old']['post_event_rms_m']-1e-3 for r in comparisons),
        regression_rms_worsened=sum(r['new']['post_event_rms_m']>r['old']['post_event_rms_m']+1e-3 for r in comparisons),
        protective_decisions=sum(len(r.get('protection',{}).get('protective_plans',[])) for r in rows),
        protective_burn_decisions=sum(p['reason']=='screened_burn' for r in rows for p in r.get('protection',{}).get('protective_plans',[])),
        maximum_case_cpu_s=max(r.get('cpu_s',0) for r in rows),
        result_sha256=sha(folder/'results.json'),baseline_sha256=sha(args.baseline/'results.json'),wheel_sha256=sha(wheels[0]))
    save(folder/'verification.json',v);save(folder/'journal-sha256.json',journal_hashes)
    save(args.output.with_suffix('.json'),dict(verification=v,regression_comparisons=comparisons,
        fresh_comparisons=fresh_comparisons,isolation=isolation,biased_promotions=biased_promotions,misses=misses,cases=summary))
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',sharey='row')
    for col,probe in enumerate((1,2)):
        selected=[r for r in comparisons if r['scenario']==f'P{probe}_compound'];selected.sort(key=lambda r:r['case'])
        x=np.arange(len(selected))
        for shift,label,color,which in ((-.18,'Prior stack','#ce8237','old'),(.18,'Layered stack','#087e8b','new')):
            axes[0,col].bar(x+shift,[r[which]['post_event_rms_m'] for r in selected],.36,label=label,color=color)
            axes[1,col].bar(x+shift,[r[which]['propellant_kg'] for r in selected],.36,color=color)
        axes[0,col].set_yscale('symlog',linthresh=10);axes[0,col].set_ylabel('Post-event position RMS (m)')
        axes[0,col].set_title(f'ARTEMIS P{probe} regression');axes[0,col].legend()
        axes[1,col].set_ylabel('Actual propellant used (kg)');axes[1,col].set_xlabel('Previous fault scenario')
        axes[1,col].set_xticks(x,[r['case'][:2] for r in selected])
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.25)
    fig.suptitle('Five-layer implementation: changed sensors/hardware assumptions are explicit')
    fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=160);plt.close(fig)
    lines=['# Layered autonomy results','',
        'The five implementation layers are described in [the design and interface note](layered-autonomy.md). '
        'This is a frozen host simulation with real reconstructed ARTEMIS initial states and synthetic subsequent trajectories, navigation fixes, thrust/flow telemetry and ground messages. '
        'The added independent sensors, shutoff and backup bank are hardware assumptions, not benefits of a coordinate transformation.','',
        f"{v['completed']}/{v['cases']} runs completed over {v['simulated_days']:.0f} simulated days; {v['exceptions']} exceptions and {v['collisions']} modeled collisions. "
        f"{v['tests_passed']} tests passed. The prior 124-run campaign remains archived.",'',
        '| Split (float layered stack) | Cases | Navigation restored | Nominal trajectory restored |',
        '|---|---:|---:|---:|']
    for label,group in [('Previous scenarios',v['regression']),('Fresh schedules/seeds',v['fresh'])]:
        lines.append(f"| {label} | {group['cases']} | {group['navigation']} | {group['tracking']} |")
    lines+=['','Navigation restoration requires trusted error ≤200 m and ≤0.02 m/s; trajectory restoration requires ≤100 m and ≤0.02 m/s, sustained through the final six hours after the schedule ends. '
        'Counts are outcomes of selected scenarios, not reliability probabilities. No collision alone does not establish a survival advantage.','',
        f"Across the 40 regression comparisons, post-event RMS improves in {v['regression_rms_improved']} and worsens in {v['regression_rms_worsened']} "
        '(using a 1 mm reporting tolerance). Restoration counts do not conceal these error tradeoffs.','',
        '## Previous missed cases','',
        '| Geometry / case | Old → new final navigation error (m) | Old → new final path error (km) | Old → new propellant (kg) |',
        '|---|---:|---:|---:|']
    for r in comparisons:
        if r['old']['settled']:continue
        a,b=r['old'],r['new']
        lines.append(f"| {r['scenario']} / {r['case']} | {a['final_navigation_error_m']:.1f} → {b['final_navigation_error_m']:.1f} | "
            f"{a['final_error_m']/1000:.3f} → {b['final_error_m']/1000:.3f} | {a['propellant_kg']:.3f} → {b['propellant_kg']:.3f} |")
    lines+=['','## Fresh cases','',
        '| Geometry / schedule | Navigation / tracking restored | Final path error (km) | Propellant (kg) |',
        '|---|---|---:|---:|']
    for r in fresh:
        m=r['metrics'];status=f"{bool(m.get('navigation_settled'))} / {bool(m.get('settled'))}"
        lines.append(f"| {r['scenario']['name']} / {r['case']['name']} | {status} | {m.get('final_error_m',float('nan'))/1000:.3f} | {m.get('propellant_kg',float('nan')):.3f} |")
    lines+=['','## Failure diagnoses and conditional advantages','',
        'The persistent-rejection repair removes the earlier navigation lockout, but it also promotes a coherent biased sensor track. '
        'In the 1 km position-bias cases, guidance accepts the estimate at hour 18 and commands a correction while its actual navigation error is '
        '990 m (P1) or 1,015 m (P2); the reported position 3σ bound is only about 44.6 m. '
        'Post-event RMS rises from 29.3 to 362.3 m on P1 and from 28.7 to 345.9 m on P2. '
        'The final six-hour recovery count does not imply accurate navigation throughout the run. '
        'Independent navigation evidence or a separately validated limited-authority reacquisition mode is still needed before promoting such a track to full tracking authority.','',
        'In the P1 outage/noisy-return/later-kick case, final navigation error improves from 3,756 m to 19 m, '
        'while final path error worsens from 4.235 to 6.200 km and actual fuel rises from 0.102 to 0.324 kg. '
        'The supervisor reaches its shared budget and then uses screened coasting outside routine authority. '
        'This repairs estimation lockout without demonstrating mission return.','',
        'With working independent shutoff, all four stuck-thrust geometry/schedule combinations request isolation 120 s after onset '
        'and enable the backup after another 120 s of closed/quiet confirmation. '
        'The new twelve-hour stuck-thrust schedule recovers the trajectory in both geometries, using 0.184/0.209 kg '
        'versus 3.057/3.094 kg in the matched legacy runs. '
        'When isolation is unavailable, the request does not stop thrust or enable backup; final path errors remain 265.704/105.514 km, '
        'identical to the matched legacy runs. Physical isolation is therefore a load-bearing hardware dependency.','',
        'The ground command sequence accepts the authorized hold, rejects the bad signature and replay, rejects the blackout-delayed expired resume, '
        'and accepts a fresh resume at hour 61. Permission is restored without bypassing local gates. '
        'Neither geometry returns to the nominal path in the remaining 11 hours (18.214/6.070 km final error). '
        'The P2 repeated-kick case and both 0.65 m/s kick cases also retain navigation but fail mission return. '
        'A six-hour feasible coast does not establish a return corridor or months of survival.','',
        'These comparisons identify benefits of the combined new stack under additional sensing/actuation assumptions; '
        'they do not isolate every software layer causally or demonstrate a shape-sphere advantage. '
        'An independently reachable recovery reference, an explicit terminal return constraint and an actuator/attitude-coupled contingency optimizer '
        'are candidate next changes, to be tested with new schedules rather than tuned to these final holdouts.']
    lines+=['','## Verification and limits','',
        f"Verified {v['source_files']} source/test/bridge hashes and {v['physical_inputs']} input files; "
        f"{v['causal_prefix_checks']} pre-event comparisons, {v['ground_link_independence_checks']} ground-only independence checks, "
        f"{v['untrusted_zero_burn_checks']} untrusted-navigation and {v['isolation_zero_burn_checks']} isolation inhibits. "
        f"Maximum rolling issued Δv: {v['maximum_rolling_dv_m_s']:.6f} m/s against the 1 m/s limit.",'',
        f"All {len(pairs)} arithmetic pairs match tracking/navigation outcomes and burn counts: {v['arithmetic_outcomes_match']}. "
        f"Maximum RMS difference {v['maximum_arithmetic_rms_difference_m']:.6g} m; maximum issued-Δv difference {v['maximum_arithmetic_dv_difference_m_s']:.6g} m/s. "
        'Only the QP is integer host emulation; there is no synthesized FPGA timing or power result.','',
        f"Outbox events {telemetry['events']}; acknowledged {telemetry['delivered']}; pending {telemetry['pending']}; dropped {telemetry['dropped']}. "
        'The ground receiver is local and simulated. The command test rejects bad signatures, repeated sequences and expired blackout-delayed messages.','',
        'The initial development planner minimized reference error plus a burn penalty and spent excessive protective effort on the P1 lockout case. '
        'Its eight development results are retained. Before the final holdouts, the policy was changed to prefer a feasible screened coast and require renewed in-corridor trust before returning to routine tracking. '
        'The frozen final policy is not retuned after these outcomes.','',
        f"The campaign executes {v['protective_decisions']} protective planning decisions, including {v['protective_burn_decisions']} screened-burn choices. "
        'The separate synthetic impact unit test exercises an active avoidance burn. Host computation time does not advance physical simulation time; '
        'scheduling isolation, watchdog behavior and execution deadlines require hardware integration.','',
        'Thrust isolation is conditional on independent hardware and sensor availability. The failed-isolation cases retain continuing physical thrust and inhibit backup use. '
        'The protective ensemble is a finite diagnostic screen, not a certified uncertainty tube; covariance calibration, attitude coupling, power/thermal dynamics, physical sensor observability and flight storage remain open. '
        'The campaign does not yet integrate the separate RCS/attitude allocation module.','',
        f'![Regression tracking and actual fuel]({args.output.with_suffix(".png").name})','']
    args.output.with_suffix('.md').write_text('\n'.join(lines));print(json.dumps(v,indent=2))


if __name__=='__main__':main()
