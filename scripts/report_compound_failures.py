#!/usr/bin/env python3
"""Verify frozen compound-fault runs and distinguish recovery from observed safety."""
import argparse
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artemis_resilience_campaign import REPO,sha,read,save
from artemis_compound_campaign import key
from qrc_pulsar_bench.compound_faults import Fault,FaultCase
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,score_recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=read(args.campaign/'results.json');rows=report['results']
    protocol=read(args.campaign/'protocol.json')
    expected={(j['scenario']['name'],j['case']['name'],j['arithmetic'],'coast_after_event' if j['coast'] else 'routine_mpc') for j in protocol['jobs']}
    assert len(rows)==len(expected) and {key(r) for r in rows}==expected
    for name,digest in report['source_sha256'].items():
        assert sha(REPO/name)==digest,(name,'current source changed')
        assert sha(args.campaign/'source'/name)==digest,(name,'snapshot changed')
    for name,digest in report['input_sha256'].items():assert sha(Path(name))==digest
    assert sha(args.campaign/'protocol.json')==report['protocol_sha256']
    lookup={key(row):row for row in rows};summary=[];nis=[];rolling_max=0.;untrusted_checks=0;prefix_checks=0
    journal_hashes={};ground_checks=0;arithmetic=[]
    for row in rows:
        metrics=row['metrics'];raw=row['case']
        case=FaultCase(raw['name'],raw['category'],raw['description'],tuple(Fault(**e) for e in raw['events']))
        if row.get('dense_samples'):
            if row['dense_samples'][-1]['time_s']>=case.onset_s:
                calculated=score_recovery(row['dense_samples'],Anomaly(case.name,case.onset_s,case.end_s),completed=metrics['completed'])
                for name,value in calculated.items():
                    if isinstance(value,float):assert np.isclose(metrics[name],value,rtol=1e-12,atol=1e-12)
                    else:assert metrics[name]==value
            issued=sum(np.linalg.norm(c['dv']) for c in row['commands'] if c['burn_s']>0)
            assert np.isclose(issued,metrics['issued_dv_m_s'],rtol=0,atol=1e-12)
            for sample,command in zip(row['samples'],row['commands']):
                assert np.all(np.isfinite(command['dv']))
                if not sample['navigation_trusted']:
                    assert command['burn_s']==0 and np.linalg.norm(command['dv'])==0;untrusted_checks+=1
                if command['burn_s']>0:assert command['status']==0
                spent=sum(np.linalg.norm(c['dv']) for c in row['commands']
                    if command['time_s']-86400<c['time_s']<=command['time_s'] and c['burn_s']>0)
                assert spent<=1.+1e-10;rolling_max=max(rolling_max,float(spent))
            current_nis=[s['nis'] for s in row['samples'] if s['nis'] is not None]
            assert all(np.isfinite(n) and n>=0 for n in current_nis);nis.extend(current_nis)
            tail=[s for s in row['samples'] if s['time_s']>=case.end_s]
            last_bad=max((i for i,s in enumerate(tail) if not s['navigation_trusted'] or s['navigation_error_m']>200 or s['navigation_velocity_error_m_s']>.02),default=-1)
            tail=tail[last_bad+1:]
            assert metrics['navigation_settled']==bool(metrics['completed'] and tail and tail[-1]['time_s']-tail[0]['time_s']>=21600)
            clean=lookup[(row['scenario']['name'],'none',row['arithmetic'],'routine_mpc')]
            if case.events:
                for a,b in zip(row['samples'],clean['samples']):
                    if a['time_s']<case.onset_s:assert a==b;prefix_checks+=1
                observable_dropout=any(e.kind=='sensor_dropout' and e.active(case.onset_s) for e in case.events)
                for a,b in zip(row['commands'],clean['commands']):
                    if a['time_s']<case.onset_s or (a['time_s']==case.onset_s and not observable_dropout and row['control']!='coast_after_event'):
                        assert a==b;prefix_checks+=1
            if case.name=='01_ground_blackout' and row['control']=='routine_mpc':
                assert row['dense_samples']==clean['dense_samples'] and row['commands']==clean['commands'];ground_checks+=1
            telemetry=row['telemetry'];path=Path(telemetry['journal_path']);journal_hashes[str(path)]=sha(path)
            events={};acked=set();pending=set();maximum=0
            for line in path.read_text().splitlines():
                record=json.loads(line);seq=record['sequence']
                if record['record']=='event':
                    assert seq==len(events)+1;events[seq]=record;pending.add(seq);maximum=max(maximum,len(pending))
                else:
                    assert seq in pending;pending.remove(seq);acked.add(seq)
            assert len(events)==telemetry['events'] and len(pending)==telemetry['pending'] and maximum==telemetry['maximum_pending']
            if not telemetry['errors']:
                assert len(acked)==telemetry['delivered'] and telemetry['dropped_events']==0
                latencies=[]
                for seq in acked:
                    available=next(s['time_s'] for s in row['samples'] if s['time_s']>=events[seq]['time_s'] and s['ground_link_up'])
                    latencies.append(available-events[seq]['time_s'])
                assert max(latencies,default=0)==telemetry['maximum_delivery_latency_s']
        observed_safety=bool(metrics['completed'] and not metrics.get('collision',False)
            and metrics.get('min_clearance_m',0)>=50000 and metrics.get('remaining_propellant_kg',0)>=.5)
        summary.append(dict(key=key(row),category=case.category,description=case.description,**metrics,
            observed_safety=observed_safety,failure=row.get('failure'),telemetry=row.get('telemetry'),
            navigation_events=row.get('navigation_events',[])))
        if row['arithmetic']=='32':
            other=lookup[(row['scenario']['name'],case.name,'float64',row['control'])]['metrics']
            executed=bool(metrics['completed'] and other['completed'])
            arithmetic.append(dict(scenario=row['scenario']['name'],case=case.name,executed=executed,
                same_tracking=metrics['response']==other['response'],same_navigation=metrics.get('navigation_settled')==other.get('navigation_settled'),
                same_burns=metrics.get('burns')==other.get('burns'),
                rms_difference_m=abs(metrics['post_event_rms_m']-other['post_event_rms_m']) if executed else None,
                dv_difference_m_s=abs(metrics['issued_dv_m_s']-other['issued_dv_m_s']) if executed else None))
    scored=[r for r in summary if r['key'][1]!='none' and r['key'][2]=='float64' and r['key'][3]=='routine_mpc']
    coast=[r for r in summary if r['key'][3]=='coast_after_event']
    categories={}
    for category in ('single','concurrent','consecutive'):
        selected=[r for r in scored if r['category']==category]
        categories[category]=dict(instances=len(selected),navigation_restored=sum(bool(r.get('navigation_settled')) for r in selected),
            trajectory_restored=sum(bool(r.get('settled')) for r in selected),observed_safety=sum(r['observed_safety'] for r in selected),
            modeled_collisions=sum(bool(r.get('collision')) for r in selected),exceptions=sum(r['failure'] is not None for r in selected))
    test_log=(args.campaign/'pytest.log').read_text();match=re.search(r'(\d+) passed,',test_log)
    assert match and 'FAILED ' not in test_log
    wheels=list((args.campaign/'wheels').glob('*.whl'));assert len(wheels)==1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in report['source_sha256']:
            if name.startswith('qrc_pulsar_bench/'):assert wheel.read(name)==(REPO/name).read_bytes()
    pairs=[p for p in arithmetic if p['executed']]
    coast_lookup={r['key'][:2]:r for r in coast}
    comparisons=[]
    for row in scored:
        counterfactual=coast_lookup[row['key'][:2]]
        if row['completed'] and counterfactual['completed']:
            baseline=counterfactual['post_event_rms_m']
            comparisons.append(dict(scenario=row['key'][0],case=row['key'][1],
                rms_reduction_percent=100*(1-row['post_event_rms_m']/baseline) if baseline else None,
                extra_propellant_kg=row['propellant_kg']-counterfactual['propellant_kg'],
                extra_issued_dv_m_s=row['issued_dv_m_s']-counterfactual['issued_dv_m_s']))
    verification=dict(cases=len(rows),completed=sum(bool(r['metrics']['completed']) for r in rows),
        exceptions=sum('failure' in r for r in rows),modeled_collisions=sum(bool(r['metrics'].get('collision')) for r in rows),
        requested_days=sum(r['scenario']['days'] for r in rows),
        simulated_days=sum(r.get('dense_samples',[{'time_s':0}])[-1]['time_s']/86400 for r in rows),
        tests_passed=int(match.group(1)),sources=len(report['source_sha256']),physical_inputs=len(report['input_sha256']),
        nonnegative_nis_samples=len(nis),untrusted_zero_command_checks=untrusted_checks,causal_prefix_checks=prefix_checks,
        ground_link_independence_checks=ground_checks,maximum_rolling_issued_dv_m_s=rolling_max,
        float_instances=len(scored),float_navigation_restored=sum(bool(r.get('navigation_settled')) for r in scored),
        float_tracking_restored=sum(bool(r.get('settled')) for r in scored),float_observed_safety=sum(r['observed_safety'] for r in scored),
        coast_tracking_restored=sum(bool(r.get('settled')) for r in coast),coast_observed_safety=sum(r['observed_safety'] for r in coast),
        categories=categories,arithmetic=arithmetic,executed_arithmetic_pairs=len(pairs),
        comparisons=comparisons,
        improved_rms_instances=sum(p['rms_reduction_percent']>1e-6 for p in comparisons),
        worse_rms_instances=sum(p['rms_reduction_percent'] < -1e-6 for p in comparisons),
        median_rms_reduction_percent=float(np.median([p['rms_reduction_percent'] for p in comparisons])),
        float_minimum_recorded_clearance_m=min(r['min_clearance_m'] for r in scored if r.get('min_clearance_m') is not None),
        float_maximum_propellant_kg=max(r.get('propellant_kg',0) for r in scored),
        telemetry_events=sum(r.get('telemetry',{}).get('events',0) for r in rows),
        telemetry_delivered=sum(r.get('telemetry',{}).get('delivered',0) for r in rows),
        telemetry_pending=sum(r.get('telemetry',{}).get('pending',0) for r in rows),
        telemetry_dropped=sum(r.get('telemetry',{}).get('dropped_events',0) for r in rows),
        telemetry_maximum_pending=max(r.get('telemetry',{}).get('maximum_pending',0) for r in rows),
        telemetry_maximum_delivery_latency_h=max(r.get('telemetry',{}).get('maximum_delivery_latency_s',0) for r in rows)/3600,
        arithmetic_outcomes_match=all(p['same_tracking'] and p['same_navigation'] and p['same_burns'] for p in pairs),
        maximum_arithmetic_rms_difference_m=max((p['rms_difference_m'] for p in pairs),default=None),
        maximum_arithmetic_dv_difference_m_s=max((p['dv_difference_m_s'] for p in pairs),default=None),
        results_sha256=sha(args.campaign/'results.json'),wheel_sha256=sha(wheels[0]),hardware_synthesis=False)
    save(args.campaign/'verification.json',verification);save(args.campaign/'journal-sha256.json',journal_hashes)
    save(args.output.with_suffix('.json'),dict(verification=verification,cases=summary))
    table_lookup={tuple(r['key']):r for r in summary}
    names=sorted({r['key'][1] for r in scored})
    fig,axes=plt.subplots(3,2,figsize=(15,12),sharex='col',sharey='row')
    for col,probe in enumerate((1,2)):
        m=[table_lookup[(f'P{probe}_compound',name,'float64','routine_mpc')] for name in names]
        c=[table_lookup[(f'P{probe}_compound',name,'float64','coast_after_event')] for name in names]
        x=np.arange(len(names))
        axes[0,col].bar(x-.18,[r.get('post_event_rms_m') or 0 for r in m],.36,label='MPC')
        axes[0,col].bar(x+.18,[r.get('post_event_rms_m') or 0 for r in c],.36,label='Coast after first event')
        axes[0,col].set_yscale('symlog',linthresh=10);axes[0,col].set_title(f'ARTEMIS P{probe} geometry')
        axes[0,col].set_ylabel('Post-event position RMS (m)');axes[0,col].legend(fontsize=9)
        axes[1,col].bar(x,[r.get('issued_dv_m_s',0) for r in m],color='#007c91')
        axes[1,col].set_ylabel('Total issued Δv (m/s)');axes[1,col].set_ylim(bottom=0)
        axes[2,col].bar(x-.18,[r.get('propellant_kg',0) for r in m],.36,label='MPC')
        axes[2,col].bar(x+.18,[r.get('propellant_kg',0) for r in c],.36,label='Coast after first event')
        axes[2,col].set_ylabel('Actual propellant used (kg)\nIncludes uncommanded thrust')
        axes[2,col].set_xlabel('Fault scenario number')
        axes[2,col].set_xticks(x,[n[:2] for n in names]);axes[2,col].set_ylim(bottom=0)
        for axis in axes[:,col]:axis.grid(axis='y',alpha=.25)
    fig.suptitle('Twenty frozen fault schedules: tracking benefit and correction cost')
    fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=160);plt.close(fig)
    lines=['# Concurrent and consecutive fault campaign','',
        'Twenty new synthetic fault schedules: 10 individual faults, 5 simultaneous combinations, 5 ordered sequences. '
        'Two reconstructed ARTEMIS initial geometries, three days each, new noise seeds 14001/14002, float64 and 32/24 MPC, '
        'matched coasting comparisons and clean controls. Subsequent spacecraft motion and fixes are simulated; this is not a replay of observed spacecraft fault responses.','',
        f"Declared {len(rows)} cases; {verification['completed']} completed; {verification['exceptions']} exceptions; "
        f"{verification['modeled_collisions']} modeled collisions. {verification['tests_passed']} tests passed.",'',
        '## Handling policy','',
        'The current tested policy quarantines malformed/replayed/future sensor packets, retains the last valid square-root estimate, '
        'requires sustained evidence for provisional or post-outage reacquisition, inhibits routine thrust while navigation is untrusted, '
        'and retains the existing error envelope, clearance screen and daily issued-Δv limit. It does not automatically rearm the recovery latch. '
        'Numerical filter failures are surfaced separately from malformed inputs. Runtime telemetry write failures are contained so guidance can continue.',
        '', 'A persistent local event journal queues status changes, reacquisitions and issued commands. A simulated receiver deduplicates and acknowledges events when the link returns. '
        'Ground-link outage does not remove onboard navigation fixes. No external notifications are sent. This is not a real radio, ground-command or flight-storage implementation.',
        '', 'For this architecture, the supported next design is layered fault detection/isolation/recovery, with an independent protective response, '
        'resource-aware recovery planning and ground review/intervention. NASA’s '
        '[fault-detection guidance](https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695798/9.07%2BFault%2BDetection%2Band%2BResponse) '
        'emphasizes autonomous protection that preserves resources and commandability, with timely autonomous action for critical activities. '
        'These tests do not select a globally optimal controller or validate a safe emergency maneuver planner.','',
        '## Results by category (float MPC only)','',
        '| Category | Instances | Navigation restored | Nominal trajectory restored | Observed safety checks passed |',
        '|---|---:|---:|---:|---:|']
    for category,counts in categories.items():lines.append(f"| {category} | {counts['instances']} | {counts['navigation_restored']} | {counts['trajectory_restored']} | {counts['observed_safety']} |")
    lines += ['', 'Navigation restoration requires trusted position error ≤200 m and velocity error ≤0.02 m/s for the final 6 h after the schedule ends. '
        'Trajectory restoration uses ≤100 m and ≤0.02 m/s over that interval. Observed safety means full completion, no modeled collision, '
        'recorded lunar clearance ≥50 km and remaining propellant ≥0.5 kg. It does not establish future survivability, power/thermal/attitude safety or mission success.',
        '',f"Matched coasting restores the nominal corridor in {verification['coast_tracking_restored']}/{len(coast)} cases and passes the observed safety checks in "
        f"{verification['coast_observed_safety']}/{len(coast)}. Counts describe these deliberately chosen scenarios, not flight reliability probabilities.",'',
        f"MPC lowers post-event position RMS in {verification['improved_rms_instances']}/{len(comparisons)} matched comparisons, "
        f"raises it in {verification['worse_rms_instances']}, and has a median reduction of {verification['median_rms_reduction_percent']:.1f}%. "
        'The comparison is against stopping commanded corrections at the first event; it does not establish fuel superiority over another capable controller. '
        'Actual propellant, including uncommanded thrust, is recorded separately from issued Δv in the JSON results.', '',
        '## All twenty schedules','',
        '| Scenario | Description | P1 navigation / tracking | P2 navigation / tracking | MPC Δv P1 / P2 (m/s) |',
        '|---|---|---|---|---:|']
    for name in names:
        a=table_lookup[('P1_compound',name,'float64','routine_mpc')];b=table_lookup[('P2_compound',name,'float64','routine_mpc')]
        status=lambda r:('yes' if r.get('navigation_settled') else 'no')+' / '+('yes' if r.get('settled') else 'no')
        lines.append(f"| {name} | {a['description']} | {status(a)} | {status(b)} | {a.get('issued_dv_m_s',0):.3f} / {b.get('issued_dv_m_s',0):.3f} |")
    lines += ['', '## Unrecovered cases', '',
        '| Geometry / scenario | Final navigation error (m) | Final trajectory error (km) | Post-event RMS MPC / coast (km) |',
        '|---|---:|---:|---:|']
    for row in scored:
        if row.get('settled'):continue
        counterfactual=coast_lookup[row['key'][:2]]
        if row['completed'] and counterfactual['completed']:
            lines.append(f"| {row['key'][0]} / {row['key'][1]} | {row['final_navigation_error_m']:.1f} | "
                f"{row['final_error_m']/1000:.3f} | {row['post_event_rms_m']/1000:.3f} / {counterfactual['post_event_rms_m']/1000:.3f} |")
        else:lines.append(f"| {row['key'][0]} / {row['key'][1]} | Incomplete: see retained failure | — | — |")
    lines += ['', f"All campaign journals: {verification['telemetry_events']} events, {verification['telemetry_delivered']} delivered to the simulated receiver, "
        f"{verification['telemetry_pending']} pending and {verification['telemetry_dropped']} dropped at run end. "
        f"Maximum backlog {verification['telemetry_maximum_pending']} events; longest delivery delay {verification['telemetry_maximum_delivery_latency_h']:.1f} h. "
        'Separate tests exercise restart, a lost acknowledgment, corruption, capacity overflow and storage failure.']
    lines += ['', 'The stuck-thrust schedule ends by construction after six hours; the controller has no independent valve-shutoff mechanism. '
        'Its small issued-Δv total must not be mistaken for small fuel use: uncommanded propulsion drives actual consumption to about 2.1 kg. '
        'Both geometries regain navigation but exceed the routine tracking envelope. P1 schedule 17 also latches recovery-required status; '
        'P1 schedule 18 remains navigation-untrusted after noisy measurements and a later kick. '
        'The [failure-handling design note](compound-fault-handling.md) explains the resulting implementation priorities.']
    lines += ['', '## Verification and limitations','',
        f"Verified {verification['sources']} source/test/runner hashes, {verification['physical_inputs']} physical inputs, the built wheel, "
        f"{verification['nonnegative_nis_samples']} nonnegative innovation statistics, {verification['untrusted_zero_command_checks']} zero-thrust checks while navigation is untrusted, "
        f"{verification['causal_prefix_checks']} exact causal-prefix comparisons and all saved telemetry journals. "
        'Uncommanded propulsion is integrated with both its force and mass flow; overlapping jets combine vector forces and summed flows at exact burn boundaries.',
        '',f"{len(pairs)} complete arithmetic pairs; all tracking/navigation classifications and burn counts match: {verification['arithmetic_outcomes_match']}. "
        'Only the QP executes in integer host emulation; navigation, physics and supervision remain float64. No FPGA synthesis or execution-time/power guarantee.',
        '', 'The earlier navigation campaign exposed empirical covariance undercoverage; positive covariance does not certify a safety margin. '
        'Coherent bad sensor data can still mimic real motion. The routine envelope/recovery latch conserves authority but provides no validated survival trajectory. '
        'Three-day runs, synthetic direct-state fixes, fixed thrust-direction error and abstract force disturbances leave substantial flight validation work.',
        '',f'![Tracking and correction comparison]({args.output.with_suffix(".png").name})','']
    args.output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
