#!/usr/bin/env python3
"""Verify and report frozen navigation recovery experiments."""
import argparse
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from artemis_resilience_campaign import sha,read,save,REPO
from artemis_navigation_campaign import key
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,score_recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--diagnosis',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=read(args.campaign/'results.json');rows=report['results']
    protocol=read(args.campaign/'protocol.json')
    expected={(j['scenario']['name'],j['anomaly']['name'],j['arithmetic'],j['navigation']) for j in protocol['jobs']}
    assert {key(row) for row in rows}==expected and len(rows)==len(expected)
    for name,digest in report['source_sha256'].items():
        assert sha(REPO/name)==digest,(name,'current source changed')
        assert sha(args.campaign/'source'/name)==digest,(name,'archive changed')
    for name,digest in report['input_sha256'].items():assert sha(Path(name))==digest
    assert sha(args.campaign/'protocol.json')==report['protocol_sha256']
    summary=[];all_nis=[];paired=[];pre_event_checks=0;onset_checks=0
    rolling_max=0.;untrusted_command_checks=0
    lookup={key(row):row for row in rows}
    for row in rows:
        metrics=row['metrics'];samples=row.get('samples',[])
        onset=row['anomaly']['onset_s'];events=row.get('navigation_events',[])
        if row.get('dense_samples'):
            scored=score_recovery(row['dense_samples'],Anomaly(**row['anomaly']),completed=metrics['completed'])
            for field,value in scored.items():
                if isinstance(value,float):assert np.isclose(metrics[field],value,rtol=1e-12,atol=1e-12)
                else:assert metrics[field]==value
            for command in row['commands']:
                if command['burn_s']>0:assert command['status']==0
                used=sum(np.linalg.norm(c['dv']) for c in row['commands']
                    if command['time_s']-86400<c['time_s']<=command['time_s'] and c['burn_s']>0)
                rolling_max=max(rolling_max,float(used))
                assert used<=1.+1e-10
            for sample,command in zip(samples,row['commands']):
                if not sample.get('navigation_trusted',True):
                    assert command['burn_s']==0 and np.linalg.norm(command['dv'])==0
                    untrusted_command_checks+=1
            issued=sum(np.linalg.norm(c['dv']) for c in row['commands'] if c['burn_s']>0)
            assert np.isclose(issued,metrics['issued_dv_m_s'],atol=1e-12,rtol=0)
        after=[s for s in samples if s['time_s']>=onset]
        recovered=next((s for s in after if s.get('reacquired')),None)
        trusted=[s for s in after if s.get('navigation_trusted',False)]
        post_confirmation=[s for s in trusted if recovered and s['time_s']>=recovered['time_s']]
        ni=[s['nis'] for s in samples if s.get('nis') is not None]
        assert all(np.isfinite(v) and v>=0 for v in ni)
        all_nis.extend(ni)
        summary.append(dict(key=key(row),split=row['scenario']['split'],**metrics,
            failure=row.get('failure',{}).get('message'),
            first_reacquisition_after_onset_h=None if recovered is None else (recovered['time_s']-onset)/3600,
            first_reacquisition_navigation_error_m=None if recovered is None else recovered['navigation_error_m'],
            first_reacquisition_velocity_error_m_s=None if recovered is None else recovered['navigation_velocity_error_m_s'],
            final_navigation_error_m=samples[-1]['navigation_error_m'] if samples else None,
            final_navigation_velocity_error_m_s=samples[-1].get('navigation_velocity_error_m_s') if samples else None,
            post_confirmation_navigation_rms_m=float(np.sqrt(np.mean([s['navigation_error_m']**2 for s in post_confirmation]))) if post_confirmation else None,
            post_confirmation_position_bound_coverage=float(np.mean([s['navigation_error_m']<=s['position_3sigma_m'] for s in post_confirmation])) if post_confirmation else None,
            post_confirmation_velocity_bound_coverage=float(np.mean([s['navigation_velocity_error_m_s']<=s['velocity_3sigma_m_s'] for s in post_confirmation])) if post_confirmation else None,
            events=events))
        if row['anomaly']['name']!='none' and row['navigation']=='maneuver_aware' and samples:
            clean=lookup[(row['scenario']['name'],'none',row['arithmetic'],row['navigation'])]
            for actual,baseline in zip(samples,clean.get('samples',[])):
                if actual['time_s']<onset:
                    assert actual==baseline;pre_event_checks+=1
            for actual,baseline in zip(row['commands'],clean['commands']):
                # Missing delivery is itself causal information at onset.
                # Otherwise the delayed onset fix still precedes the fault.
                if actual['time_s']<onset or (actual['time_s']==onset and row['anomaly']['outage_s']==0):
                    assert actual==baseline;onset_checks+=1
        if row['arithmetic']=='32':
            baseline=lookup[(row['scenario']['name'],row['anomaly']['name'],'float64',row['navigation'])]
            paired.append(dict(scenario=row['scenario']['name'],anomaly=row['anomaly']['name'],
                executed_pair=bool(metrics['completed'] and baseline['metrics']['completed']),
                same_response=metrics['response']==baseline['metrics']['response'],
                same_reacquisitions=metrics.get('navigation_reacquisitions')==baseline['metrics'].get('navigation_reacquisitions'),
                same_burns=metrics.get('burns')==baseline['metrics'].get('burns'),
                post_event_rms_difference_m=(abs(metrics['post_event_rms_m']-baseline['metrics']['post_event_rms_m'])
                    if metrics['completed'] and baseline['metrics']['completed'] else None),
                issued_dv_difference_m_s=abs(metrics.get('issued_dv_m_s',0)-baseline['metrics'].get('issued_dv_m_s',0))))
    failure=np.load(args.diagnosis/'failed_update.npz');p=failure['p'];r=failure['r']
    innovation=failure['z']-failure['predicted'][:6];s=p[:6,:6]+r
    diagnostic=dict(original_nis=float(innovation@np.linalg.solve(s,innovation)),
        original_innovation_condition=float(np.linalg.cond(s)),
        measurement_variance_added=(np.diag(s)-np.diag(p[:6,:6])).tolist(),
        maximum_prior_position_variance_m2=float(max(np.diag(p)[:3])),
        original_minimum_posterior_variance=float(min(np.diag(failure['posterior']))),
        bad_process_noise_steps=len(read(args.diagnosis/'diagnosis.json')['bad_process_steps']))
    verification=dict(cases=len(rows),completed=sum(bool(r['metrics']['completed']) for r in rows),
        failures=sum('failure' in r for r in rows),requested_days=sum(r['scenario']['days'] for r in rows),
        reference_preflight_rejections=sum(r['metrics']['response']=='not_run' for r in rows),
        numerical_terminations=sum('failure' in r and r['metrics']['response']!='not_run' for r in rows),
        completed_without_collision=sum(bool(r['metrics']['completed']) and not r['metrics'].get('collision',False) for r in rows),
        minimum_clearance_m=min(r['metrics'].get('min_clearance_m',float('inf')) for r in rows),
        maximum_propellant_kg=max(r['metrics'].get('propellant_kg',0) for r in rows),
        source_files=len(report['source_sha256']),physical_inputs=len(report['input_sha256']),
        nonnegative_nis_samples=len(all_nis),minimum_nis=min(all_nis),
        maximum_rolling_issued_dv_m_s=rolling_max,untrusted_zero_command_checks=untrusted_command_checks,
        pre_event_sample_pairs=pre_event_checks,identical_causal_prefix_commands=onset_checks,
        arithmetic_pairs=paired,results_sha256=sha(args.campaign/'results.json'),
        diagnostic=diagnostic,hardware_synthesis=False)
    test_log=(args.campaign.parent/'pytest-final.log').read_text()
    match=re.search(r'(\d+) passed,',test_log)
    assert match and 'FAILED ' not in test_log
    verification['tests_passed']=int(match.group(1))
    wheels=list((args.campaign.parent/'wheels').glob('*.whl'));assert len(wheels)==1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in report['source_sha256']:
            if name.startswith('qrc_pulsar_bench/'):
                assert wheel.read(name)==(REPO/name).read_bytes(),(name,'wheel changed')
    verification['wheel_sha256']=sha(wheels[0])
    executed_pairs=[p for p in paired if p['executed_pair']]
    verification['executed_arithmetic_pairs']=len(executed_pairs)
    verification['matching_arithmetic_outcomes']=all(p['same_response'] and p['same_reacquisitions'] and p['same_burns'] for p in executed_pairs)
    verification['maximum_arithmetic_rms_difference_m']=max(p['post_event_rms_difference_m'] for p in executed_pairs)
    verification['maximum_arithmetic_issued_dv_difference_m_s']=max(p['issued_dv_difference_m_s'] for p in executed_pairs)
    coverage=[r for r in summary if r['key'][2]=='float64' and r['post_confirmation_position_bound_coverage'] is not None]
    verification['post_confirmation_position_bound_coverage_range']=[
        min(r['post_confirmation_position_bound_coverage'] for r in coverage),
        max(r['post_confirmation_position_bound_coverage'] for r in coverage)]
    save(args.campaign/'verification.json',verification)
    output=dict(verification=verification,cases=summary)
    save(args.output.with_suffix('.json'),output)
    previous=read(args.previous/'results.json')['results']+read(args.previous/'continuation.json')['results']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
    for col,probe in enumerate((1,2)):
        new=lookup[(f'P{probe}_regression','kick_50cm_s','float64','maneuver_aware')]
        old=next(r for r in previous if r['scenario']['probe']==probe and r['anomaly']['name']=='kick_50cm_s'
                 and r['arithmetic']=='float64' and r['control']=='routine_mpc')
        onset=new['anomaly']['onset_s']
        for row,label,color in [(old,'Previous filter','#c05621'),(new,'Square-root + reacquisition','#007c91')]:
            for axis,field,collection in [(axes[0,col],'navigation_error_m','samples'),(axes[1,col],'position_error_m','dense_samples')]:
                points=[s for s in row[collection] if s['time_s']>=onset]
                axis.semilogy([(s['time_s']-onset)/3600 for s in points],[max(s[field],.01) for s in points],label=label,color=color)
                if row.get('failure'):
                    axis.scatter([(points[-1]['time_s']-onset)/3600],[points[-1][field]],marker='x',s=65,color='#b22222',zorder=5)
                axis.grid(alpha=.25)
        for event in new['navigation_events']:
            for axis in axes[:,col]:axis.axvline((event['available_s']-onset)/3600,color='#444',ls=':',alpha=.7)
        axes[0,col].set_title(f'ARTEMIS P{probe}: injected 0.5 m/s jump')
        axes[1,col].set_xlabel('Hours after injection')
        axes[0,col].legend(fontsize=9)
    axes[0,0].set_ylabel('Navigation position error (m)')
    axes[1,0].set_ylabel('Trajectory position error (m)')
    fig.suptitle('Navigation recovery and trajectory recovery are separate outcomes')
    fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=160);plt.close(fig)
    bounds_path=args.output.with_name(args.output.name+'-uncertainty').with_suffix('.png')
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for axis,(scenario,event) in zip(axes.flat,[('P1_regression','kick_50cm_s'),('P2_fresh','kick_0.2m_s'),
                                               ('P1_gap_holdout','kick_with_6h_outage'),('P2_gap_holdout','kick_with_6h_outage')]):
        row=lookup[(scenario,event,'float64','maneuver_aware')];onset=row['anomaly']['onset_s']
        points=[s for s in row.get('samples',[]) if s['time_s']>=onset]
        axis.plot([(s['time_s']-onset)/3600 for s in points],
            [s['navigation_error_m']/s['position_3sigma_m'] if s['navigation_trusted'] else np.nan for s in points])
        axis.axhline(1,color='#b22222',ls='--');axis.grid(alpha=.25)
        axis.set_title(f'{scenario}: {event}');axis.set_xlabel('Hours after injection')
        axis.set_ylabel('Position error / nominal 3σ radius')
    fig.suptitle('Numerical validity does not establish calibrated uncertainty; gaps indicate untrusted navigation')
    fig.tight_layout();fig.savefig(bounds_path,dpi=160);plt.close(fig)
    lines=['# Navigation recovery and covariance repair','',
        f'The campaign freezes the implementation and {len(rows)} scenarios/arithmetic combinations before execution. '
        'Each trajectory starts from one reconstructed ARTEMIS state; subsequent spacecraft dynamics and sensor fixes are simulated.',
        '', 'V3 retains V2’s original regression/fresh split names for provenance; those cases are now reused development cases. '
        'Only the additional gap_holdout cases use new noise/direction seeds unseen before the cold-reacquisition repair. '
        'Their initial geometries are reused eligible references. V2 evidence and its 100.5-hour outage-recovery miss are retained separately.',
        '',f"Completed {verification['completed']}/{len(rows)} declared cases. Reference preflight rejections: {verification['reference_preflight_rejections']}; numerical terminations: {verification['numerical_terminations']}.",'',
        'The user’s objective is ground-supervised autonomy: ground may authorize the mission to run autonomously and observe; '
        'the spacecraft should react to anomalies, adjust within authority, and report to ground for review/intervention. '
        'It must also preserve survival during extended loss of ground contact. '
        'Navigation restoration, observed clearance/resource preservation and return to the original trajectory are separate criteria. '
        'Synthetic navigation fixes are assumed available without ground except in declared sensor outages. '
        'This is a short-duration simulated navigation/control test, not a demonstrated onboard sensor system or long-blackout survival guarantee.',
        '', 'The tested measurement cadence is 30 minutes. Three-fix confirmation normally takes 90 minutes after injection; '
        'this does not validate a fast emergency protection loop. The current bounded in-memory navigation log is not a persistent '
        'ground notification/acknowledgment system.',
        '',f"Across executed cases: {verification['completed_without_collision']} completed without a modeled collision; "
        f"minimum recorded lunar clearance {verification['minimum_clearance_m']/1000:.3f} km; "
        f"maximum propellant use {verification['maximum_propellant_kg']:.4f} kg from an initial 20 kg propellant load. "
        'Power, thermal, attitude, communications reacquisition and future safe-return feasibility are not scored.', '',
        '## Reproduced covariance failure','',
        f"The archived P2 replay produces NIS {diagnostic['original_nis']:.6f}, despite NIS being nonnegative by definition. "
        f"Prior position variance reaches {diagnostic['maximum_prior_position_variance_m2']:.6g} m²; "
        f"the innovation covariance condition number is {diagnostic['original_innovation_condition']:.6g}. "
        'Five of six measurement variances round away when added to the prior. The negative NIS passes the old upper-bound gate; '
        'the subsequent Joseph update produces a negative velocity variance. No invalid process-noise matrix was detected.',
        '', 'The repair stores covariance factors, propagates them with QR, and conditions a joint state/measurement factor. '
        'NIS is a squared norm after triangular whitening. The live factor is never reconstructed from a rounded full covariance. '
        'All updates validate before committing state, factors, clocks or counters. '
        'This follows standard square-root filtering principles; see [JPL’s square-root derivation](https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A-orig.pdf).',
        '', '## Reacquisition and its limits','',
        'A significant rejected velocity discrepancy seeds one separate provisional track. At least three distinct fixes over an hour '
        'must fit its propagated motion, including known finite burns, and pass uncertainty limits. '
        'The provisional track replaces the prior only after confirmation; correlated estimates are not fused as independent. '
        'MPC is inhibited while navigation is suspect. Existing 5 km / 0.5 m/s authority limits and 1 m/s daily issued budget remain.',
        '', 'A separate cold-reacquisition path starts after a real gap of more than three hours in received fixes. '
        'It does not require agreement with the stale prior, but still requires three dynamically consistent fixes. '
        'Repeated rejected fixes cannot themselves activate this path. Its event reason is observation_gap, not proof of an impulse.',
        '', 'A dynamically coherent sensor fault is observationally indistinguishable from a real jump in this single direct-state stream. '
        'Passing these tests establishes conditional consistency, not authenticated physical origin. '
        'Independent sensors, realistic tracking observables and covariance calibration remain necessary.',
        '', '## Float results','',
        '| Geometry / split | Fault | Navigation recoveries | First confirmation after onset (h) | Final navigation error (m) | Trajectory response | Final trajectory error (m) | Issued Δv (m/s) | Min lunar clearance (km) |',
        '|---|---|---:|---:|---:|---|---:|---:|---:|']
    for r in summary:
        if r['key'][2]!='float64':continue
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        clearance=r.get('min_clearance_m')
        lines.append(f"| {r['key'][0]} / {r['key'][3]} | {r['key'][1]} | {r.get('navigation_reacquisitions',0)} | {fmt(r['first_reacquisition_after_onset_h'])} | {fmt(r['final_navigation_error_m'])} | {r['response']} | {fmt(r.get('final_error_m'))} | {fmt(r.get('issued_dv_m_s'))} | {fmt(None if clearance is None else clearance/1000)} |")
    lines += ['', '## Verification and scope','',
        f"Passed {verification['tests_passed']} tests and verified the built wheel, {verification['source_files']} source/test/runner hashes, {verification['physical_inputs']} input hashes, "
        f"{verification['nonnegative_nis_samples']} nonnegative NIS values, exact pre-event sample pairing and identical commands before fault information is observable. "
        'Metrics are independently recomputed from saved trajectories and commands. All failed cases remain in the report.',
        '',f"Float64 versus 32/24 QP: {len(executed_pairs)} executed pairs; matching trajectory classifications, navigation-reacquisition counts "
        f"and burn counts: {verification['matching_arithmetic_outcomes']}. "
        f"Largest post-event RMS difference {verification['maximum_arithmetic_rms_difference_m']:.6g} m; "
        f"largest issued-Δv difference {verification['maximum_arithmetic_issued_dv_difference_m_s']:.6g} m/s. "
        'Rejected reference cases are excluded from these arithmetic comparisons.',
        '', 'Integer comparisons run only the 32/24 QP core on the host. Navigation QR, force propagation and supervision remain host float64. '
        'No FPGA synthesis, timing, power, radiation or flight qualification is claimed. '
        'Positive covariance is necessary for a valid estimate, but does not prove a nonlinear EKF is statistically calibrated.',
        '', f"In the confirmed float tracks, the empirical fraction of position errors within the reported largest-axis three-sigma radius ranges "
        f"from {100*verification['post_confirmation_position_bound_coverage_range'][0]:.1f}% "
        f"to {100*verification['post_confirmation_position_bound_coverage_range'][1]:.1f}%. "
        'These time samples are correlated; this is not a flight confidence bound. The misses require further covariance-model calibration '
        'before using this uncertainty as a certified collision-avoidance margin.',
        '', 'Plot: red crosses mark the last saved points of the archived numerical failure; dotted lines mark new navigation confirmation.',
        '',f'![Navigation and trajectory comparison]({args.output.with_suffix(".png").name})',
        '',f'![Empirical uncertainty check]({bounds_path.name})','']
    args.output.with_suffix('.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
