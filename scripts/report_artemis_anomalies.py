#!/usr/bin/env python3
"""Verify all frozen fault cases and report automatic recovery and failures."""
import argparse
import hashlib
from dataclasses import asdict
import json
import zipfile
from pathlib import Path
import numpy as np
from artemis_resilience_campaign import REPO,read,save,sha
from artemis_anomaly_campaign import jobs,key
from qrc_pulsar_bench.anomaly_benchmark import Anomaly,score_recovery


def strict_success(row):
    m=row['metrics'];counts=m['status_counts']
    return bool(m['settled'] and m['min_clearance_m']>=50000 and not any(counts.get(str(k),0) for k in (1,4,16,32,64,128,512)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    report=read(root/'results.json');rows=list(report['results']);source=read(root/'source-sha256.json');inputs=read(root/'input-sha256.json')
    assert report['source_sha256']==source and report['input_sha256']==inputs
    continuation=read(root/'continuation.json') if (root/'continuation.json').exists() else None
    if continuation:
        assert continuation['source_sha256']==source and continuation['input_sha256']==inputs
        assert continuation['protocol_sha256']==sha(root/'protocol.json')
        assert continuation['base_results_sha256']==sha(root/'results.json')
        assert continuation['runner_sha256']==sha(root/'source/scripts/artemis_anomaly_continuation.py')
        rows+=continuation['results']
    rows.sort(key=key)
    for name,digest in source.items():assert sha(REPO/name)==sha(root/'source'/name)==digest,name
    for name,digest in inputs.items():assert sha(Path(name))==digest,name
    test_sources=read(root/'test-source-sha256.json') if (root/'test-source-sha256.json').exists() else {}
    for name,digest in test_sources.items():assert sha(REPO/name)==sha(root/'source'/name)==digest,name
    wheels=list((root/'wheels').glob('*.whl'));assert len(wheels)==1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for name in source:
            if name.startswith('qrc_pulsar_bench/'):
                assert hashlib.sha256(wheel.read(name)).hexdigest()==source[name],name
    repair=read(root/'harness-repair.json') if (root/'harness-repair.json').exists() else None
    previous=Path(repair['prior_controller_sources']) if repair else None
    if repair:
        assert sha(previous)==repair['prior_manifest_sha256']
        assert all(source[name]==digest for name,digest in read(previous).items())
        assert sha(Path(repair['old_attempt'])/'protocol.json')==sha(root/'protocol.json')
    assert report['protocol_sha256']==sha(root/'protocol.json')
    assert report['runner_sha256']==sha(root/'source/scripts/artemis_anomaly_campaign.py')
    expected={(s.name,e.name,b,'coast_after_event' if c else 'routine_mpc'):(s,e) for s,e,b,c in jobs()}
    assert len(rows)==len(expected) and {key(r) for r in rows}==set(expected)
    by_key={key(r):r for r in rows};budget_peaks=[]
    for row in rows:
        s,e=expected[key(row)]
        assert row['scenario']==json.loads(json.dumps(asdict(s))) and row['anomaly']==asdict(e)
        m=row['metrics'];points=row['dense_samples'];commands=row['commands']
        scored=score_recovery(points,e,completed=m['completed'])
        for name,value in scored.items():assert value==m[name],(key(row),name)
        assert row['nominal_min_clearance_m']>=50000
        assert m['completed']==(not m['collision'] and points[-1]['time_s']==s.days*86400)
        issued=[c for c in commands if c['burn_s']>0]
        assert np.isclose(m['issued_dv_m_s'],sum(np.linalg.norm(c['dv']) for c in issued),atol=1e-12)
        assert m['burns']==len(issued) and m['actual_burns']==sum(c['actual_burn_s']>0 for c in commands)
        assert m['status_counts']=={str(x):sum(c['status']==x for c in commands) for x in {c['status'] for c in commands}}
        assert all(not np.linalg.norm(c['dv']) for c in commands if c['status'])
        peak=max((sum(np.linalg.norm(c['dv']) for c in issued if end['time_s']-86400<c['time_s']<=end['time_s']) for end in issued),default=0.)
        assert peak<=1.+1e-10;budget_peaks.append(float(peak))
        if row['control']=='coast_after_event':assert all(not np.linalg.norm(c['dv']) for c in commands if c['time_s']>=e.onset_s)
        baseline=by_key[(s.name,'none',row['arithmetic'],'routine_mpc')]
        # Identical pre-event truth, observations and commands demonstrates that
        # future injections did not modify the reference or exogenous noise.
        for field in ('samples','dense_samples','commands'):
            assert [v for v in row[field] if v['time_s']<e.onset_s]==[v for v in baseline[field] if v['time_s']<e.onset_s],(key(row),field)
        assert row['initial_state']==baseline['initial_state'] and row['model']==baseline['model']
        if row['control']=='routine_mpc' and e.outage_s==0:
            issued_at_onset=next(c for c in commands if c['time_s']==e.onset_s)
            clean_at_onset=next(c for c in baseline['commands'] if c['time_s']==e.onset_s)
            # The requested command is unchanged. Actual burn execution is a
            # plant output and must change immediately on total thrust loss.
            assert all(issued_at_onset[k]==clean_at_onset[k] for k in issued_at_onset if k!='actual_burn_s')
        if row['control']=='routine_mpc' and row['arithmetic']=='float64' and e.name=='permanent_thrust_failure':
            coast_points=by_key[(s.name,e.name,'float64','coast_after_event')]['dense_samples']
            length=min(len(points),len(coast_points));assert points[:length]==coast_points[:length]
            assert all(not c['actual_burn_s'] for c in commands if c['time_s']>=e.onset_s)
    fault_float=[r for r in rows if r['anomaly']['name']!='none' and r['control']=='routine_mpc' and r['arithmetic']=='float64']
    comparison=[];precision=[]
    for row in fault_float:
        name,event,_,_=key(row);m=row['metrics']
        coast=by_key[(name,event,'float64','coast_after_event')]['metrics']
        coast_row=by_key[(name,event,'float64','coast_after_event')]
        coast_same_epoch=next((p['position_error_m'] for p in coast_row['dense_samples'] if p['time_s']==row['dense_samples'][-1]['time_s']),None)
        baseline=by_key[(name,'none','float64','routine_mpc')]['metrics']
        fixed=by_key[(name,event,'32','routine_mpc')]
        comparison.append(dict(geometry=name,event=event,response=m['response'],strict_success=strict_success(row),
            completed=m['completed'],failure=row.get('failure'),last_sample_h=row['dense_samples'][-1]['time_s']/3600,
            settling_h=m['settling_after_onset_h'],peak_m=m['post_event_peak_m'],rms_m=m['post_event_rms_m'],final_m=m['final_error_m'],
            peak_velocity_error_m_s=max(p['velocity_error_m_s'] for p in row['dense_samples'] if p['time_s']>=row['anomaly']['onset_s']),
            issued_dv_m_s=m['issued_dv_m_s'],extra_issued_dv_m_s=m['issued_dv_m_s']-baseline['issued_dv_m_s'],
            delivered_dv_m_s=m['delivered_dv_m_s'],propellant_kg=m['propellant_kg'],burns=m['burns'],
            coast_response=coast['response'],coast_rms_m=coast['post_event_rms_m'],coast_final_m=coast['final_error_m'],
            coast_at_mpc_stop_m=coast_same_epoch,
            coast_completed=coast['completed'],
            recovery_required_h=m['first_recovery_required_h'],fixes_rejected=m['fixes_rejected'],status_counts=m['status_counts']))
        precision.append(dict(geometry=name,event=event,same_response=m['response']==fixed['metrics']['response'],
            comparable_full_duration=bool(m['completed'] and fixed['metrics']['completed']),
            same_terminal_condition=(row.get('failure',{}).get('message')==fixed.get('failure',{}).get('message')),
            same_strict_success=strict_success(row)==strict_success(fixed),
            rms_difference_m=abs(m['post_event_rms_m']-fixed['metrics']['post_event_rms_m']),
            issued_dv_difference_m_s=abs(m['issued_dv_m_s']-fixed['metrics']['issued_dv_m_s']),
            burn_count_difference=fixed['metrics']['burns']-m['burns']))
    log=(root/'pytest.log').read_text();assert '370 passed' in log
    failures=[dict(geometry=r['scenario']['name'],event=r['anomaly']['name'],arithmetic=r['arithmetic'],control=r['control'],**r['failure']) for r in rows if 'failure' in r]
    verification=dict(status='evidence_verified',runs=len(rows),simulated_days=sum(r['dense_samples'][-1]['time_s']/86400 for r in rows),
        requested_simulated_days=sum(r['scenario']['days'] for r in rows),completed_runs=sum(r['metrics']['completed'] for r in rows),
        numerical_exception_cases=len(failures),anomaly_instances=len(fault_float),source_files=len(source),physical_inputs=len(inputs),tests_passed=370,
        exact_pre_event_pairing=True,onset_commands_unaware_of_hidden_fault=True,recomputed_metrics=True,
        permanent_thrust_failure_matches_coasting=True,wheel_sha256=sha(wheels[0]),
        unchanged_prior_controller_source_files=len(read(previous)) if previous else None,
        archived_test_sources_verified=len(test_sources),
        maximum_rolling_issued_dv_m_s=max(budget_peaks),results_sha256=sha(root/'results.json'),
        continuation_sha256=sha(root/'continuation.json') if continuation else None,
        protocol_sha256=sha(root/'protocol.json'),no_controller_retuning=True,no_manual_rearming=True,
        hardware_synthesis_performed=False)
    save(root/'verification.json',verification)
    summary=dict(verification=verification,comparisons=comparison,precision=precision,numerical_failures=failures,
        baseline=[dict(geometry=r['scenario']['name'],arithmetic=r['arithmetic'],metrics=r['metrics']) for r in rows if r['anomaly']['name']=='none'],
        outcomes={outcome:sum(c['response']==outcome for c in comparison) for outcome in ('contained','recovered','not_recovered')},
        strict_successes=sum(c['strict_success'] for c in comparison),
        coast_settled=sum(c['coast_response']!='not_recovered' for c in comparison))
    save(root/'summary.json',summary)
    dest=REPO/'docs';save(dest/'artemis-anomaly-results-2026-10-31.json',summary)
    lines=['# Blind anomaly injection and automatic recovery','',
        f"The frozen controller meets the sustained recovery criterion in {summary['outcomes']['contained']+summary['outcomes']['recovered']}/18 anomaly instances, with {summary['strict_successes']} also meeting the additional screening conditions. The matched coasting comparisons settle in {summary['coast_settled']}/18 cases. These are finite stress-test outcomes, not estimated probabilities of flight success.",'',
        f"The largest sampled position excursion among recovered/contained floating-point cases is {max((c['peak_m'] for c in comparison if c['response']!='not_recovered'),default=0):.1f} m. Recovery does not mean remaining inside the 100 m corridor throughout the event.",'',
        'Frozen RoutineMPC/sparse_05 is tested without retuning, manual resets or automatic rearming. Each case starts from one reconstructed ARTEMIS state and follows a reference propagated from that initial state. Later vehicle states, sensor fixes and anomalies are synthetic. Future spacecraft observations are never read. The January geometries were previously used; the fault profiles and noise seeds are new stress evidence, not independent new flight datasets.','',
        'Recovery means both position error <=100 m and velocity error <=0.02 m/s for at least six hours continuously through the end, evaluated on five-minute samples. Finite faults are scored after they end; persistent faults require regulation while they remain active. “Contained” means the response never leaves that corridor. “Strict” also requires no clearance/solver/recovery/budget rejection and at least 50 km actual sampled lunar clearance. Temporary stale-navigation inhibition is allowed. No sampled criterion certifies between-sample safety.','',
        'The controller receives only its causal navigation estimate, predicted mass and the nominal reference. Impulses modify plant velocity; its first delayed measurement still predates the kick. Sensor bias follows measurement time. Thrust loss changes actual thrust and propellant flow; the filter sees only its issued command history. Fault labels, timings and directions are available to the experiment and scorer only.','',
        'The generic vehicle starts at 500 kg with 0.2 N thrust and 220 s specific impulse. Faults start 24 h into the three-day P1 run and 48 h into the seven-day P2 run. Existing random force bias, 30 m / 0.002 m/s noisy fixes, 60 s delivery delay and finite-burn errors remain. The amplitudes below are declared engineering stress cases, not estimates of flight fault probabilities.','',
        '| Anomaly | Injected change |','|---|---|',
        '| kick_5mm_s / kick_5cm_s / kick_50cm_s | Instantaneous plant-only velocity jump of 0.005 / 0.05 / 0.5 m/s in a seeded inertial direction |',
        '| acceleration_pulse | Unknown 2e-6 m/s² inertial acceleration for 6 h |',
        '| persistent_force_error | Unknown 3e-7 m/s² inertial acceleration through the end |',
        '| half_thrust_24h | Actual thrust and propellant flow reduced to 50% for 24 h; command model unchanged |',
        '| navigation_bias_6h | Sensor bias of 150 m position and 0.006 m/s velocity for 6 h |',
        '| combined_fault | 0.02 m/s kick, 6 h navigation outage, 75% actual thrust/flow for 12 h |',
        '| permanent_thrust_failure | Zero actual thrust and propellant flow from onset onward |','',
        f"Scored {len(rows)} cases: {verification['completed_runs']} completed their full duration and {len(failures)} stopped on numerical exceptions. Propagated {verification['simulated_days']:.3f} of {verification['requested_simulated_days']:g} requested simulated days. The matrix has two geometries, nine anomalies each, four clean arithmetic baselines, eighteen matched no-correction comparisons and both floating-point/32-bit MPC. No-correction comparisons share the entire controlled pre-event trajectory and then coast.",'',
        f"Floating-point outcomes across eighteen anomaly instances: {summary['outcomes']}; strict successes {summary['strict_successes']}/18. Coast comparisons that settle: {summary['coast_settled']}/18.",'']
    for name in sorted({c['geometry'] for c in comparison}):
        lines += ['## '+name,'','| Anomaly | Response | Strict | Settling from onset (h) | Peak error (m) | Last error (m) | Extra issued Δv (m/s) | Coast at same epoch (m) |',
                  '|---|---|---|---:|---:|---:|---:|---:|']
        for c in comparison:
            if c['geometry']!=name:continue
            settling='—' if c['settling_h'] is None else f"{c['settling_h']:.2f}"
            outcome='numerical failure' if c['failure'] else c['response']
            coast_value='—' if c['coast_at_mpc_stop_m'] is None else f"{c['coast_at_mpc_stop_m']:.2f}"
            lines.append(f"| {c['event']} | {outcome} | {c['strict_success']} | {settling} | {c['peak_m']:.2f} | {c['final_m']:.2f} | {c['extra_issued_dv_m_s']:+.5f} | {coast_value} |")
        lines.append('')
    if failures:
        lines+=['## Numerical exceptions','',
            'These cases terminate at the recorded failure time. Their last error and RMS cover only the available partial history, not the full requested duration. Status 512 is a terminal annotation from the experiment; it is not an onboard recovery mode. No state reset or further trajectory propagation follows the failed numerical check.','',
            '| Geometry | Anomaly | Arithmetic | Control | Stop (h) | Error |','|---|---|---|---|---:|---|']
        for f in failures:lines.append(f"| {f['geometry']} | {f['event']} | {f['arithmetic']} | {f['control']} | {f['time_s']/3600:.2f} | {f['message']} |")
        lines.append('')
    lines += ['Extra issued delta-v is relative to the same geometry/noise clean baseline. A negative value can mean inhibited control or early termination, not improved fuel efficiency. Actual delivered delta-v, propellant, rejection counts and recovery latch times are retained in JSON. All cases, including failures and permanent actuator loss, are included.','',
        f"32/24 arithmetic agrees on response classification in {sum(p['same_response'] for p in precision)}/18 anomaly instances. The hardware controller and fault recovery have not been synthesized or flight-qualified.",'',
        f"Validation: {verification['tests_passed']} tests, {len(source)} frozen numerical source files, {len(inputs)} hashed physical inputs, full case matrix, exact pre-event pairing, recomputed recovery metrics and correction budgets, and packaged-module hashes. Failed runs retain their actual stopping times; no continuation of an invalid navigation state is credited as recovery.",'',
        'The same nominal force model underlies reference, estimator and plant; plant-only faults add model mismatch. This is a controlled causal simulation experiment, not proof against arbitrary faults or validation of direct shape-sphere dynamics.','',
        'Settling time is conservative: it begins after the last corridor excursion and includes background tracking excursions also present in the clean baseline. It is not an isolated fault-detection latency. The position-only plot cannot establish the velocity criterion; both histories are supplied.','',
        *(['An earlier batch attempt was stopped because NumPy burn-count integers could not be serialized to JSON. The repaired harness converts those counts to builtin integers and reports errors immediately. The fault protocol and all 68 prior numerical source files remained unchanged; timing diagnostics and abandoned work are excluded from the 58 scored runs. Receipt: `harness-repair.json` in the campaign directory.',''] if repair else []),
        *([f"The initial batch saved {len(report['results'])} cases before a navigation covariance exception stopped it. A separate continuation retained those exact records and ran only the unfinished cases, capturing exceptions with their partial histories. Both phases use identical frozen numerical sources, physical inputs and fault protocol; no successful case was replaced.",''] if continuation else []),
        '![Matched anomaly histories](artemis-anomaly-tracking-2026-10-31.png)','',
        '![Matched velocity histories](artemis-anomaly-velocity-2026-10-31.png)','',
        '## Reproduction','', '```sh','PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'python scripts/artemis_anomaly_campaign.py --output /path/to/new-campaign \\',
        '  --data /path/to/ARTEMIS-SPKs --kernels /path/to/planetary-kernels \\',
        '  --physics /path/to/lunar-inputs --native /path/to/liblunar.so --workers 4',
        '# If a numerical exception stops the batch, retain its results and continue:',
        'PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'python scripts/artemis_anomaly_continuation.py --output /path/to/new-campaign \\',
        '  --data /path/to/ARTEMIS-SPKs --kernels /path/to/planetary-kernels \\',
        '  --physics /path/to/lunar-inputs --native /path/to/liblunar.so --workers 8',
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q > /path/to/new-campaign/pytest.log',
        'python -m pip wheel . --no-deps --no-build-isolation -w /path/to/new-campaign/wheels',
        'PYTHONPATH=. python scripts/report_artemis_anomalies.py /path/to/new-campaign','```','']
    (dest/'artemis-anomaly-results-2026-10-31.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geometries=sorted({r['scenario']['name'] for r in rows});events=('kick_5cm_s','kick_50cm_s','combined_fault')
    for field,limit,ylabel,suffix in [('position_error_m',100.,'Position error (m)','tracking'),
                                    ('velocity_error_m_s',.02,'Velocity error (m/s)','velocity')]:
        fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for i,event in enumerate(events):
            for j,name in enumerate(geometries):
                for label,ev,control in [('Clean baseline','none','routine_mpc'),('MPC anomaly',event,'routine_mpc'),('Coast after event',event,'coast_after_event')]:
                    row=by_key[(name,ev,'float64',control)];onset=row['anomaly']['onset_s']
                    points=[v for v in row['dense_samples'] if v['time_s']>=onset-6*3600]
                    axes[i,j].plot([(v['time_s']-onset)/3600 for v in points],[max(v[field],limit*1e-5) for v in points],
                        label=label,linewidth=1,linestyle='--' if control=='coast_after_event' else '-',zorder=3 if label=='MPC anomaly' else 2)
                    if row.get('failure'):
                        axes[i,j].plot((points[-1]['time_s']-onset)/3600,points[-1][field],marker='x',color='red',
                            markersize=8,linestyle='none',label='Numerical stop',zorder=4)
                axes[i,j].axvline(0,color='grey',linestyle=':');axes[i,j].axhline(limit,color='black',linestyle='--',linewidth=.7)
                axes[i,j].set(title=name+' / '+event,xlabel='Hours from anomaly onset',ylabel=ylabel,yscale='log');axes[i,j].legend(fontsize=8)
        fig.suptitle('Blind fault injection: frozen MPC versus matched coasting')
        fig.savefig(dest/f'artemis-anomaly-{suffix}-2026-10-31.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
