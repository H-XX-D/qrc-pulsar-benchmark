#!/usr/bin/env python3
"""Verify frozen campaign receipts and export the measured controller settings."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from qrc_pulsar_bench.mpc_strategies import select_strategies, score_run

REPO = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = args.campaign
    protocol = read(root / 'protocol.json')
    sources = read(root / 'source-sha256.json')
    inputs = read(root / 'input-sha256.json')
    for name, digest in sources.items():
        assert sha(REPO / name) == digest, name
        assert sha(root / 'source' / name) == digest, name
    for name, digest in inputs.items():
        assert sha(Path(name)) == digest, name
    training = read(root / 'train.json')
    precision = read(root / 'precision-selection.json')
    selection = training['selection']
    assert select_strategies(training['results']) == selection
    assert precision['training_sha256'] == sha(root / 'train.json')
    assert precision['precision_sha256'] == sha(root / 'precision.json')
    assert precision['selected_bits'] in (24, 32)
    counts = {}
    for phase in ('pilot', 'train', 'precision', 'evaluate'):
        data = read(root / f'{phase}.json')
        assert data['source_sha256'] == sources
        assert data['input_sha256'] == inputs
        assert data['protocol_sha256'] == sha(root / 'protocol.json')
        assert data['runner_sha256'] == sha(REPO / 'scripts/artemis_strategy_campaign.py')
        rows = data['results']
        keys = {(r['case'], r['seed'], r['configuration'], r['arithmetic']) for r in rows}
        assert len(keys) == len(rows)
        for row in rows:
            assert score_run(row) == row['metrics']
            assert row['dense_score_step_s'] == protocol['score_step_s']
            assert row['strategy'] == protocol['settings'][row['configuration']]
        counts[phase] = len(rows)
    assert counts['pilot'] == 4 and counts['train'] == 60 and counts['precision'] == 8
    policies = list(dict.fromkeys(['baseline', *selection['selected']]))
    configurations = [(name, 'float64') for name in policies]
    configurations.append((selection['primary'], str(precision['selected_bits'])))
    expected = {(c['name'], seed, name, bits) for c in protocol['fresh_cases']
                for seed in protocol['fresh_seeds'] for name, bits in configurations}
    assert keys == expected
    followup_path = root / 'sparse05-hardware-qualification.json'
    followup = read(followup_path)
    assert followup['training_sha256'] == sha(root / 'train.json')
    assert followup['evaluation_sha256'] == sha(root / 'evaluate.json')
    assert followup['source_sha256'] == sources
    assert followup['runner_sha256'] == sha(REPO / 'scripts/qualify_artemis_sparse_mpc.py')
    refs = {(r['case'], r['seed']): r for phase in ('train', 'evaluate')
            for r in read(root / f'{phase}.json')['results']
            if r['configuration'] == followup['configuration'] and r['arithmetic'] == 'float64'}
    assert len(followup['results']) == 12
    assert {(r['case'], r['seed']) for r in followup['results']} == set(refs)
    recalculated = []
    for row in followup['results']:
        assert row['configuration'] == followup['configuration'] and row['arithmetic'] == '32'
        assert score_run(row) == row['metrics']
        m, r = row['metrics'], refs[(row['case'], row['seed'])]['metrics']
        dr, dv = abs(m['rms_m']/r['rms_m']-1), abs(m['dv_m_s']/r['dv_m_s']-1)
        db, dp = abs(m['burns']-r['burns']), abs(m['peak_m']-r['peak_m'])
        recalculated.append(dict(case=row['case'], seed=row['seed'], split=row['split'],
                                 rms_relative_change=dr, dv_relative_change=dv, burn_count_change=db,
                                 peak_change_m=dp, passed=dr<=.01 and dv<=.01 and db<=1 and dp<=10
                                 and not m['solver_or_screen_faults'] and not m['collision']))
    assert recalculated == followup['checks']
    assert followup['arithmetic_passed'] == all(c['passed'] for c in recalculated)
    counts['post_evaluation_arithmetic'] = len(followup['results'])
    regression = read(root / 'baseline-regression.json')
    assert all(regression['checks'].values())
    audit = read(root / 'sparse-objective-audit.json')['results']
    objective_error = max(r['relative_objective_excess'] for r in audit)
    command_error = max(r['first_dv_difference_m_s'] for r in audit)
    assert len(audit) == 8 and objective_error < 1e-7 and command_error < 1e-5
    assert '349 passed' in (root / 'pytest.log').read_text()
    wheels = list((root / 'wheels').glob('*.whl'))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        for module in ('mpc_strategies', 'predictive_mpc', 'fpga_mpc', 'artemis_pid_benchmark'):
            name = f'qrc_pulsar_bench/{module}.py'
            assert hashlib.sha256(wheel.read(name)).hexdigest() == sources[name]
    receipts = {name: {'path': str(root / name), 'sha256': sha(root / name)} for name in
                ('protocol.json', 'train.json', 'precision-selection.json', 'evaluate.json',
                 'source-sha256.json', 'input-sha256.json', 'sparse-objective-audit.json',
                 'baseline-regression.json', 'pytest.log', 'sparse05-hardware-qualification.json')}
    fresh = read(root / 'evaluate.json')['results']
    def passes(row, rms, peak):
        m = row['metrics']
        return m['rms_m']<=rms and m['peak_m']<=peak and not m['collision'] and not m['solver_or_screen_faults']
    name = followup['configuration']
    candidate = [r for r in fresh if r['configuration']==name and r['arithmetic']=='float64']
    primary = [r for r in fresh if r['configuration']==selection['primary'] and r['arithmetic']=='float64']
    assessment = dict(training_primary_passes_fresh_acceptance=all(passes(r,100,500) for r in primary),
                      development_candidate_passes_fresh_acceptance=all(passes(r,100,500) for r in candidate),
                      development_candidate_passes_every_tight_case=all(passes(r,50,300) for r in candidate),
                      additional_arithmetic_qualified=followup['arithmetic_passed'])
    config = dict(classification='Frozen research settings; development preference informed by January evaluation, not independently reconfirmed',
                  training_primary=selection['primary'], development_candidate=name, assessment=assessment,
                  training_choices=selection['choices'],
                  settings={name: protocol['settings'][name] for name in selection['selected']},
                  precision=dict(total_bits=followup['bits'], fractional_bits=followup['fraction'],
                                 arithmetic_qualified=followup['arithmetic_passed'],
                                 scope='Bounded integer QP core only', iterations_per_solve=256),
                  hardware_device=None,
                  host_operations=['navigation', 'force propagation', 'QP formation', 'coast trigger',
                                   'scheduling', 'IRLS weights when enabled', 'nonlinear first-burn screening'],
                  validated_duration_days=3, receipts=receipts)
    save(root / 'selected-controller.json', config)
    verification = dict(status='evidence_verification_passed', controller_assessment=assessment,
                        run_counts=counts, total_simulations=sum(counts.values()),
                        source_files=len(sources), physical_inputs=len(inputs),
                        source_and_snapshot_hashes_match=True, metric_recomputation_matches=True,
                        fresh_matrix_complete=True, baseline_regression=regression['checks'],
                        sparse_objective_max_relative_excess=objective_error,
                        sparse_first_command_max_difference_m_s=command_error,
                        tests_passed=349, wheel_sha256=sha(wheels[0]),
                        available_tools={name: shutil.which(name) for name in
                                         ('clang++', 'vitis_hls', 'vivado_hls', 'iverilog', 'verilator')},
                        hardware_synthesis_performed=False, receipts=receipts)
    save(root / 'verification.json', verification)
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    main()
