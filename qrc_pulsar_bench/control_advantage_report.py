"""Summarize completed frozen experiments without selecting new settings.

Usage: python -m qrc_pulsar_bench.control_advantage_report INPUT_DIR OUTPUT_DIR
Requires the optional matplotlib plotting dependency. Raw receipts stay in the
input directory; their hashes and compact numerical results accompany the plot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


LABELS = ('original', 'tuned_mpc', 'tuned_pd', 'tuned_target',
          'cartesian_mpc', 'no_bias_feedforward')
NAMES = dict(original='Original', tuned_mpc='Predictive', tuned_pd='PD',
             tuned_target='Target point', cartesian_mpc='Cartesian predictive',
             no_bias_feedforward='No bias feedforward')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('samples', 'commands')}
    result['online_cpu_s'] = row['planner_cpu_s'] + row['navigation_cpu_s']
    result['online_force_calls'] = sum(c['online_force_calls'] for c in row['commands'])
    result['minimum_burn_spacing_s'] = None
    burns = [c['time_s'] for c in row['commands'] if c['dv_m_s'] > 0]
    if len(burns) > 1:
        result['minimum_burn_spacing_s'] = min(np.diff(burns))
    return result


def aggregate(rows):
    metrics = ('steady_rms_m', 'rms_m', 'delta_v_m_s', 'fuel_kg',
               'planner_cpu_s', 'navigation_cpu_s', 'online_cpu_s',
               'burn_count', 'navigation_rms_m')
    result = {f'mean_{key}': float(np.mean([r[key] for r in rows])) for key in metrics}
    result.update(cases=len(rows), worst_steady_rms_m=max(r['steady_rms_m'] for r in rows),
                  maximum_sampled_error_m=max(r['max_m'] for r in rows),
                  max_online_force_calls=max(r['max_online_force_calls'] for r in rows),
                  force_budget_violations=sum(r['force_budget_violations'] for r in rows))
    return result


def build_summary(source):
    reports = {name: json.loads((source / f'{name}.json').read_text())
               for name in ('train', 'evaluate', 'month')}
    train = reports['train']
    for report in reports.values():
        if (report['source_sha256'] != train['source_sha256'] or
                report['protocol_sha256'] != train['protocol_sha256']):
            raise ValueError('Experiment source or protocol changed between phases')
    if train['protocol_sha256'] != digest(source / 'protocol.json'):
        raise ValueError('Protocol receipt mismatch')
    for name, expected in train['source_sha256'].items():
        archived = source / 'source' / name
        implementation = archived if archived.exists() else Path(__file__).parent / name
        if digest(implementation) != expected:
            raise ValueError(f'Experiment source mismatch: {name}')
    configuration_names = [c['configuration'] for family in train['selection'].values()
                           for c in family['candidates']]
    training_keys = [(r['scenario']['name'], r['configuration']) for r in train['results']]
    expected_training = {(f'train_{letter}', name) for letter in 'abc' for name in configuration_names}
    if (len(configuration_names) != 24 or len(training_keys) != 72 or
            set(training_keys) != expected_training):
        raise ValueError('Incomplete or duplicate training results')
    expected = {
        'evaluate': {(f'test_{letter}', label) for letter in 'abcdef' for label in LABELS},
        'month': {('month_holdout', label) for label in LABELS[:4]},
    }
    for phase, keys in expected.items():
        rows = reports[phase]['results']
        actual = [(r['scenario']['name'], r['label']) for r in rows]
        if len(set(actual)) != len(actual) or set(actual) != keys:
            raise ValueError(f'Incomplete or duplicate {phase} results')
        if reports[phase]['frozen_selection'] != train['selection']:
            raise ValueError('Evaluation did not use the frozen selection')
    rows = [compact(r) for r in reports['evaluate']['results']]
    indexed = {(r['scenario']['name'], r['label']): r for r in rows}
    receipt_names = ['train', 'evaluate', 'month', 'protocol']
    receipt_names.extend(n for n in ('coordinate-summary', 'coordinate-accuracy',
                                    'coordinate-adaptive-accuracy') if (source / f'{n}.json').exists())
    summary = dict(
        experiment='Synthetic high lunar orbit controller comparison, 2026-10-31',
        metric_definition='Arithmetic mean of six scenario RMS values; steady excludes first 24 h. Errors sampled every 6 h.',
        limitations=['Synthetic independent direct-state fixes, not flight telemetry.',
                     'Bounded training grid; six evaluation cases and one month scenario, not a failure probability.',
                     'CPU is host process time; online CPU excludes reference construction and truth propagation.',
                     'No direct shape-sphere dynamics or FPGA timing measured.'],
        source_sha256=train['source_sha256'], protocol_sha256=train['protocol_sha256'],
        inputs=train['inputs'], selection=train['selection'],
        receipts={f'{n}.json': dict(path=str(source / f'{n}.json'), sha256=digest(source / f'{n}.json'))
                  for n in receipt_names},
        evaluation=sorted(rows, key=lambda r: (r['scenario']['name'], LABELS.index(r['label']))),
        aggregate={label: aggregate([r for r in rows if r['label'] == label]) for label in LABELS},
        subsets={subset: {label: aggregate([r for r in rows if r['label'] == label
                                           and r['scenario']['name'] in cases]) for label in LABELS}
                 for subset, cases in [('ordinary_fixes', ['test_a', 'test_b']),
                                        ('stressed_cases', ['test_c', 'test_d', 'test_e', 'test_f'])]},
        month=sorted([compact(r) for r in reports['month']['results']], key=lambda r: LABELS.index(r['label'])),
        coordinate_accuracy=(json.loads((source / 'coordinate-summary.json').read_text())
                             if (source / 'coordinate-summary.json').exists() else []),
    )
    summary['paired_mpc_vs_target'] = [dict(
        scenario=f'test_{letter}',
        rms_ratio=indexed[(f'test_{letter}', 'tuned_mpc')]['steady_rms_m'] / indexed[(f'test_{letter}', 'tuned_target')]['steady_rms_m'],
        dv_ratio=indexed[(f'test_{letter}', 'tuned_mpc')]['delta_v_m_s'] / indexed[(f'test_{letter}', 'tuned_target')]['delta_v_m_s'],
        online_cpu_ratio=indexed[(f'test_{letter}', 'tuned_mpc')]['online_cpu_s'] / indexed[(f'test_{letter}', 'tuned_target')]['online_cpu_s']) for letter in 'abcdef']
    summary['coordinate_policy_comparison'] = [dict(
        scenario=f'test_{letter}',
        absolute_steady_rms_difference_m=abs(indexed[(f'test_{letter}', 'tuned_mpc')]['steady_rms_m'] - indexed[(f'test_{letter}', 'cartesian_mpc')]['steady_rms_m']),
        absolute_dv_difference_m_s=abs(indexed[(f'test_{letter}', 'tuned_mpc')]['delta_v_m_s'] - indexed[(f'test_{letter}', 'cartesian_mpc')]['delta_v_m_s'])) for letter in 'abcdef']
    return summary, reports


def plot(summary, reports, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = dict(original='#8c564b', tuned_mpc='#146a9c', tuned_pd='#777777', tuned_target='#d17d13')
    ax = axes[0, 0]
    for family, color in [('mpc', colors['tuned_mpc']), ('target', colors['tuned_target']), ('pd', colors['tuned_pd'])]:
        candidates = summary['selection'][family]['candidates']
        ax.scatter([r['mean_delta_v_m_s'] for r in candidates], [r['mean_steady_rms_m'] for r in candidates],
                   color=color, label=family.upper() if family != 'target' else 'Target point', alpha=.7)
        selected = next(r for r in candidates if r['configuration'] == summary['selection'][family]['selected'])
        ax.scatter(selected['mean_delta_v_m_s'], selected['mean_steady_rms_m'], marker='*', s=180, color=color, edgecolor='black')
    ax.axhline(300, color='black', linestyle=':', linewidth=1)
    ax.set(xscale='log', yscale='log', xlabel='Mean commanded delta-v over 6 days (m/s)', ylabel='Mean steady RMS (m)', title='Training only: 8 settings per family')
    ax.legend(fontsize=8)
    cases = [f'test_{letter}' for letter in 'abcdef']
    for label in ('original', 'tuned_mpc', 'tuned_target'):
        rows = [next(r for r in summary['evaluation'] if r['scenario']['name'] == case and r['label'] == label) for case in cases]
        axes[0, 1].plot(range(6), [r['steady_rms_m'] for r in rows], 'o-', color=colors[label], label=NAMES[label])
        axes[1, 0].plot(range(6), [r['delta_v_m_s'] for r in rows], 'o-', color=colors[label], label=NAMES[label])
        month = next(r for r in reports['month']['results'] if r['label'] == label)
        axes[1, 1].plot([s['time_s']/86400 for s in month['samples']], [s['error_m'] for s in month['samples']], color=colors[label], label=NAMES[label])
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xticks(range(6), ['A', 'B', 'C', 'D', 'E', 'F'])
        ax.set_xlabel('Frozen evaluation case')
        ax.legend(fontsize=8)
    axes[0, 1].set(yscale='log', ylabel='Steady RMS (m)', title='Unseen 9-day cases: tracking')
    axes[1, 0].set(ylabel='Commanded delta-v (m/s)', title='Unseen 9-day cases: correction cost')
    axes[1, 1].axvspan(15, 17, color='gray', alpha=.15, label='Lost fixes')
    axes[1, 1].set(yscale='log', xlabel='Mission day', ylabel='Sampled position error (m)', title='Separate 30-day case')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Synthetic lunar orbit control — common forces, sensors and actuator limits', fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary, reports = build_summary(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'controller-advantage-results-2026-10-31.json'
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plot(summary, reports, args.output / 'controller-advantage-results-2026-10-31.png')
    print(destination)


if __name__ == '__main__':
    main()
