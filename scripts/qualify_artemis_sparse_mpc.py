#!/usr/bin/env python3
"""Post-evaluation arithmetic check of the preselected tight-tracking finalist.

No controller settings change. January cases are reused, not a new holdout.
"""
import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
from artemis_strategy_campaign import initialize, run, read, save, sha, REPO

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'base', 'kernels', 'physics', 'native'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output
    training = read(root / 'train.json')
    evaluation = read(root / 'evaluate.json')
    name = training['selection']['choices']['tightest']
    assert name == 'sparse_05'
    assert len(evaluation['results']) == 32
    sources = read(root / 'source-sha256.json')
    for path, digest in sources.items():
        assert sha(REPO / path) == digest
    for path, digest in read(root / 'input-sha256.json').items():
        assert sha(Path(path)) == digest
    references = [r for d in (training, evaluation) for r in d['results']
                  if r['configuration'] == name and r['arithmetic'] == 'float64']
    assert len(references) == 12
    jobs = [(r['case'], r['seed'], name, '32') for r in references]
    path = root / 'sparse05-hardware-qualification.json'
    report = read(path) if path.exists() else dict(
        classification='Post-evaluation arithmetic qualification; same frozen settings and reused January cases, not new holdout evidence',
        configuration=name, bits=32, fraction=24,
        training_sha256=sha(root / 'train.json'), evaluation_sha256=sha(root / 'evaluate.json'),
        source_sha256=sources, runner_sha256=sha(Path(__file__)), results=[])
    assert report['training_sha256'] == sha(root / 'train.json')
    assert report['evaluation_sha256'] == sha(root / 'evaluate.json')
    assert report['source_sha256'] == sources
    assert report['runner_sha256'] == sha(Path(__file__))
    done = {(r['case'], r['seed'], r['configuration'], r['arithmetic']) for r in report['results']}
    print('Post-evaluation 32-bit checks:', len(jobs) - len(done), flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            initializer=initialize, initargs=(root, args.base, args.kernels, args.physics, args.native, 'evaluate'),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(run, [j for j in jobs if j not in done]):
            report['results'].append(row)
            save(path, report)
            print(row['case'], row['seed'], row['metrics'], flush=True)
    checks = []
    for row in report['results']:
        ref = next(r for r in references if (r['case'], r['seed']) == (row['case'], row['seed']))
        m, r = row['metrics'], ref['metrics']
        dr = abs(m['rms_m'] / r['rms_m'] - 1)
        dv = abs(m['dv_m_s'] / r['dv_m_s'] - 1)
        db = abs(m['burns'] - r['burns'])
        dp = abs(m['peak_m'] - r['peak_m'])
        checks.append(dict(case=row['case'], seed=row['seed'], split=row['split'],
                           rms_relative_change=dr, dv_relative_change=dv,
                           burn_count_change=db, peak_change_m=dp,
                           passed=dr<=.01 and dv<=.01 and db<=1 and dp<=10
                           and not m['solver_or_screen_faults'] and not m['collision']))
    assert len(checks) == 12
    report['checks'] = checks
    report['arithmetic_passed'] = all(c['passed'] for c in checks)
    save(path, report)
    print('CHECKS', checks, flush=True)
    for path, digest in sources.items():
        assert sha(REPO / path) == digest

if __name__ == '__main__':
    main()
