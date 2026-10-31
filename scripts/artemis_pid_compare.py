#!/usr/bin/env python3
"""Run the original frozen gain on fresh cases as a paired comparator, not tune it."""
import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
from artemis_pid_accuracy_validation import initialize,run,read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','kernels','physics','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args();root=a.output
    training=read(root/'pid-train.json');protocol=read(root/'pid-accuracy-protocol.json')
    if sha(root/'pid-train.json')!=protocol['selection_data_sha256']:raise ValueError('Training receipt changed')
    name=training['selection']['selected'];repo=Path(__file__).resolve().parents[1]
    if any(sha(repo/n)!=s for n,s in training['source_sha256'].items()):raise ValueError('Implementation changed')
    jobs=[(c['name'],s,name,'float64') for c in protocol['fresh_cases'] for s in protocol['fresh_seeds']]
    report=dict(classification='Paired comparison of original frozen fuel-first gain; no new selection',configuration=name,
                source_sha256=training['source_sha256'],protocol_sha256=sha(root/'pid-accuracy-protocol.json'),runner_sha256=sha(Path(__file__)),results=[])
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,
         initargs=(root,a.kernels,a.physics,a.native,'fresh'),mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(run,jobs):
            report['results'].append(row);save(root/'pid-accuracy-comparison.json',report)
            print(row['case'],row['seed'],row['steady_rms_m'],row['dv_m_s'],flush=True)


if __name__=='__main__':main()
