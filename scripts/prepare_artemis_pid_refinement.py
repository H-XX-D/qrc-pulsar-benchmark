#!/usr/bin/env python3
"""Freeze an accuracy-first choice from September training and fetch fresh dates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();root=a.output
    train=json.loads((root/'pid-train.json').read_text());chosen=min([r for r in train['selection']['candidates'] if r['feasible']],key=lambda r:r['worst_steady_rms_m'])
    digest=hashlib.sha256((root/'pid-train.json').read_bytes()).hexdigest()
    protocol=dict(reason='Accuracy-first follow-up: choose minimum worst September training RMS and evaluate on fresh November cases.',chosen=chosen,selection_data_sha256=digest,
                  fresh_cases=[dict(name=f'P{i}_nov',probe=i,utc='2011-11-01T00:00:00',split='fresh_evaluation') for i in (1,2)],fresh_seeds=[6101,6102],
                  precision_recheck='Compare 24/32-bit against float64 on September training at revised gains before November evaluation.')
    path=root/'pid-accuracy-protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=protocol[k] for k in ('chosen','selection_data_sha256','fresh_cases','fresh_seeds')):raise ValueError('Fresh protocol changed')
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    data=root/'fresh-data';data.mkdir(exist_ok=True)
    for i in (1,2):
        name=f'ARTEMIS_P{i}_DEF_2009022_2012022.bsp';source=root/'data'/name;target=data/name
        if not target.exists():shutil.copy2(source,target)
        if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():raise ValueError('Definitive SPK changed')
    def fetch(item):
        probe,day=item;name=f'{probe}_l1_state_201111{day:02}_v02.cdf';target=data/name
        url=f'https://spdf.gsfc.nasa.gov/pub/data/themis/{probe}/l1/state/2011/{name}'
        if not target.exists():
            partial=target.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=90) as response,partial.open('wb') as f:shutil.copyfileobj(response,f)
            partial.replace(target)
        return dict(url=url,file=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(fetch,[(p,d) for p in ('thb','thc') for d in range(1,5)]))
    (root/'fresh-download-verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen from training:',chosen['configuration'],'; eight fresh V02 files verified')


if __name__=='__main__':main()
