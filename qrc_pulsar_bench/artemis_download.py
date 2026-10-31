"""Download the explicitly versioned definitive products for the replay.

No fallback from definitive V02 files to predictive V01 files is allowed.
Existing files are retained; checksums identify the exact local input bytes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def download(output):
    output=Path(output);data=output/'data';data.mkdir(parents=True,exist_ok=True)
    base='https://naif.jpl.nasa.gov/pub/naif/THEMIS/kernels/spk/'
    jobs=[(base+f'ARTEMIS_P{i}_DEF_2009022_2012022.bsp',data/f'ARTEMIS_P{i}_DEF_2009022_2012022.bsp') for i in (1,2)]
    for month in (2,9,10):
        for day in range(1,5):
            for probe in ('thb','thc'):
                name=f'{probe}_l1_state_2011{month:02}{day:02}_v02.cdf'
                jobs.append((f'https://spdf.gsfc.nasa.gov/pub/data/themis/{probe}/l1/state/2011/{name}',data/name))
    jobs.extend([(base+'aareadme.txt',data/'spk-readme.txt'),
        ('https://spdf.gsfc.nasa.gov/pub/data/themis/documents/tdas/docs/thm_soc_124_L1_File_Def.pdf',data/'thm_soc_124_L1_File_Def.pdf')])
    def fetch(job):
        url,path=job;cached=path.exists()
        if not cached:
            temporary=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=90) as response,temporary.open('wb') as target:
                shutil.copyfileobj(response,target)
            temporary.replace(path)
        return dict(url=url,file=str(path),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reused=cached)
    with ThreadPoolExecutor(max_workers=4) as pool:manifest=list(pool.map(fetch,jobs))
    # Keep acquisition history rather than overwrite the original manifest.
    (output/'download-verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args();print(f'Verified {len(download(a.output))} files in {a.output}')


if __name__=='__main__':main()
