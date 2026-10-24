"""Download cited public PDFs and render the precise Day figure used for extraction.

Already-present PDFs are reused. No paper PDFs are redistributed with results.
Requires Poppler pdftoppm on PATH for raster reproduction.
"""
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
import subprocess
import json,hashlib

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);a=p.parse_args()
    a.sources.mkdir(parents=True,exist_ok=True)
    meta=json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
    for key in ['day2011','grift2019']:
        target=a.sources/(key+'.pdf')
        if not target.exists():
            with urlopen(Request(meta[key]['url'],headers={'User-Agent':'Mozilla/5.0'}),timeout=90) as response:
                content=response.read()
            if not content.startswith(b'%PDF'):raise RuntimeError(f'Not a PDF: {meta[key]["url"]}')
            target.write_bytes(content)
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        print(target.name,'matches recorded PDF:',actual==meta['hashes'][target.name])
        if actual!=meta['hashes'][target.name]:
            raise RuntimeError('Source PDF changed. Recheck axes/pages before extraction; do not silently reuse calibration.')
    subprocess.run(['pdftoppm','-f','12','-singlefile','-r','150','-png',
                    str(a.sources/'day2011.pdf'),str(a.sources/'day_fig11')],check=True)
    print('Ready for tools/extract_experiments.py --sources',a.sources)


if __name__=='__main__':main()
