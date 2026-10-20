"""Download pinned third-party IMS feature tables and verify their hashes.

This retrieves processed CSVs, NOT official raw accelerometer recordings.
Source and raw-data terms are documented in data_docs/README.md.
"""
from pathlib import Path
import argparse
import hashlib
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1]/'data')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'data_docs/provenance.json').read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        out = args.destination/entry['file']
        if out.exists():
            if hashlib.sha256(out.read_bytes()).hexdigest() != entry['sha256']:
                raise RuntimeError(f'{out}: existing file has a different hash; inspect it before replacing')
            print(f'{out.name}: verified existing file')
            continue
        url = ('https://raw.githubusercontent.com/fatemeh-studio/nasa-bearing-anomaly/'
               + manifest['commit'] + '/' + entry['upstream_path'])
        with urlopen(Request(url, headers={'User-Agent':'plasticity-course-project'}), timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise RuntimeError(f'{out.name}: downloaded content does not match the recorded SHA-256')
        tmp = out.with_suffix('.csv.partial')
        tmp.write_bytes(content)
        tmp.replace(out)
        print(f'{out.name}: downloaded and verified')


if __name__=='__main__':
    main()
