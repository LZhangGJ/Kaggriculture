"""Verify listed delivery bytes. This never parses holdout seeds."""
import hashlib
from pathlib import Path
import sys


def main(root):
    count=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,name=line.split('  ',1);path=root/name
        if path.resolve()!=root.resolve() and root.resolve() not in path.resolve().parents:raise ValueError('Checksum path escapes bundle')
        h=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(4*1024*1024):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
        count+=1
    print(f'PASS: {count} file checksums match. No holdout characteristics inspected.')


if __name__=='__main__':main(Path(__file__).resolve().parents[1])
