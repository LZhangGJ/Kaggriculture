"""Fetch a hash-pinned checkpoint from this repository's GitHub release."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['epoch6','epoch7','encoder'],default='epoch6')
    p.add_argument('--output',type=Path,default=Path('checkpoints'));a=p.parse_args()
    info=json.loads((Path(__file__).parent/'metadata/checkpoints.json').read_text());entry=info['assets'][a.name]
    a.output.mkdir(parents=True,exist_ok=True);dest=a.output/entry['file']
    if not dest.exists():
        subprocess.run(['gh','release','download',info['tag'],'--repo',info['repo'],'--pattern',entry['file'],'--dir',str(a.output)],check=True)
    with dest.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=entry['sha256']:raise ValueError('Checkpoint hash mismatch; do not load this file')
    print(str(dest.resolve()))

if __name__=='__main__':main()
