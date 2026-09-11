"""Verify delivery hashes and local documentation links without compiling."""
from pathlib import Path
import argparse, hashlib, json, re
P=Path(__file__).resolve().parent
EXCLUDE={'build','runs','__pycache__','.venv','.pytest_cache'}

def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
    files=[p for p in P.rglob('*') if p.is_file() and not EXCLUDE.intersection(p.relative_to(P).parts) and p.name!='MANIFEST.json']
    if args.freeze:
        rows={str(p.relative_to(P)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)}
        (P/'MANIFEST.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest=json.loads((P/'MANIFEST.json').read_text())
    assert {str(p.relative_to(P)) for p in files}==set(manifest),'unexpected or missing files'
    for name,row in manifest.items():
        assert sha(P/name)==row['sha256'],name
        assert (P/name).stat().st_size==row['bytes'],name
        assert row['bytes']<100*1024**2,name
    link_count=0
    for doc in P.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):continue
            target=link.split('#')[0]
            assert (doc.parent/target).exists(),(doc.name,target)
            link_count+=1
    provenance=json.loads((P/'SOURCE_PROVENANCE.json').read_text(encoding='utf-8-sig'))
    changed=[]
    for row in provenance:
        if sha(P/row['path'])!=row['sha256']:changed.append(row['path'])
    assert changed==['src/f3_policy.cpp'],changed
    print(json.dumps(dict(status='PASS',files=len(files),bytes=sum(r['bytes'] for r in manifest.values()),
        local_links=link_count,source_changes=changed),indent=2))
if __name__=='__main__':main()
