"""Read changed/new source files since the full audit; check frozen sets for overlap.

This supplement uses size/mtime to find changes. Use a fresh full audit before a
release when independent full-byte verification is required.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from audit_seeds import KEY,CLI,SEED_LIST_CLI,INTEGER,TEXT,CHUNK,OVERLAP,dump,sha
from seed_sets import ROOT


def refresh(root,out):
    summary=json.loads((root/'audit/summary.json').read_text());repo=Path(summary['source_root'])
    original={r['path']:r for r in map(json.loads,(root/'audit/files.jsonl').open())}
    known_hashes={r.get('sha256') for r in original.values()}
    allseeds=set(json.loads((root/'audit/exclusions.json').read_text())['seeds']);records=[];gaps=[]
    for p in sorted(repo.rglob('*')):
        if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        rel=p.relative_to(repo).as_posix();stat=p.stat();previous=original.get(rel)
        if previous and stat.st_size==previous['size_at_inventory'] and stat.st_mtime_ns==previous['mtime_ns_at_inventory']:continue
        if p.suffix not in TEXT and p.suffix!='.gz':
            digest=sha(p)
            if digest in known_hashes:
                records.append(dict(path=rel,bytes=stat.st_size,sha256=digest,mtime_ns_at_read=stat.st_mtime_ns,
                    changed_while_reading=p.stat().st_size!=stat.st_size or p.stat().st_mtime_ns!=stat.st_mtime_ns,
                    seeds=[],status='Binary bytes identical to an already inventoried source file; adjacent text scanned'))
            else:gaps.append(dict(path=rel,reason='Changed binary; fresh full/supplement audit required',sha256=digest))
            continue
        seeds=set();digest=hashlib.sha256();read=0;tail=b''
        with p.open('rb') as f:
            while read<stat.st_size:
                block=f.read(min(CHUNK,stat.st_size-read))
                if not block:break
                read+=len(block);digest.update(block)
                if p.suffix=='.gz':continue
                data=tail+block
                for m in KEY.finditer(data):seeds.update(int(x) for x in INTEGER.findall(m[2]) if 0<=int(x)<2**64)
                for m in CLI.finditer(data):seeds.add(int(m[1]))
                for m in SEED_LIST_CLI.finditer(data):seeds.update(int(x) for x in INTEGER.findall(m[1]))
                tail=data[-OVERLAP:]
        if p.suffix=='.gz':
            with gzip.open(p,'rb') as f:
                tail=b''
                while block:=f.read(CHUNK):
                    data=tail+block
                    for m in KEY.finditer(data):seeds.update(int(x) for x in INTEGER.findall(m[2]) if 0<=int(x)<2**64)
                    tail=data[-OVERLAP:]
        after=p.stat()
        records.append(dict(path=rel,bytes=read,sha256=digest.hexdigest(),mtime_ns_at_read=stat.st_mtime_ns,
            changed_while_reading=after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns,seeds=sorted(seeds)))
        allseeds.update(seeds)
    overlap={}
    for name,path in [('representative','manifests/representative.json'),('stress','manifests/stress.json'),('stress_pool','manifests/stress_pool.json'),('holdout','sealed/holdout.json')]:
        seeds=json.loads((root/path).read_text())['seeds'];overlap[name]=len(set(seeds)&allseeds)
    report=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_root=str(repo),changed_or_new_files=len(records),
        records=records,total_exclusions=len(allseeds),overlap_counts=overlap,status='PASS' if not any(overlap.values()) and not gaps else 'BLOCKED',
        gaps=gaps,scope='Changed/new file supplement; unchanged files trusted by saved size/mtime. Holdout checked numerically only.',
        release_requirement='Re-audit after candidate freeze. Active campaign can add seeds after this timestamp.')
    dump(out,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','gaps')}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();refresh(a.root,a.out)
