"""Prepare portable fixtures and remove duplicate build caches from the Git package."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gzip,hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parents[1]
OUT=ROOT/'.publish/LZhangGJ_Kaggriculture/nt/latest_20260904_dynamic_generator_cross300'
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
    rid='R03_F001_E104234696_P0'
    with gzip.open(EXP/f'search29_beam12_top100/receipts_v1/{rid}/stage00.json.gz','rt') as f:s=json.load(f)
    write(OUT/'data/day0_frontier_reference.json',dict(route_id=rid,**{k:s[k] for k in ('day','proposals','frontier','beam')}))
    src=EXP/'handoff_20260904/evidence_pretrim';idx=read(src/'index.json');dest=OUT/'evidence';dest.mkdir(exist_ok=False)
    excluded=[]
    def keep(m):
        p=Path(m['path'])
        return 'source' not in p.parts and p.suffix not in ('.so','.o','.a','.pyc') and '__pycache__' not in p.parts
    def work(part):
        path=dest/part['path'];members=[m for m in part['members'] if keep(m)]
        if not members:return None
        with zipfile.ZipFile(src/part['path']) as old,zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as new:
            for m in members:
                data=old.read(m['path']);assert hashlib.sha256(data).hexdigest()==m['sha256'];new.writestr(m['path'],data)
        with zipfile.ZipFile(path) as z:
            for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
        return dict(path=path.name,size=path.stat().st_size,sha256=sha(path),members=members)
    with ThreadPoolExecutor(max_workers=4) as w:parts=[p for p in w.map(work,idx['shards']) if p]
    for p in idx['shards']:
        excluded.extend(m['path'] for m in p['members'] if not keep(m))
    idx.update(shards=parts,files=sum(len(p['members']) for p in parts),
        uncompressed_bytes=sum(m['size'] for p in parts for m in p['members']),compressed_bytes=sum(p['size'] for p in parts),
        excluded_duplicate_source_and_build_artifacts=excluded)
    write(dest/'index.json',idx)
    prep=read(OUT/'PREPARATION_RECEIPT.json');prep.update(evidence_files=idx['files'],shards=len(parts),
        uncompressed_bytes=idx['uncompressed_bytes'],compressed_bytes=idx['compressed_bytes'],
        user_scope='No build cache, toolchain or duplicate source snapshots; source, inputs, full results and reports only',
        excluded_duplicate_source_and_build_artifacts=excluded)
    write(OUT/'PREPARATION_RECEIPT.json',prep)
    shutil.copy2(__file__,OUT/'tools/trim_original_evidence.py')
    print(json.dumps({k:idx[k] for k in ('files','uncompressed_bytes','compressed_bytes')}),flush=True)
if __name__=='__main__':main()
