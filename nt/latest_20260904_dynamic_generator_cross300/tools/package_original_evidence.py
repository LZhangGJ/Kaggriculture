"""Package this completed experiment without changing any original evidence."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gzip, hashlib, json, shutil, sys, time, zipfile

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = ROOT / '.publish/LZhangGJ_Kaggriculture/nt/latest_20260904_dynamic_generator_cross300'
SETS = ['search8_multiseed', 'search8_top20_multiseed', 'search29_multiseed',
        'search29_top20_multiseed', 'search29_beam12_top100', 'search29_cross300']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(p,rel):
    dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): assert sha(dest)==sha(p),('Do not overwrite',dest)
    else: shutil.copy2(p,dest)

def main():
    tic=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'evidence/index.json').exists(), 'Already packaged; do not regenerate evidence'
    sys.path.insert(0,str(EXP/'search8_multiseed'))
    import run as original
    frozen,payload=original.inputs(EXP/'search8_multiseed/receipts_v1')
    data=OUT/'data';data.mkdir(exist_ok=True)
    with gzip.open(data/'route_pool.json.gz','wt',encoding='utf8',compresslevel=6) as f:
        json.dump(payload,f,separators=(',',':'))
    write(data/'frozen_inputs.json',frozen)
    copy(EXP/'search29_cross300/receipts_v1/ranked300.json','data/plans300.json')
    # Exact DP27 reference rows, not only aggregated win rates.
    refs={}
    for p in (EXP/'search29_cross300/receipts_v1/games/R03_F001_E104234696_P0/027').glob('*.json'):
        refs[p.stem]=read(p)['rows']
    write(data/'dp27_reference96.json',refs)
    for rel,h in frozen['production']['source_hashes'].items():
        assert sha(EXP/rel)==h,(rel,'frozen production source changed')
        copy(EXP/rel,'engine/'+rel)
    # Also include linked adapters and tests not exhaustively listed by the old receipt.
    for p in (EXP/'native').rglob('*'):
        if p.is_file() and p.suffix in ('.cpp','.hpp','.inc') and 'build' not in p.relative_to(EXP/'native').parts:
            copy(p,'engine/'+p.relative_to(EXP).as_posix())
    copy(EXP/'search8_multiseed/search.cpp','engine/search8_multiseed/search.cpp')
    copy(EXP/'native/build/build_receipt.json','engine/original_build_receipt.json')
    # The official referee is frozen, includes its own schema and runtime wrapper.
    referee=ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'
    copy(referee/'cpu_runtime.py','referee/cpu_runtime.py')
    for p in (referee/'official').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.md'):
            copy(p,'referee/official/'+p.relative_to(referee/'official').as_posix())
    for p in EXP.glob('*.md'): copy(p,'reports/development/'+p.name)
    for p in (EXP/'reports').rglob('*.md'): copy(p,'reports/development/'+p.relative_to(EXP/'reports').as_posix())
    for name in SETS:
        for p in (EXP/name).glob('*'):
            if p.is_file() and p.suffix in ('.md','.csv'):copy(p,'reports/'+name+'/'+p.name)
        for p in (EXP/name/'receipts_v1').glob('*.json'):
            if p.name in ('inputs.json','acceptance.json','summary.json','official.json','build.json','search_done.json','run_done.json'):
                copy(p,'reports/'+name+'/receipts_v1/'+p.name)
    sub=ROOT/'submission/pending_dp27_fixed_macro_dynamic_20260904'
    for p in sub.iterdir():
        if p.is_file():copy(p,'submission/dp27/'+p.name)
    # All six raw trees, including failures and original scripts; only bytecode/cache excluded.
    files=[];excluded=[]
    for name in SETS:
        for p in sorted((EXP/name).rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(EXP).as_posix()
            if '__pycache__' in p.parts or p.suffix=='.pyc':excluded.append(rel)
            else:files.append(p)
    groups=[];group=[];size=0
    for p in files:
        n=p.stat().st_size
        if group and size+n>128*1024**2:groups.append(group);group=[];size=0
        group.append(p);size+=n
    if group:groups.append(group)
    dest=OUT/'evidence';dest.mkdir(exist_ok=True)
    def pack(job):
        index,items=job;path=dest/f'part-{index:03d}.zip';members=[]
        assert not path.exists(),path
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p in items:
                rel=p.relative_to(EXP).as_posix();z.write(p,rel)
                members.append(dict(path=rel,size=p.stat().st_size,sha256=sha(p)))
        assert path.stat().st_size<90*1024**2,('Git file too big',path)
        # Actually decompress each entry and compare SHA256, not just ZIP central directory CRC.
        with zipfile.ZipFile(path) as z:
            for m in members:
                with z.open(m['path']) as f:
                    h=hashlib.sha256()
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                assert h.hexdigest()==m['sha256'],m['path']
        result=dict(path=path.name,size=path.stat().st_size,sha256=sha(path),members=members)
        print(json.dumps(dict(shard=index,files=len(items),compressed_bytes=result['size'])),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as workers:shards=list(workers.map(pack,enumerate(groups,1)))
    write(dest/'index.json',dict(schema='lossless-evidence-shards-v1',status='ALL_MEMBERS_DECOMPRESSED_SHA256_VERIFIED',
        original_root=str(EXP),sets=SETS,files=len(files),uncompressed_bytes=sum(p.stat().st_size for p in files),
        compressed_bytes=sum(x['size'] for x in shards),excluded_cache_files=excluded,shards=shards))
    copy(__file__,'tools/package_original_evidence.py')
    write(OUT/'PREPARATION_RECEIPT.json',dict(status='FROZEN_SOURCE_AND_EVIDENCE_PACKED',seconds=time.perf_counter()-tic,
        original_source_hashes=frozen['production']['source_hashes'],pool_routes=len(payload['routes']),
        evidence_files=len(files),shards=len(shards),uncompressed_bytes=sum(p.stat().st_size for p in files),
        compressed_bytes=sum(x['size'] for x in shards),excluded=excluded,
        caveat='Package preparation only. Portable build and execution must pass separately.'))
    print(json.dumps(read(OUT/'PREPARATION_RECEIPT.json')|{'original_source_hashes':'see receipt','excluded':'see receipt'}),flush=True)
if __name__=='__main__':main()
