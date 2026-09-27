"""Build all exact labels and numeric features without starting training."""
import argparse,json,pickle,time,traceback
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from compression import zstd
import torch
from exact_records import load_seat
from exact_decoder import prepare_turn
from exact_training import pack_turns
from features_v2 import PublicHistory
from exact_identity import make_identity,validate_identity
from cache_identity import file_sha


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def build(entry,output,identity):
    torch.set_num_threads(1);start=time.monotonic();source=Path(entry['source']);sha=file_sha(source)
    path=Path(output)/(source.name+'.exact.zst');receipt=path.with_suffix('.json')
    if path.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha']==sha and old['cache_digest']==identity['digest'] and old['sha']==file_sha(path):return old
        raise ValueError('Existing cache receipt mismatch')
    turns=load_seat(source);history=PublicHistory();stats=Counter();tmp=path.with_suffix('.tmp')
    with zstd.open(tmp,'wb',level=3) as stream:
        for begin in range(0,len(turns),identity['chunk']):
            prepared=[prepare_turn(t,history,identity['worker_quantities'],identity['market_quantities'])
                      for t in turns[begin:begin+identity['chunk']]]
            chunk=pack_turns(prepared,begin,entry['outcome']);stats.update(chunk['stats'])
            pickle.dump(chunk,stream,protocol=5)
    tmp.replace(path)
    result=dict(entry,file=str(path.resolve()),source_sha=sha,cache_digest=identity['digest'],sha=file_sha(path),
                bytes=path.stat().st_size,seconds=time.monotonic()-start,turns=len(turns),stats=dict(stats))
    write_json(receipt,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--chunk',type=int,default=16);p.add_argument('--limit',type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);identity=make_identity(a.root,a.audit,a.chunk,a.limit)
    ip=a.output/'identity.json'
    if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Output belongs to a different cache')
    write_json(ip,identity)
    rows=[json.loads(l) for l in (a.root/'manifest.jsonl').read_text().splitlines()]
    entries=[dict(source=f,outcome=r['outcomes'][s],episode=r['episode'],seat=s) for r in rows for s,f in enumerate(r['files'])]
    if a.limit:entries=entries[:a.limit]
    if len({e['source'] for e in entries})!=len(entries):raise ValueError('Duplicate trajectories')
    start=time.monotonic();results=[];stats=Counter()
    status=dict(complete=False,trajectories=0,total=len(entries),seconds=0,bytes=0,cache_digest=identity['digest'])
    write_json(a.output/'status.json',status)
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(build,e,str(a.output),identity) for e in entries]
            for future in as_completed(futures):
                r=future.result();results.append(r);stats.update(r['stats'])
                status.update(trajectories=len(results),seconds=time.monotonic()-start,bytes=sum(r['bytes'] for r in results),stats=dict(stats))
                write_json(a.output/'status.json',status)
                print(json.dumps({k:v for k,v in status.items() if k!='stats'}),flush=True)
        results.sort(key=lambda r:(r['episode'],r['seat']))
        (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
        validate_identity(identity)
        status.update(complete=True,manifest_sha=file_sha(a.output/'manifest.jsonl'),turns=sum(r['turns'] for r in results))
        write_json(a.output/'status.json',status)
    except BaseException as exc:
        status.update(failed=True,error=repr(exc));write_json(a.output/'status.json',status);raise


if __name__=='__main__':main()
