"""C++ whole-game parameter screens with source snapshots and complete rows."""
from pathlib import Path
import argparse, csv, hashlib, json, shutil, statistics, sys, time, zlib
EXP=Path(__file__).resolve().parents[1];sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native

def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--out',required=True);p.add_argument('--audit',action='store_true');p.add_argument('--new-scenes',action='store_true');p.add_argument('--quiet',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--count',type=int,default=10);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);configs=json.loads(Path(a.configs).read_text())
    data=json.loads(zlib.decompress((EXP/'native/g001_frozen.json.zlib').read_bytes()));g=native.G001(data)
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());src=out/'source'
    for rel,h in build['source_hashes'].items():
        f=EXP/rel;assert hashlib.sha256(f.read_bytes()).hexdigest()==h
        to=src/rel;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,to)
    rows=[];summaries=[];started=time.perf_counter()
    for label,config in configs.items():
        summary=dict(label=label,config=config)
        for mode in ('pass','g001'):
            if a.seed is not None:tasks=[(seed,seat) for seed in range(a.seed,a.seed+a.count) for seat in (0,1)]
            elif a.new_scenes:tasks=[(seed,seat) for seed in range(20261101,20261111) for seat in (0,1)]
            elif mode=='pass':tasks=[(seed,0) for seed in range(10000,10010)]
            else:tasks=[(seed,seat) for seed in range(20260903,20260908) for seat in (0,1)]
            part=native.batch([x[0] for x in tasks],[x[1] for x in tasks],config,g if mode=='g001' else None,16)
            assert len(part)==len(tasks) and all(r['steps']==719 for r in part)
            summary[mode]=dict(wins=sum(x['win'] for x in part),games=len(part),mean_cash=statistics.fmean(x['cash'] for x in part),mean_margin=statistics.fmean(x['margin'] for x in part),min_cash=min(x['cash'] for x in part),overflow=statistics.fmean(x['overflow'][x['seat']] for x in part))
            rows.extend(dict(label=label,opponent=mode,**x) for x in part)
        if a.audit:
            audit=native.audit(20260903,0,config,g)
            (out/(label+'_day_audit.json')).write_text(json.dumps(audit,indent=2),encoding='utf8')
        summaries.append(summary)
        if not a.quiet: print(json.dumps(summary),flush=True)
        if len(summaries)%25==0 or len(summaries)==len(configs):
            (out/'results.json').write_text(json.dumps(dict(summaries=summaries,rows=rows,build=build,seconds=time.perf_counter()-started,new_scenes=a.new_scenes,seed_start=a.seed,seed_count=a.count,opponent_sha256=data['source_sha256']),indent=2),encoding='utf8')
            if a.quiet: print(json.dumps(dict(completed=len(summaries),total=len(configs),best=max((s['g001']['wins'] for s in summaries)),seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
