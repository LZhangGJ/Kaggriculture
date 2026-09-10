"""Predeclared paired small panel. Policy sees no seed or opponent identity."""
from pathlib import Path
import argparse,concurrent.futures as cf,multiprocessing as mp,json,time,hashlib
from arena import game
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=2609125000);p.add_argument('--seeds',type=int,default=3);p.add_argument('--workers',type=int,default=3);p.add_argument('--opening',choices=['none','m12'],default='none');a=p.parse_args()
 if not 1<=a.seeds<=8:raise SystemExit('Maximum 8 seeds / 16 games per opponent')
 if a.out.exists():raise SystemExit('Refusing overwrite')
 a.out.mkdir(parents=True);b=a.binary.resolve(strict=True)
 names=[x['id']for x in json.loads((ROOT/'POOL.json').read_text())];jobs=[]
 for seed in range(a.start,a.start+a.seeds):
  for name in names:
   for seat in (0,1):
    trace=str(a.out.resolve()/f'traces/{name}_{seed}_{seat}.json.gz') if seat==0 else ''
    jobs.append((name,seed,seat,str(b),a.opening,{},719,trace))
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),start=a.start,seeds=a.seeds,opening=a.opening,names=names,both_seats=True,config_sha256=hashlib.sha256((ROOT/'policy/config.json').read_bytes()).hexdigest()),indent=2))
 rows=[];t=time.monotonic()
 with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  futures=[pool.submit(game,j) for j in jobs]
  for f in cf.as_completed(futures):
   r=f.result();rows.append(r)
   with (a.out/'rows.jsonl').open('a') as out:out.write(json.dumps(r)+'\n')
   if len(rows)%6==0 or r['runtime_error']:print(json.dumps(dict(done=len(rows),total=len(jobs),wins=sum(r.get('win',False)for r in rows),errors=sum(bool(r['runtime_error'])for r in rows),seconds=round(time.monotonic()-t,1))),flush=True)
 rows.sort(key=lambda r:(r['seed'],r['opponent'],r['opponent_seat']))
 def count(rr):return dict(games=len(rr),wins=sum(r.get('win',False)for r in rr),errors=sum(bool(r['runtime_error'])for r in rr),ties=sum(r.get('tie',False)for r in rr),mean_margin=sum(r.get('margin',0)for r in rr)/max(1,len(rr)))
 summary={'overall':count(rows),'by_opponent':{n:count([r for r in rows if r['opponent']==n])for n in names},'by_seed':{str(s):count([r for r in rows if r['seed']==s])for s in range(a.start,a.start+a.seeds)},'seconds':time.monotonic()-t}
 (a.out/'rows.json').write_text(json.dumps(rows,indent=2));(a.out/'RESULTS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
 if summary['overall']['errors']:raise SystemExit(1)
if __name__=='__main__':main()
