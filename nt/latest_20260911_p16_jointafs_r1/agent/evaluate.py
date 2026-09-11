"""Paired 11-opponent check, bounded to eight seeds (16 games/opponent/version).
The first three are existing development seeds; next five are the frozen
confirmation block. Re-running with --seeds 8 adds only missing games.
"""
from pathlib import Path
import argparse,concurrent.futures as cf,multiprocessing as mp,hashlib,json,sys
from arena import game
R=Path(__file__).resolve().parent
SEEDS=[2609125000,2609125001,2609125002,2609205000,2609205001,2609205002,2609205003,2609205004]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run_one(pair):
 arm,job=pair;r=game(job);r['arm']=arm;return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,default=3);p.add_argument('--workers',type=int,default=3);p.add_argument('--binary',type=Path);p.add_argument('--seed-start',type=int,help='Optional new contiguous seeds; never changes the policy input');a=p.parse_args()
 if not 1<=a.seeds<=8 or not 1<=a.workers<=4:p.error('Use 1..8 seeds and 1..4 workers')
 names=[x['id']for x in json.loads((R/'POOL.json').read_text())]
 if len(names)!=11 or len(set(names))!=11:raise SystemExit('Expected 11 distinct frozen opponent entries')
 binaries={'parent':R/'baselines/merged.so','joint':a.binary.resolve() if a.binary else R/'policy/joint.so'}
 seeds=SEEDS if a.seed_start is None else list(range(a.seed_start,a.seed_start+8))
 protocol={'seeds':seeds,'opponents':names,'both_seats':True,'opening':'none','binary_sha256':{k:sha(v)for k,v in binaries.items()},'config_sha256':sha(R/'policy/config.json'),'referee_sha256':sha(R/'referee/official/kaggriculture.py'),'pool_sha256':sha(R/'POOL.json'),'policy_receives':'official current observation only; no seed, opponent id or future','scope':'Local official rules with original realtime opponents; not Kaggle sandbox timing certification'}
 a.out.mkdir(parents=True,exist_ok=True);pp=a.out/'PROTOCOL.json'
 if pp.exists():
  if json.loads(pp.read_text())!=protocol:raise SystemExit('Refusing to mix source/config/binaries/seeds in one run')
 else:pp.write_text(json.dumps(protocol,indent=2))
 log=a.out/'rows.jsonl';rows=[json.loads(x)for x in log.read_text().splitlines()if x.strip()]if log.exists()else[]
 keys=[(r['arm'],r['opponent'],r['seed'],r['opponent_seat'])for r in rows]
 if len(keys)!=len(set(keys)):raise SystemExit('Duplicate rows in existing log')
 done=set(keys);jobs=[]
 for seed in seeds[:a.seeds]:
  for name in names:
   for seat in [0,1]:
    for arm,b in binaries.items():
     if(arm,name,seed,seat)not in done:
      jobs.append((arm,(name,seed,seat,str(b.resolve()),'none',{},719,'')))
 if jobs:
  with cf.ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'))as pool:
   for i,r in enumerate(pool.map(run_one,jobs),1):
    rows.append(r)
    with log.open('a')as f:f.write(json.dumps(r)+'\n')
    if i%22==0 or i==len(jobs):print(json.dumps({'new_done':i,'new_total':len(jobs),'errors':sum(bool(x['runtime_error'])for x in rows)}),flush=True)
 selected=[r for r in rows if r['seed']in seeds[:a.seeds]]
 def stats(rr):return{'games':len(rr),'wins':sum(r.get('win',False)for r in rr),'ties':sum(r.get('tie',False)for r in rr),'errors':sum(bool(r['runtime_error'])for r in rr)}
 result={'overall':{arm:stats([r for r in selected if r['arm']==arm])for arm in binaries},'by_opponent':{name:{arm:stats([r for r in selected if r['arm']==arm and r['opponent']==name])for arm in binaries}for name in names}}
 result['status']='PASS_COMPLETED_NOT_A_WINRATE_GUARANTEE' if not any(bool(r['runtime_error'])for r in selected)else 'FAIL'
 (a.out/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if result['status']=='FAIL':raise SystemExit(1)
if __name__=='__main__':main()
