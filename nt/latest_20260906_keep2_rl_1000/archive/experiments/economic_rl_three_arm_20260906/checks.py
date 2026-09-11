"""G1/G2: action-level baseline preservation, official replay, MLP and PPO audit."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time,gzip,importlib.util,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from runtime import P,ROOT,NAMES,make_pool,rollout,store_result,save,read,digest,summary
ARMS=['c3auto','f3','c3j7']
BASES={'c3auto':ROOT/'experiments/daily_dp_c3_autonomous0_20260905/stage/agent.py',
 'f3':ROOT/'experiments/daily_dp_f5_c2_transfer_20260905/candidates/F3_A0/agent.py',
 'c3j7':ROOT/'experiments/daily_dp_c3_review_20260905/stage/agent.py'}
ARENA=ROOT/'gpt_review/codex/dp_combined_handoff_20260905/KAGGRICULTURE_DP_COMBINED_20260905'
def check_game(job):
 arm,mode,n,out=job;out=Path(out);data=read(out/'games.json')[n]
 with gzip.open(out/'traces.json.gz','rt')as f:actions=json.load(f)[n]
 sys.path.insert(0,str(ARENA));import run_arena as arena
 env=arena.native.Env(data['seed']);official=arena.LocalGame(data['seed']);seat=data['seat']
 rival,state=arena.rival(data['opponent'])
 if mode==0:
  spec=importlib.util.spec_from_file_location('frozen_'+arm,BASES[arm]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  own=m.Agent()if arm=='f3'else m.agent
 def normalized(action):
  def atom(a):
   a=list(a)
   if len(a)==2 and isinstance(a[1],str) and a[0]in('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):a.append(1)
   return a
  return dict(farmer=atom(action['farmer']),hands=[atom(x)for x in action.get('hands',[])],market=[atom(x)for x in action.get('market',[])])
 for step,pair in enumerate(actions):
  if mode==0:assert normalized(own(env.observation(seat)))==normalized(pair[seat]),('KEEP action mismatch',arm,step,seat)
  enemy=rival.act(env,1-seat)if state is None else rival.act(env,1-seat,state)
  assert enemy==pair[1-seat],('opponent mismatch',step)
  env.step(pair);official.advance(pair)
  for p in (0,1):
   a,b=env.observation(p),official.observation(p)
   for k in ('farms','private','market','town','day','hour','player'):
    assert arena.canon(a[k])==arena.canon(b[k]),('official state',arm,step,p,k)
 assert len(actions)==719 and env.done and official.done
 assert env.observation(seat)['farms'][seat]['money']==data['cash']
 return dict(arm=arm,mode=mode,seed=data['seed'],seat=seat,opponent=data['opponent'],steps=719,status='PASS')
def main():
 out=P/'checks_v3';out.mkdir(exist_ok=False);pool=make_pool();jobs=[];stats={};start=time.perf_counter()
 freeze={str(p):digest(p)for p in (P/'build').glob('*.so')}
 for arm in ARMS:
  for mode in (0,3):
   r=rollout(pool,arm,mode=mode,start=64000000,count=1,sample=17,trace=True)
   target=out/f'{arm}_{mode}';store_result(target,r);stats[f'{arm}_{mode}']=summary(r)
   print(arm,mode,json.dumps(summary(r)),flush=True)
   assert not any(x['error']for x in r['rows']),r['rows']
   assert all(x['reference_calls']==0 for x in r['rows'])if arm!='c3j7'else all(x['reference_calls']==29 for x in r['rows'])
   assert np.all(r['changed']==(r['choice']!=0))
   jobs.extend((arm,mode,n,str(target))for n in range(14))
  # Same policy RNG + jobs: scheduling and reset must not change any action.
  one=rollout(pool,arm,mode=3,start=64000000,count=1,sample=17,threads=1,trace=True)
  with gzip.open(out/f'{arm}_3/traces.json.gz','rt')as f:assert json.load(f)==one['traces'],('thread/state isolation',arm)
 rows=[]
 with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'))as workers:
  for f in as_completed([workers.submit(check_game,j)for j in jobs]):
   try:rows.append(f.result())
   except Exception as e:rows.append(dict(status='FAIL',error=repr(e)))
   print('official',len(rows),len(jobs),rows[-1]['status'],flush=True)
 save(out/'interface_receipt.json',dict(status='PASS'if all(r['status']=='PASS'for r in rows)else'FAIL',rows=rows,steps=sum(r.get('steps',0)for r in rows),seconds=time.perf_counter()-start,summaries=stats))
 assert all(r['status']=='PASS'for r in rows),[r for r in rows if r['status']!='PASS']
 assert all(digest(p)==h for p,h in freeze.items())
 save(P/'G1_RECEIPT.json',dict(status='PASS',official_steps=sum(r['steps']for r in rows),keep_equal_steps=sum(r['steps']for r in rows if r['mode']==0),thread_equivalence_games=42,hashes=freeze))
 print('G1 PASS',flush=True)
if __name__=='__main__':main()
