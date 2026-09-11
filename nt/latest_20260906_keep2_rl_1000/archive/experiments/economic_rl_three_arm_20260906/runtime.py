from pathlib import Path
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,zlib,hashlib,time
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(P/'build'))
import _economic_rl_native as native
NAMES=['g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day']
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,x):
 f=Path(f);tmp=f.with_name(f.name+'.writing');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(f)
def make_pool():
 reg=ROOT/'experiments/daily_dp_v7_20260903/opponents/registry.json';meta=read(reg)['opponents'];data={}
 for n in NAMES[:5]:
  path=reg.parent.parent/meta[n]['asset'];data[n]=json.loads(zlib.decompress(path.read_bytes()))
  assert data[n]['source_sha256']==meta[n]['source_sha256']==digest(reg.parent.parent/meta[n]['source'])
 return native.RLPool(data)
def jobs(start,count):
 return [(s,seat,j) for s in range(start,start+count) for j in range(7) for seat in (0,1)]
def rollout(pool,arm,model='',mode=0,start=64000000,count=1,sample=1,threads=16,trace=False,joblist=None):
 j=jobs(start,count) if joblist is None else joblist
 tic=time.perf_counter()
 result=pool.batch(str(P/'build'/f'{arm}.so'),str(model),mode,[x[0]for x in j],[x[1]for x in j],[x[2]for x in j],sample,threads,trace)
 result['call_seconds']=time.perf_counter()-tic
 result['bridge_seconds']=result['call_seconds']-result['wall_seconds']
 result['mode']=mode
 for r in result['rows']:r['opponent']=NAMES[r['opponent']]
 return result
def summary(result):
 rows=result['rows'];good=[r for r in rows if not r['error']]
 def group(rr):
  return dict(games=len(rr),wins=sum(r['win']for r in rr),win_rate=np.mean([r['win']for r in rr]) if rr else None,
   mean_cash=np.mean([r['cash']for r in rr]) if rr else None,mean_margin=np.mean([r['margin']for r in rr]) if rr else None)
 s=dict(status='PASS' if len(good)==len(rows) else 'FAIL',overall=group(good),per_opponent={n:group([r for r in good if r['opponent']==n])for n in NAMES},wall_seconds=result['wall_seconds'],
  mode=result.get('mode'),
  call_seconds=result.get('call_seconds',result['wall_seconds']),bridge_seconds=result.get('bridge_seconds',0),
  games_per_second=len(good)/result['wall_seconds'],environment_steps_per_second=sum(r['steps']for r in good)/result['wall_seconds'],
  effective_actor_records=int(np.sum(np.sum(result['mask'],axis=1)>1)),decisions=len(result['choice']),
  non_keep=int(np.sum(result['choice']!=0)),changed=int(np.sum(result['changed'])),
  max_action_ms=max(r['max_action_ms']for r in rows),errors=[r for r in rows if r['error']],
  mean_times={k:np.mean([r[k]for r in good])for k in ('policy_seconds','enemy_seconds','env_seconds','model_seconds')})
 return s
def store_result(out,result):
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 save(out/'games.json',result['rows']);save(out/'summary.json',summary(result))
 np.savez_compressed(out/'decisions.npz',**{k:v for k,v in result.items()if isinstance(v,np.ndarray)})
 if any(result['traces']):
  import gzip
  with gzip.open(out/'traces.json.gz','wt')as f:json.dump(result['traces'],f)
