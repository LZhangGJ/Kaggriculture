"""Fresh candidate trace: re-run both actions through official Python rules.
Also re-run candidate via Python ctypes on each official observation and require
identical actions. This tests the exported wrapper, not only C++ self-consistency.
Opponent Python equivalence is not claimed by this test.
"""
from pathlib import Path
import sys, json, gzip, importlib.util, time, argparse
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'arena/gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
sys.path.insert(0,str(R/'arena/experiments/daily_dp_v7_20260903/native/build'))
from cpu_runtime import LocalGame,load_engine
import _dp7_native as native
spec=importlib.util.spec_from_file_location('triad_agent',R/'policy/agent.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def normalized(a):
 def atom(x):
  if x[0] in ('PICKUP','PLACE'):return x[:2]+[x[2] if len(x)>2 else 1]
  if x[0] in ('BUY_PRODUCT','BUY_ANIMAL','BUY_SEED','SELL'):return x[:3]
  if x[0]=='PLANT':return x[:2]
  return x[:1]
 return dict(farmer=atom(a.get('farmer',['PASS'])),hands=[atom(x) for x in a.get('hands',[])],market=[atom(x) for x in a.get('market',[])])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--out',required=True);ap.add_argument('--limit',type=int,default=14);a=ap.parse_args()
 folder=R/'runs'/a.run;protocol=json.loads((folder/'protocol.json').read_text());out=R/a.out;out.mkdir(exist_ok=False,parents=True)
 engine=load_engine();rows=[];tic=time.perf_counter()
 for p in sorted(folder.glob('trace_*.json.gz'))[:a.limit]:
  t=json.loads(gzip.decompress(p.read_bytes()));seed=t['seed'];seat=t['seat'];game=LocalGame(seed,engine=engine);env=native.Env(seed)
  ag=module.Agent(protocol['config'],R/'snapshots'/(protocol['binary_hash']+'.so'))
  for step,aa in enumerate(t['actions']):
   got=ag(game.observation(seat),game.configuration)
   if normalized(got)!=normalized(aa[seat]):
    raise AssertionError(dict(kind='wrapper_action',seed=seed,seat=seat,step=step,actual=got,expected=aa[seat]))
   game.advance(aa);env.step(aa)
   for pp in (0,1):
    obs=game.observation(pp);nobs=env.observation(pp)
    for field in ('farms','private','market','town','day','hour'):
     if json.loads(json.dumps(obs[field]))!=json.loads(json.dumps(nobs[field])):
      raise AssertionError(dict(kind='official_state',seed=seed,seat=seat,step=step,field=field))
  assert game.done and env.done
  f=game.observation(seat)['farms'];row=dict(seed=seed,seat=seat,opponent=t['opponent'],transitions=len(t['actions']),cash=f[seat]['money'],opponent_cash=f[1-seat]['money'],state_mismatches=0,wrapper_action_mismatches=0)
  ag.close();rows.append(row);print(json.dumps(row),flush=True)
 (out/'acceptance.json').write_text(json.dumps(dict(status='PASS',scope='official rules plus exported candidate action parity; not proof on all states, not opponent Python parity',binary_hash=protocol['binary_hash'],source_hash=protocol['source_hash'],rows=rows,seconds=time.perf_counter()-tic),indent=2))
if __name__=='__main__':main()
