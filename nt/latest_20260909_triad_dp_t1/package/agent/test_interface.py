"""Observation boundary, config validation and dual-seat state isolation tests."""
from pathlib import Path
import sys,ctypes,importlib.util,json,tempfile,shutil,copy,math,argparse
p=argparse.ArgumentParser();p.add_argument('--arena-root',required=True);p.add_argument('--output',default='interface_tests.json');a=p.parse_args()
root=Path(__file__).resolve().parent;sys.path.insert(0,a.arena_root);import run_arena as arena

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

m=load(root/'agent.py','interface_target');tests=[]
n=len(m._PARAM_ORDER)
arr=(ctypes.c_double*n)(*[float(m.PARAMS[k]) for k in m._PARAM_ORDER])
assert m._LIB.mp_config(arr,n-1)==-1;tests.append('reject wrong configuration length')
bad=(ctypes.c_double*n)(*arr);bad[0]=float('nan');assert m._LIB.mp_config(bad,n)==-1;tests.append('reject non-finite configuration')
assert m._LIB.mp_config(arr,n)==0
state=arena.native.Env(71111);obs=state.observation(0);altered=copy.deepcopy(obs)
altered.update(seed=999999,opponent_id='FAKE',opponent_private={'shed':{'MILK':999999}},future_shops=['YARN_STORE']*8)
assert bytes(m._pack(obs))==bytes(m._pack(altered));tests.append('unknown seed ID future and rival-private fields ignored by serializer')
a1=m.agent(obs,{'seed':123,'opponent_id':'g001'});a2=m.agent(altered,{'seed':987,'opponent_id':'lynn_v5'})
assert a1==a2;tests.append('actions invariant to forbidden extra input fields and configuration seed')
# Use three different library files so isolated references cannot share C++ globals.
with tempfile.TemporaryDirectory() as td:
 tmp=Path(td);mods=[]
 for i in range(3):
  d=tmp/str(i);d.mkdir()
  for f in ('agent.py','agent.so','config.json'):shutil.copy2(root/f,d/f)
  mods.append(load(d/'agent.py',f'isolated_{i}'))
 shared,left,right=mods
 for step in range(719):
  o0,o1=state.observation(0),state.observation(1)
  p0=shared.agent(o0);p1=shared.agent(o1);q0=left.agent(o0);q1=right.agent(o1)
  assert p0==q0 and p1==q1,(step,p0,q0,p1,q1)
  state.step([p0,p1])
 tests.append('719 paired self-play ticks: shared dual-seat state equals two isolated policy instances')
 receipt={'passed':True,'tests':tests,'self_play_calls_compared':1438,'note':'These are state-isolation and information-boundary tests, not opponent win-rate evidence.'}
 Path(a.output).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
