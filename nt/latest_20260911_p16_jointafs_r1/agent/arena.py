"""Small official live panel; prefix is fixture only; policy observes no seed or opponent id."""
import sys,os,copy,json,gzip,hashlib,importlib.util,time,traceback,concurrent.futures as cf,multiprocessing as mp,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'referee'))
from cpu_runtime import LocalGame,load_engine

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
codec=module(ROOT/'policy/agent.py','p16_codec')
class FreshPublic:
 def __init__(self,name):
  self.directory=ROOT/'opponents'/name;self.cwd=Path.cwd();self.path=sys.path[:];self.names={p.stem for p in self.directory.glob('*.py')};self.displaced={n:sys.modules.pop(n) for n in self.names if n in sys.modules};sys.path.insert(0,str(self.directory));os.chdir(self.directory)
  self.mod=module(self.directory/'main.py','main');self.call=self.mod.agent
 def close(self):
  for n,m in list(sys.modules.items()):
   p=getattr(m,'__file__',None)
   if n in self.names or p and Path(p).resolve().is_relative_to(self.directory):sys.modules.pop(n,None)
  sys.modules.update(self.displaced);sys.path[:]=self.path;os.chdir(self.cwd)
def digest(a):return hashlib.sha256(json.dumps(a,separators=(',',':')).encode()).hexdigest()
def game(job):
 name,seed,seat,binary,opening,config,stop,trace=job
 t=time.monotonic();pub=agent=None;row=dict(opponent=name,seed=seed,opponent_seat=seat,binary=binary,opening=opening,runtime_error=None)
 try:
  pub=FreshPublic(name);cfg=json.loads((ROOT/'policy/config.json').read_text());cfg.update(config)
  agent=codec.Agent(config=cfg,binary_path=binary);env=LocalGame(seed,load_engine());prefix=json.loads((ROOT/'fixtures/melon12_opening.json').read_text())['actions'];days=[];actions=[];lat=0.;brief=[]
  while not env.done and env.t<stop:
   obs=[env.observation(i)for i in (0,1)];own=obs[1-seat];step=env.t
   if step%24==0:days.append(dict(step=step,observations=copy.deepcopy(obs),debug=agent.debug()))
   if opening=='reset' and step==24:agent.close();agent=codec.Agent(config=cfg,binary_path=binary)
   start=time.monotonic()
   a=copy.deepcopy(prefix[step]) if opening=='m12' and step<24 else agent(own)
   lat=max(lat,time.monotonic()-start)
   b=pub.call(obs[seat],copy.deepcopy(env.configuration));out=[None,None];out[1-seat]=a;out[seat]=b
   assert env.configuration.seed is None
   if step<96:brief.append({'step':step,'own':copy.deepcopy(own),'action':copy.deepcopy(a),'debug':agent.debug()})
   actions.append(copy.deepcopy(out));env.advance(out)
  obs=[env.observation(i)for i in (0,1)];days.append(dict(step=env.t,observations=copy.deepcopy(obs),debug=agent.debug()));cash=[f['money']for f in obs[0]['farms']];margin=cash[1-seat]-cash[seat]
  row.update(steps=env.t,own_cash=cash[1-seat],opponent_cash=cash[seat],margin=margin,win=margin>0,tie=margin==0,action_hash=digest(actions),latency_max=lat)
  row['day2_animals']={}
  for d in days:
   if d['step']==48:
    for r in d['observations'][1-seat]['farms'][1-seat]['tiles']:
     for tile in r:
      if isinstance(tile,dict) and tile.get('animal'):k=tile['animal'];row['day2_animals'][k]=row['day2_animals'].get(k,0)+1
  if trace:
   p=Path(trace);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gzip.compress(json.dumps(dict(result=row,days=days,actions=actions,brief=brief),separators=(',',':')).encode()));row['trace']=str(p)
 except Exception:row['runtime_error']=traceback.format_exc()
 finally:
  if agent:agent.close()
  if pub:pub.close()
 row['seconds']=time.monotonic()-t
 return row

