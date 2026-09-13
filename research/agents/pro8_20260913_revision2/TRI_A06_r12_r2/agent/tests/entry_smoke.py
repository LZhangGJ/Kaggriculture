"""Runtime entry, isolation, matching-native and explicit-failure checks."""
from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--agent',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('root_smoke',a.agent);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
rows=[];last=None;t=time.monotonic()
for case in json.loads((a.feedback/'SELECTED_CASES.json').read_text()):
 trace=json.load(gzip.open(a.feedback/case['trace'],'rt'));m.reset();matched=0
 for i,obs in enumerate(trace['observations'][:48]):
  action=m.agent(copy.deepcopy(obs),trace['configuration']);assert action==trace['own_actions'][i],(case['id'],i);matched+=1
 rows.append({'case':case['id'],'root_module_agent_calls':48,'exact':matched});last=trace
m.reset();reference=m.agent(copy.deepcopy(last['observations'][0]),last['configuration']);m.reset();assert m.agent(copy.deepcopy(last['observations'][0]),last['configuration'])==reference
m.reset()
negative=[]
for name,change in [('bad_clock',lambda o:o.update(day=10)),('terminal_step',lambda o:o.update(step=719,day=29,hour=23)),('bad_inventory_count',lambda o:o['private'].update(inventories=[]))]:
 obs=copy.deepcopy(last['observations'][0]);change(obs);agent=m.create_agent()
 try:
  agent(obs,last['configuration'])
 except (ValueError,RuntimeError) as err:negative.append({'case':name,'exception':type(err).__name__,'text':str(err)})
 else:raise AssertionError('malformed observation silently accepted: '+name)
 finally:agent.close()
r={'cases':rows,'exact_calls':sum(x['exact'] for x in rows),'reset_determinism':True,'explicit_rejections':negative,'native_sha256':hashlib.sha256((a.agent.parent/'policy/a06.so').read_bytes()).hexdigest(),'seconds':time.monotonic()-t,'new_games':0}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('cases','explicit_rejections')}))
