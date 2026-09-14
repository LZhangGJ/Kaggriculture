"""Replay own observations for entry/regression checks; this runs zero counterfactual games."""
import argparse,copy,gzip,hashlib,importlib.util,json,os,random,resource,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--agent',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--limit',type=int,default=48);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-exact',action='store_true');a=p.parse_args()
a.out=a.out.resolve()
assert 1<=a.limit<=719 and not a.out.exists()
trace=json.loads(gzip.decompress(a.trace.read_bytes()));assert trace['configuration']['seed'] is None
path=a.agent.resolve();sys.path.insert(0,str(path.parent));os.chdir(path.parent);random.seed(0)
spec=importlib.util.spec_from_file_location('own_trace_entry',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
tick=time.perf_counter();rows=[]
try:
 for t,obs in enumerate(trace['observations'][:a.limit]):
  started=time.perf_counter();action=mod.agent(copy.deepcopy(obs),copy.deepcopy(trace['configuration']))
  assert isinstance(action,dict) and all(k in action for k in ('farmer','hands','market'))
  json.dumps(action,allow_nan=False)
  rows.append({'step':t,'seconds':time.perf_counter()-started,'same_as_parent':action==trace['own_actions'][t],'action':action})
finally:
 if hasattr(mod,'reset'):mod.reset()
result={'scope':'Saved own-observation entry check, not a closed-loop game or win-rate test','new_games':0,'calls':len(rows),'exact_parent_actions':sum(x['same_as_parent'] for x in rows),'seconds':time.perf_counter()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rows':rows}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if a.require_exact:assert result['exact_parent_actions']==a.limit
