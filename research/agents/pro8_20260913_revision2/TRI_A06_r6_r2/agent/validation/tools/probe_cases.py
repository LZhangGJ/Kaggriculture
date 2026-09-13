"""Actual root-entry observations only; no opponent, environment, or game results."""
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--agent',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-exact',action='store_true');a=p.parse_args()
root=a.agent.resolve();inp=a.input.resolve();dest=a.out.resolve();dest.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('root_entry_test',root);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
summary=[];started=time.perf_counter()
for f in sorted((inp/'own_traces').glob('*.gz')):
 trace=json.load(gzip.open(f,'rt'));assert trace['configuration']['seed'] is None
 m.reset();rows=[];daily=[];first=None;tick=time.perf_counter()
 for i,ob in enumerate(trace['observations'][:-1]):
  t=time.perf_counter();act=m.agent(copy.deepcopy(ob),copy.deepcopy(trace['configuration']));dt=time.perf_counter()-t
  assert isinstance(act,dict) and all(k in act for k in ('farmer','hands','market'));assert len(act['market'])<=10;assert len(act['hands'])==len(ob['farms'][trace['seat']]['hands']);json.dumps(act,allow_nan=False)
  same=act==trace['own_actions'][i]
  if not same and first is None:first=i
  rows.append({'step':i,'same_as_parent':same,'seconds':dt,'action':act})
  if i%24==0:
   agent=m._seats[trace['seat']];d=agent.debug();daily.append({'step':i,'debug':d})
 m.reset()
 (dest/(trace['game_id']+'.json')).write_text(json.dumps({'scope':'Saved own-visible observations. After first divergence the future inputs are stale. No closed-loop game.','first_divergence':first,'new_games':0,'rows':rows,'daily':daily},separators=(',',':'))+'\n')
 matches=sum(x['same_as_parent'] for x in rows);r={'id':trace['game_id'],'calls':len(rows),'exact_parent_actions':matches,'first_divergence':first,'seconds':time.perf_counter()-tick,'max_call_seconds':max(x['seconds'] for x in rows),'new_games':0,'parent_cash_margin':next(c['row']['margin'] for c in json.loads((inp/'SELECTED_CASES.json').read_text()) if c['id']==trace['game_id'])};summary.append(r);print(json.dumps(r),flush=True)
res={'agent_sha256':hashlib.sha256(root.read_bytes()).hexdigest(),'native_sha256':hashlib.sha256((root.parent/'policy/a06.so').read_bytes()).hexdigest(),'cases':summary,'seconds':time.perf_counter()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_games':0};(dest/'SUMMARY.json').write_text(json.dumps(res,indent=2)+'\n')
if a.require_exact:assert all(x['exact_parent_actions']==719 for x in summary)
