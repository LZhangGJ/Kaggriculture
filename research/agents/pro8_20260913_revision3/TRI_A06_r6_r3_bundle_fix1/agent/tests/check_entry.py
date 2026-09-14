from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,resource,time,statistics
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=R/'parent');p.add_argument('--out',type=Path,default=R/'build/entry_checks.json');a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('entry_candidate',R/'main.py');parent=load('entry_parent',a.parent/'main.py');results=[];timing=[];calls=0
for f in sorted((R/'evidence/r3_feedback/own_traces').glob('*.gz'))+sorted((R/'evidence/own_visible/own_traces').glob('*.gz')):
 t=json.load(gzip.open(f,'rt'));pa=parent.create_agent();ca=m.create_agent();record={'game_id':t['game_id'],'equal_prefix':0,'first_difference':None,'scope':'Stop at first parent/candidate action difference; no observed suffix used for a counterfactual.'}
 for step,ob in enumerate(t['observations'][:-1]):
  pp=pa(ob,t['configuration']);tt=time.perf_counter();cc=ca(ob,t['configuration']);timing.append(time.perf_counter()-tt);calls+=1
  if pp!=cc:
   record.update(first_difference=step,parent_action=pp,candidate_action=cc);break
  record['equal_prefix']+=1
 pa.close();ca.close();results.append(record)
# Independent saved-state cold-entry, both seats, through genuine module-level
# entry plus reset. No connected replay and no fabricated future observations.
independent=[]
for f in sorted((R/'evidence/r3_feedback/own_traces').glob('*.gz')):
 t=json.load(gzip.open(f,'rt'))
 for idx in (0,48,120,216,312,432,576,696,718):
  ob=t['observations'][idx];m.reset();tt=time.perf_counter();cc=m.agent(ob,t['configuration']);timing.append(time.perf_counter()-tt);calls+=1
  aa=m.create_agent();dup=aa(ob,t['configuration']);aa.close();assert cc==dup
  assert len(cc['hands'])==len(ob['farms'][ob['player']]['hands']) and len(cc['market'])<=10
  independent.append({'game_id':t['game_id'],'step':idx,'action':cc,'pass':True})
m.reset();ordered=sorted(timing);out={'scope':'Root entry, shape, isolated context and reset checks; common prefixes stop at divergence. No new games.','new_games':0,'parent_native_sha256':hashlib.sha256((a.parent/'policy/a06.so').read_bytes()).hexdigest(),'candidate_native_sha256':hashlib.sha256((R/'policy/a06.so').read_bytes()).hexdigest(),'prefix_results':results,'independent_cold_entry_checks':independent,'timed_calls':len(timing),'inference_seconds':{'min':min(timing),'median':statistics.median(timing),'p95':ordered[int(.95*(len(ordered)-1))],'max':max(timing),'sum':sum(timing)},'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('prefix_results','independent_cold_entry_checks')}));print('first differences',[(x['game_id'],x['first_difference']) for x in results])
