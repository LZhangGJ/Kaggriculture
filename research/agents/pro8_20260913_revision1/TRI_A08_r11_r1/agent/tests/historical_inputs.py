"""One native version per process; saved legal observations, NOT closed-loop games."""
from pathlib import Path
import argparse,importlib.util,hashlib,gzip,json,time,ctypes
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--limit',type=int,default=480);p.add_argument('--stop-first-difference',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sp=importlib.util.spec_from_file_location('isolated_history',a.root/'main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
d=json.loads(gzip.decompress(a.replay.read_bytes()));s=d['result']['seat'];agent=m.create_agent(a.binary);rows=[];complete=False
try:
 for step in range(min(a.limit,719)):
  before=d['steps'][step][s]['observation'];t=time.perf_counter();action=agent(before);elapsed=time.perf_counter()-t;debug=agent.debug()
  rows.append({'step':step,'source_legal_observation_sha256':hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'action':action,'historical_r8_action':d['actions'][step][s],'debug':debug,'seconds':elapsed})
  if a.stop_first_difference and action!=d['actions'][step][s]:break
 complete=True
finally:
 agent.close();a.out.parent.mkdir(parents=True,exist_ok=True)
 summary={'scope':'saved legal inputs only; after first different output the subsequent inputs remain historical, not a candidate trajectory','new_games':0,'stop_first_difference':a.stop_first_difference,'requested_limit':a.limit,'complete_requested_calls':complete,'id':d['result']['id'],'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(a.replay.read_bytes()).hexdigest(),'calls':len(rows),'match_historical_actions':sum(x['action']==x['historical_r8_action'] for x in rows),'first_historical_difference':next((x['step'] for x in rows if x['action']!=x['historical_r8_action']),None),'last_paid_selected':rows[-1]['debug'].get('paid_selected',0) if rows else None,'last_sale_changed':rows[-1]['debug'].get('sale_schedule_changed',0) if rows else None,'max_seconds':max((x['seconds'] for x in rows),default=0)}
 a.out.write_bytes(gzip.compress(json.dumps({'summary':summary,'rows':rows},separators=(',',':')).encode(),mtime=0));print(json.dumps(summary),flush=True)
