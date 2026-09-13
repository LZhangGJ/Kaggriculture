"""Stop each historical replay at the first changed action; never use its suffix."""
import sys,importlib.util,pathlib,json,gzip,time,hashlib
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import LocalGame,load_engine,compare_frame
spec=importlib.util.spec_from_file_location('revised_prefix_root',R.parent/'main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
engine=load_engine();results=[]
for case in json.loads((R/'input/SELECTED_CASES.json').read_text()):
 name=case['id'];seat=case['row']['seat'];r=json.load(gzip.open(R/'input/replays'/f'{name}.json.gz','rt'));g=LocalGame(r['result']['seed'],engine);agent=m.create_agent();ts=time.perf_counter();row={'case':name,'seat':seat,'matched_prefix_steps':0,'first_divergence':None}
 try:
  for t in range(719):
   obs=g.observation(seat);action=agent(obs,g.configuration)
   if action!=r['actions'][t][seat]:
    d=agent.debug();search=d.get('last_search',{})
    row['first_divergence']={'step':t,'day':obs['day'],'parent_action':r['actions'][t][seat],'candidate_action':action,'candidate_selected_family':search.get('selected_id'),'current_own_cash':obs['farms'][seat]['money']};break
   g.advance(r['actions'][t]);compare_frame(g,r['steps'][t+1]);row['matched_prefix_steps']+=1
 finally:agent.close()
 row['wall_seconds']=time.perf_counter()-ts;row['saved_future_used_after_divergence']=False;results.append(row);print(json.dumps(row),flush=True)
 (R/'diagnostics/prefix_comparison.json').write_text(json.dumps({'scope':'Legal equal-action prefix test only. Stops BEFORE applying any divergent action; no revised terminal cash or win-rate claimed.','candidate_native_sha256':hashlib.sha256((R.parent/'policy/a06.so').read_bytes()).hexdigest(),'rows':results},indent=2))
