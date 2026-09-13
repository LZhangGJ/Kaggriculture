"""Reproduce saved actions (not new games) and independently reconcile engine frames."""
import sys,importlib.util,pathlib,json,gzip,hashlib,time,ctypes,collections,datetime
R=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import LocalGame,load_engine,compare_frame
spec=importlib.util.spec_from_file_location('parent_audit_main',R/'input/agent/main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
engine=load_engine();rows=[];cases=json.loads((R/'input/SELECTED_CASES.json').read_text())
for case in cases:
 name=case['id'];seat=case['row']['seat'];p=R/'input/replays'/f'{name}.json.gz';r=json.load(gzip.open(p,'rt'));agent=m.create_agent();g=LocalGame(r['result']['seed'],engine)
 agent.lib.td_contract_json.argtypes=[ctypes.c_void_p];agent.lib.td_contract_json.restype=ctypes.c_char_p
 trace=R/'diagnostics'/f'parent_{name}.jsonl.gz';t0=time.perf_counter();times=[];daily=[];start=[s.observation.farms[s.observation.player]['money'] for s in g.state];day_start=list(start)
 with gzip.open(trace,'wt') as f:
  compare_frame(g,r['steps'][0])
  for t in range(719):
   obs=g.observation(seat);assert 'seed' not in obs
   ts=time.perf_counter();action=agent(obs,g.configuration);times.append(time.perf_counter()-ts)
   if action!=r['actions'][t][seat]:raise AssertionError(('parent action divergence',name,t,action,r['actions'][t][seat]))
   debug=agent.debug();contract=json.loads(agent.lib.td_contract_json(agent.handle).decode())
   if t%24:debug.pop('last_search',None)
   rec={'step':t,'action':action,'debug':debug,'contract':contract,'own_cash':obs['farms'][seat]['money'],'own_shed':obs['private']['shed']}
   f.write(json.dumps(rec,separators=(',',':'))+'\n')
   g.advance(r['actions'][t]);compare_frame(g,r['steps'][t+1])
   if (t+1)%24==0 or t==718:
    end=[s.observation.farms[s.observation.player]['money'] for s in g.state]
    daily.extend({'day':t//24,'seat':s,'cash_start':day_start[s],'cash_end':end[s],'net_cash_change':end[s]-day_start[s]} for s in (0,1));day_start=end
  assert g.done and g.t==719 and len(daily)==60
  audit=json.loads((R/'input/audits'/f'{name}.json').read_text())
  for d in daily:
   a=audit['players'][d['seat']]['days'][d['day']];assert a['cash_start']==d['cash_start'] and a['cash_end']==d['cash_end'];assert abs(sum(a['flow'].values())-d['net_cash_change'])<1e-8
 final=[s.reward for s in g.state];row={'case':name,'seat':seat,'scope':'historical reproduction; not new results','matched_actions':719,'verified_transitions':719,'reconciled_player_days':60,'final_cash':final,'wall_seconds':time.perf_counter()-t0,'agent_total_seconds':sum(times),'agent_max_seconds':max(times),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()};rows.append(row);agent.close();print(json.dumps(row),flush=True)
 (R/'diagnostics/parent_replay_audit.json').write_text(json.dumps({'scope':'Reproduced exact parent actions and official transitions against saved joint actions. All 7 are historical diagnostics, not win-rate sample.','rows':rows},indent=2))
