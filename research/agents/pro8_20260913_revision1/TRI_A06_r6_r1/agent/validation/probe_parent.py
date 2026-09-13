import importlib.util,json,gzip,time,resource,hashlib,pathlib,sys
R=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('probe_main',R/'input/agent/main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=[('soil_v219g_4145681841_seat0',0,72),('thomas_955_v2_4145681832_seat0',0,72),('submission_56149565_4145681830_seat0',0,72)];rows=[]
for name,seat,n in cases:
 r=json.load(gzip.open(R/'input/replays'/f'{name}.json.gz','rt'));agent=m.create_agent();times=[];matches=0
 for t in range(n):
  obs=r['steps'][t][seat]['observation'];assert obs['step']==t and 'seed' not in obs
  started=time.perf_counter();action=agent(obs,r['configuration']);times.append(time.perf_counter()-started)
  same=action==r['actions'][t][seat];matches+=same
  if not same:raise AssertionError((name,t,action,r['actions'][t][seat]))
 row={'case':name,'count':n,'exact_action_matches':matches,'total_seconds':sum(times),'max_seconds':max(times),'mean_seconds':sum(times)/n,'debug_final':agent.debug()};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='debug_final'}),flush=True);agent.close()
(R/'diagnostics/parent_probe.json').write_text(json.dumps({'scope':'Legal saved-observation probe, not new games. All actions matched recorded baseline prefixes. No counterfactual suffix use.','rows':rows,'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
