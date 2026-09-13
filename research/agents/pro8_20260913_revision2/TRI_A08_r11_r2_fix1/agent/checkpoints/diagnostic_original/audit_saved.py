"""Own-visible saved-state diagnostics only; never counterfactual games."""
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,pathlib,resource,time
p=argparse.ArgumentParser();p.add_argument('--trace',type=pathlib.Path,required=True);a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('codec',root/'parent_immutable/policy/agent.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
agent=mod.Agent(binary_path=root/'probes/diagnostic_parent.so',config=json.loads((root/'parent_immutable/policy/config.json').read_text()))
agent.lib.audit_services.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t];agent.lib.audit_services.restype=ctypes.c_char_p
tr=json.loads(gzip.decompress(a.trace.read_bytes()));start=time.perf_counter();rows=[];same=0;times=[];mismatch=[]
for i,o in enumerate(tr['observations'][:-1]):
 tick=time.perf_counter();action=agent(copy.deepcopy(o));times.append(time.perf_counter()-tick);same+=action==tr['own_actions'][i]
 if o['hour']==0:
  packed=mod._pack(o);d=json.loads(agent.lib.audit_services(agent.handle,packed,len(packed)));d['debug']=agent.debug();rows.append(d)
  for path in d['paths']:
   k=path['kind'];day=d['day'];pos=path['pos'];tile=o['farms'][o['player']]['tiles'][pos//10][pos%10]
   if k in (2,3) and path['path_kind']==k and tile['planted_day']<day and path['end']>day:
    forecast=path['path_f'][day][8]<-0.5;execution=11 in path['jobs']
    if forecast!=execution:mismatch.append(dict(step=i,day=day,pos=pos,kind=k,birth=path['birth'],forecast_fertilize=forecast,execution_fertilize=execution,path_labor=path['first_labor'],jobs=path['jobs'],current_yield=tile['yield_units'],fert=tile['fertilized_until_day']))
agent.close();out={'scope':'Frozen saved own observations; no new games','trace':a.trace.name,'trace_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),'calls':len(times),'exact_parent_actions':same,'seconds':time.perf_counter()-start,'max_action_seconds':max(times),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'fertilizer_mismatches':mismatch,'days':rows}
path=root/'analysis'/f'{a.trace.stem}.audit.json';path.write_text(json.dumps(out,indent=1)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='days'}))
