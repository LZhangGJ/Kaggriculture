import importlib.util,json,gzip,ctypes,sys,time,copy,resource,hashlib
from pathlib import Path
b=Path('/mnt/data/r3_work');p=b/'parent/main.py';spec=importlib.util.spec_from_file_location('r3_diag_entry',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
for case in json.loads((b/'feedback/SELECTED_CASES.json').read_text()):
 trace=json.loads(gzip.decompress((b/'feedback'/case['trace']).read_bytes()));a=m.create_agent(b/'diagnostic.so');a.lib.td_delivery_audit.argtypes=[ctypes.c_void_p];a.lib.td_delivery_audit.restype=ctypes.c_char_p
 start=time.perf_counter();exact=0;records=[]
 for t,o in enumerate(trace['observations'][:719]):
  action=a(copy.deepcopy(o),trace['configuration']);exact+=action==trace['own_actions'][t]
  r=json.loads(a.lib.td_delivery_audit(a.handle).decode());r.update(actual_step=t,action=action,same_parent=action==trace['own_actions'][t]);records.append(r)
 a.close();out=b/'logs'/('audit_'+case['id']+'.json.gz');out.write_bytes(gzip.compress(json.dumps(records,separators=(',',':')).encode(),mtime=0))
 row={'case':case['id'],'calls':len(records),'exact_parent_actions':exact,'seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_games':0};rows.append(row);print(json.dumps(row),flush=True)
 assert exact==719
(b/'logs/diagnostic_summary.json').write_text(json.dumps(rows,indent=2))
