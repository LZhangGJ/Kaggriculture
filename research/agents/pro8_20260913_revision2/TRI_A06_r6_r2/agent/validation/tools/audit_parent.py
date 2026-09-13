"""Replay legal own observations; collect own plan diagnostics, not game outcomes."""
import argparse,ctypes,gzip,hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--binary',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
root=args.root.resolve();inp=args.input.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('production_main',root/'main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
allrows=[];start=time.perf_counter()
for f in sorted((inp/'own_traces').glob('*.gz')):
 trace=json.load(gzip.open(f,'rt'));tick=time.perf_counter();a=m.create_agent(binary_path=args.binary);rows=[];nmatch=0;first=None
 if args.binary:
  a.lib.td_land_audit.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t];a.lib.td_land_audit.restype=ctypes.c_char_p
 a.lib.td_contract_json.argtypes=[ctypes.c_void_p];a.lib.td_contract_json.restype=ctypes.c_char_p
 for i,o in enumerate(trace['observations'][:-1]):
  act=a(o,trace['configuration']);same=act==trace['own_actions'][i];nmatch+=same
  if not same and first is None:first=i
  if o['hour']==0 or any(x[0]=='BUY_LAND' for x in act['market']):
   rec={'step':i,'same':same,'action':act,'debug':a.debug(),'contract':json.loads(a.lib.td_contract_json(a.handle).decode())}
   if args.binary:
    pack=m.codec._pack(o);rec['land_audit']=json.loads(a.lib.td_land_audit(a.handle,pack,len(pack)).decode())
   rows.append(rec)
 a.close();(out/(trace['game_id']+'.json')).write_text(json.dumps(rows,separators=(',',':'))+'\n')
 summary={'id':trace['game_id'],'calls':719,'exact':nmatch,'first_divergence':first,'seconds':time.perf_counter()-tick,'new_games':0};allrows.append(summary);print(json.dumps(summary),flush=True)
res={'scope':'Historical legal own-observation diagnostics. No counterfactual games or wins. All candidate future observations after divergence are stale.','cases':allrows,'seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(out/'SUMMARY.json').write_text(json.dumps(res,indent=2)+'\n');assert all(x['exact']==719 for x in allrows)
