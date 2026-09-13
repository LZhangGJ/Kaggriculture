from pathlib import Path
import sys,importlib.util,ctypes,json,gzip,copy,time,argparse
W=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('parent_main',W/'parent/main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--case');args=p.parse_args()
cases=json.loads((W/'input/SELECTED_CASES.json').read_text());summaries=[]
for case in cases:
 if args.case and case['id']!=args.case:continue
 t0=time.perf_counter();tr=json.loads(gzip.decompress((W/'input'/case['trace']).read_bytes()));a=m.create_agent();a.lib.td_contract_json.argtypes=[ctypes.c_void_p];a.lib.td_contract_json.restype=ctypes.c_char_p;rows=[];matched=0
 for step,o in enumerate(tr['observations'][:719]):
  act=a(copy.deepcopy(o),copy.deepcopy(tr['configuration']));assert act==tr['own_actions'][step],(case['id'],step);matched+=1
  if step%24==0 or (case['id']=='submission_56149565_852484341_seat0' and 210<=step<=239):
   rows.append({'step':step,'action':act,'debug':a.debug(),'contract':json.loads(a.lib.td_contract_json(a.handle))})
 a.close()
 with gzip.open(W/'logs'/f"parent_context_{case['id']}.json.gz",'wt') as z:json.dump(rows,z)
 summary={'case':case['id'],'calls':matched,'matches':matched,'seconds':time.perf_counter()-t0,'new_games':0};summaries.append(summary);print(summary,flush=True)
(W/'logs'/f"parent_full_probe_{args.case or 'all'}.json").write_text(json.dumps(summaries,indent=2))
