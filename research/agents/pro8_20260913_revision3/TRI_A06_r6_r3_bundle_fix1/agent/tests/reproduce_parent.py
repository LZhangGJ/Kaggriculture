from pathlib import Path
import importlib.util,gzip,json,ctypes,time,resource
import argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'build/parent_reproduction');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
W=ROOT
sp=importlib.util.spec_from_file_location('baseline',W/'parent/main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
summ=[]
for f in sorted((W/'evidence/r3_feedback/own_traces').glob('*.gz')):
 t=json.load(gzip.open(f,'rt'));ag=m.create_agent();records=[];tick=time.perf_counter();eq=0
 fn=ag.lib.td_contract_json;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_char_p
 for step,ob in enumerate(t['observations'][:-1]):
  act=ag(ob,t['configuration']);assert act==t['own_actions'][step],(f.name,step);eq+=1
  if step%24==0 or step in (217,218,219,313,314,315):
   records.append({'step':step,'cash':ob['farms'][ob['player']]['money'],'hands':len(ob['farms'][ob['player']]['hands']),'action':act,'debug':ag.debug(),'contract':json.loads(fn(ag.handle))})
 ag.close();out={'scope':'Exact parent reproduction only; no new games','calls':eq,'seconds':time.perf_counter()-tick,'records':records};
 with gzip.open(args.out/('baseline_'+f.name),'wt') as z:json.dump(out,z)
 summ.append({'case':t['game_id'],'calls':eq,'seconds':out['seconds']});print(summ[-1],flush=True)
(args.out/'parent_full_reproduction.json').write_text(json.dumps({'cases':summ,'calls':sum(x['calls'] for x in summ),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
