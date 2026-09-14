from pathlib import Path
import argparse,copy,gzip,hashlib,importlib.util,json,time,resource
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--exact',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sp=importlib.util.spec_from_file_location('r4_actual_root',a.root/'main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
start=time.perf_counter();results=[]
for path in sorted((a.feedback/'own_traces').glob('*.gz')):
 t=json.load(gzip.open(path,'rt'));m.reset();records=[];eq=0
 for k,ob in enumerate(t['observations'][:719]):
  t0=time.perf_counter();act=m.agent(copy.deepcopy(ob),t['configuration']);same=act==t['own_actions'][k];records.append({'step':k,'same':same,'action':act,'seconds':time.perf_counter()-t0,'source_if_different':None if same else t['own_actions'][k]})
  if not same:break
  eq+=1
 m.reset();row={'case':t['game_id'],'calls':len(records),'same':eq,'first_difference':None if eq==719 else k,'max_call_seconds':max(x['seconds'] for x in records)};results.append(row)
 with gzip.open(a.out/path.name,'wt') as f:json.dump({'scope':'Actual root entry on common saved prefix only; no saved future after first differing action.','rows':records},f,separators=(',',':'))
 print(row,flush=True)
 if a.exact:assert eq==719,row
(a.out/'SUMMARY.json').write_text(json.dumps({'new_games':0,'results':results,'seconds':time.perf_counter()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'native_sha256':hashlib.sha256((a.root/'policy/a06.so').read_bytes()).hexdigest()},indent=2))
