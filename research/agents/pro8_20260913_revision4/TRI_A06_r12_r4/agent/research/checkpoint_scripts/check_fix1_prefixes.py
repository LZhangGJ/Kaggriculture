from pathlib import Path
import importlib.util,json,gzip,time,copy
w=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fix1root',w/'candidate/main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
summary=[]
for path in sorted((w/'feedback/own_traces/candidate_r3').glob('*.gz')):
 t=json.load(gzip.open(path,'rt'));a=m.create_agent();rows=[];first=None;clock=time.monotonic()
 for step,o in enumerate(t['observations'][:719]):
  action=a(copy.deepcopy(o),t['configuration']);same=action==t['own_actions'][step]
  dbg=a.debug();audit={k:v for k,v in dbg.items() if k.startswith('r3_fix1')}
  if step%24==0 and step>=648 or not same:rows.append({'step':step,'same':same,'fix1':action,'r3':t['own_actions'][step],'audit':audit})
  if not same:first=step;break
 r={'case':t['game_id'],'matched':first if first is not None else 719,'calls':step+1,'first_divergence':first,'seconds':time.monotonic()-clock,'last_debug':audit,'rows':rows,'new_games':0,'scope':'Candidate compared to its actual r3 source trajectory only until first divergence.'};a.close();summary.append(r)
 print(r['case'],r['matched'],r['first_divergence'],r['seconds'],json.dumps(audit),flush=True)
(w/'logs/fix1_prefix_summary.json').write_text(json.dumps(summary,indent=2))
