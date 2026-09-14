from pathlib import Path
import json,gzip,ctypes,importlib.util,time,copy
w=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('diagnostic',w/'diagnostic_r3/main.py');root=importlib.util.module_from_spec(spec);spec.loader.exec_module(root)
allrows=[]
for file in sorted((w/'feedback/own_traces/candidate_r3').glob('*.gz')):
 tr=json.load(gzip.open(file,'rt'));agent=root.create_agent();lib=agent.lib;lib.td_supply_audit.argtypes=[ctypes.c_void_p];lib.td_supply_audit.restype=ctypes.c_char_p;rows=[];prev='null';t=time.monotonic();exact=0
 for s,obs in enumerate(tr['observations'][:719]):
  act=agent(copy.deepcopy(obs),tr['configuration'])
  if act!=tr['own_actions'][s]:raise AssertionError((file.name,s,'diagnostic changed decision'))
  exact+=1;row=lib.td_supply_audit(agent.handle).decode()
  if row!=prev and row!='null': rows.append(json.loads(row))
  prev=row
 agent.close();out={'case':tr['game_id'],'exact':exact,'seconds':time.monotonic()-t,'rows':rows,'scope':'Unchanged r3 entry replay; alternative quantity outcomes are conditional current-day projections, not counterfactual real futures.'}
 (w/'logs'/('diagnostic_'+tr['game_id']+'.json')).write_text(json.dumps(out,indent=2));allrows.append({k:v for k,v in out.items() if k!='rows'})
 print(out['case'],out['exact'],out['seconds'],flush=True)
 for r in rows:
  full=r['profiles'][-1];base=r['profiles'][0];print(' step',r['step'],'q',full['buy'],'cover',len(base['covered']),len(full['covered']), 'dcash',full['cash']-base['cash'],'dwork',full['work']-base['work'],flush=True)
  for x in r['profiles'][1:-1]:
   if set(x['covered'])>=set(full['covered']):print('  smaller',x['buy'],'samecover','cash_vs_full',x['cash']-full['cash'],'yield', [a-b for a,b in zip(x['yield'],full['yield'])], 'stock',[a-b for a,b in zip(x['stock'],full['stock'])],flush=True)
(w/'logs/diagnostic_summary.json').write_text(json.dumps(allrows,indent=2))
