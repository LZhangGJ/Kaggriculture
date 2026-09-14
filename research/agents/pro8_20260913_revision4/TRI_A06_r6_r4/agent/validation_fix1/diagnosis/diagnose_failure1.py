import pathlib,gzip,json,importlib.util,ctypes,time,collections,copy
B=pathlib.Path('/mnt/data/fix1_work');R=B/'bundle';F=B/'feedback'
s=importlib.util.spec_from_file_location('actual_main',R/'main.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
name='submission_56149565_201542742_seat0'
records={};summary={}
for v,lib in [('candidate_r3',R/'policy/a06.so'),('parent_r2',R/'parent/policy/a06.so')]:
 t=json.load(gzip.open(F/'own_traces'/v/(name+'.json.gz'),'rt'));ag=m.create_agent(binary_path=lib);rows=[];started=time.perf_counter()
 ag.lib.td_contract_json.argtypes=[ctypes.c_void_p];ag.lib.td_contract_json.restype=ctypes.c_char_p
 for i,o in enumerate(t['observations'][:-1]):
  a=ag(o,t['configuration']);assert a==t['own_actions'][i],(v,i,'action mismatch')
  debug=json.loads(ag.debug());contract=json.loads(ag.lib.td_contract_json(ag.handle))
  row={'step':i,'cash':o['farms'][0]['money'],'wheat_shed':o['private']['shed']['WHEAT'],'wheat_cargo':sum(iv.get('WHEAT',0) for iv in o['private']['inventories']),'debug':debug,'contract':contract};rows.append(row)
 ag.close();records[v]=rows;summary[v]={'calls':len(rows),'exact':len(rows),'seconds':time.perf_counter()-started}
 with gzip.open(B/'logs'/(v+'_actual_reproduction.json.gz'),'wt') as f:json.dump(rows,f)
 print(v,summary[v],flush=True)
 for d in range(30):
  x=rows[d*24];z=rows[min(718,d*24+22)];D=x['debug'];E=z['debug'];print(d,'cash',x['cash'],'need/feed',D['daily_need_wheat'],D['feed_stock_target'],'shed',x['wheat_shed'],'hires',D.get('labor_target'),'endcash',z['cash'],'lchecks/offers/jobs',E.get('labor_checks'),E.get('labor_offers'),E.get('labor_jobs'),'selected',D.get('last_search'),flush=True)
(B/'logs'/'actual_reproduction_summary.json').write_text(json.dumps(summary,indent=2))
