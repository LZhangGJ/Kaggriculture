from pathlib import Path
import importlib.util,json,gzip,hashlib,time,resource,ctypes,collections,sys
R=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('parentmain',R/'input/agent/main.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
summary={'scope':'supplied historical development trajectories; zero new games','cases':[]};start=time.perf_counter()
for cp in json.load(open(R/'input/SELECTED_CASES.json')):
 f=R/'input/replays'/(cp['id']+'.json.gz');r=json.loads(gzip.decompress(f.read_bytes()));seat=r['result']['seat'];a=m.create_agent();fn=a.lib.td_contract_json;fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_char_p
 rows=[];harvests=[];decays=[];st=time.perf_counter()
 for t in range(719):
  o=r['steps'][t][seat]['observation'];act=a(o);assert act==r['actions'][t][seat],(cp['id'],t)
  own=o['farms'][seat];nexto=r['steps'][t+1][seat]['observation'];farm=own['tiles'];nf=nexto['farms'][seat]['tiles']
  actors=[own['farmer'],*own['hands']]
  for pos,atom in zip(actors,[act['farmer'],*act['hands']]):
   x,y=pos;tile=farm[y][x]
   if atom[0]=='HARVEST' and isinstance(tile,dict) and tile.get('crop'):
    harvests.append({'step':t,'pos':y*10+x,'kind':tile['crop'],'age':o['day']-tile['planted_day'],'yield':tile['yield_units'],'cash':own['money']})
  for y in range(10):
   for x in range(10):
    tile=farm[y][x];nt=nf[y][x]
    if isinstance(tile,dict) and tile.get('kind')=='PLANT' and isinstance(nt,dict) and nt.get('kind')=='WEED':decays.append({'step':t,'pos':y*10+x,'before':tile})
  if t%24==0:
   dbg=a.debug();ct=json.loads(fn(a.handle));rows.append({'step':t,'own_cash':own['money'],'opponent_cash':o['farms'][1-seat]['money'],'own_private':o['private'],'action':act,'debug':dbg,'contract':ct})
  if t==718:final=a.debug()
 a.close();out={'id':cp['id'],'seconds':time.perf_counter()-st,'calls':719,'match_all':True,'harvests':harvests,'decays':decays,'days':rows,'final_debug':final}
 (R/'logs'/(cp['id']+'_parent_trace.json.gz')).write_bytes(gzip.compress(json.dumps(out).encode(),mtime=0))
 row={'id':cp['id'],'scope':'exact historical action reproduction','calls':719,'seconds':out['seconds'],'crop_deaths':len(decays),'finite_harvests':dict(collections.Counter((x['kind']+':'+str(x['age'])) for x in harvests if x['kind'] in ('WHEAT','CARROT','MELON'))),'r12_calls':final['r12_calls'],'r12_changes':final['r12_changes'],'r12_existing_changes':final['r12_existing_changes']}
 summary['cases'].append(row);print(json.dumps(row),flush=True)
 del r,rows
summary.update(seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(R/'logs/historical_reproduction.json').write_text(json.dumps(summary,indent=2))
