"""Artificial terminal-route positive/negative fixtures, checked by official rules.
No full game and no claim these artificial farms are historical observations.
"""
from pathlib import Path
import argparse,copy,ctypes,gzip,importlib.util,json,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--lib',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sp=importlib.util.spec_from_file_location('terminal_r4_entry',a.root/'main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);sys.path.insert(0,str(a.root/'tests'));from official_prefix import load_referee
from conditional import verify
rt,e=load_referee(a.feedback);trace=json.load(gzip.open(next((a.feedback/'own_traces').glob('*.gz')),'rt'));ag=m.create_agent(binary_path=a.lib);fn=ag.lib.td_r4_terminal_branch;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_int];fn.restype=ctypes.c_char_p
results=[]
for step,floor in [(710,False),(715,False),(716,False),(718,False),(710,True)]:
 ob=copy.deepcopy(trace['observations'][step]);ob.update(step=step,day=29,hour=step%24,player=0)
 for f in ob['farms']:
  f.update(money=1000,farmer=[4,4],hands=[],hires_today=0,tiles=[[None for _ in range(10)] for _ in range(10)],unlocked_quadrants=['NW','NE','SW','SE'])
 ob['farms'][0]['tiles'][4][3]={'kind':'PLANT','crop':'TOMATO','planted_day':10,'yield_units':4,'watered_today':True,'fertilized_until_day':-1,'max_lifespan_step':-1,'consecutive_unwatered':0}
 ob['private']={'shed':{i:0 for i in m.codec._ITEMS},'seeds':{i:0 for i in m.codec._ITEMS[:5]},'inventories':[{'MILK':3}]};ob['town']['unlocked_shops']=[];ob['market']['inventory']={i:10000 for i in m.codec._ITEMS[:9]};ob['market']['prices']=dict(zip(m.codec._ITEMS[:9],[25,35,60,120,250,50,160,200,100]))
 if floor:ob['market']['inventory']['TOMATO']=20000;ob['market']['prices']['TOMATO']=1
 pair=[]
 for mode in (0,1):
  z=m.codec._pack(ob);raw=fn(ag.handle,z,len(z),mode).decode();assert not raw.startswith('ERROR'),raw;branch=json.loads(raw);branch['official']=verify(trace,ob,branch,m.codec,rt,e,capture=True);branch['initial_synthetic_observation']=ob
  with gzip.open(a.out/f'terminal_{step}_floor{int(floor)}_mode{mode}.json.gz','wt') as f:json.dump(branch,f,separators=(',',':'))
  pair.append(branch)
 p0,p1=pair;should_accept=step<=715 and not floor
 assert bool(p1['terminal_added'])==should_accept,(step,floor,p1['terminal_added'])
 if should_accept:assert p1['final_cash']>p0['final_cash'] and p1['remaining_bag_tomato']==0 and p0['remaining_bag_tomato']>0
 row={'step':step,'at_floor':floor,'parent_conditional_cash':p0['final_cash'],'candidate_conditional_cash':p1['final_cash'],'parent_stranded_tomato':p0['remaining_bag_tomato'],'candidate_stranded_tomato':p1['remaining_bag_tomato'],'terminal_added':p1['terminal_added'],'official_prefixes':sum(len(x['ticks']) for x in pair),'pass':True};results.append(row);print(row,flush=True)
ag.close();(a.out/'SUMMARY.json').write_text(json.dumps({'new_games':0,'synthetic_pairs':results},indent=2))
