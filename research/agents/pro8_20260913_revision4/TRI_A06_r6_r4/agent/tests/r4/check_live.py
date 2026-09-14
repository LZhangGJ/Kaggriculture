"""Warm production main.py entry, native clone, official own-only fresh futures."""
from pathlib import Path
import argparse,ctypes,gzip,importlib.util,json,time,sys,copy
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--lib',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sp=importlib.util.spec_from_file_location('live_r4_entry',a.root/'main.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
sys.path.insert(0,str(a.root/'tests'));from official_prefix import load_referee
from conditional import verify
runtime,engine=load_referee(a.feedback);summ=[]
for path in sorted((a.feedback/'own_traces').glob('*.gz')):
 t=json.load(gzip.open(path,'rt'));name=t['game_id'];indices={'aurax_reactive_v1_395620922_seat0':438,'submission_56149565_563140739_seat1':483,'thomas_955_v2_20452605_seat1':509,'submission_56149565_1075824552_seat0':600};start=indices[name]
 m.reset();ag=m.create_agent(binary_path=a.lib);ag.lib.td_r4_enable.argtypes=[ctypes.c_int];ag.lib.td_r4_enable(1);fn=ag.lib.td_r4_branch;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_int];fn.restype=ctypes.c_char_p
 for k,ob in enumerate(t['observations'][:start]):
  act=m.agent(copy.deepcopy(ob),t['configuration']);chk=ag(copy.deepcopy(ob),t['configuration']);assert act==chk==t['own_actions'][k],('not common prefix',name,k)
 real=m._seats[t['seat']] if 'seat' in t else m._seats[t['observations'][0]['player']]
 real.lib.td_clone.argtypes=[ctypes.c_void_p];real.lib.td_clone.restype=ctypes.c_void_p
 clone=m.create_agent();clone.close();clone.handle=real.lib.td_clone(real.handle);clone.seat=real.seat;clone.last=real.last
 ob=t['observations'][start];z=m.codec._pack(ob);raw=fn(ag.handle,z,len(z),1).decode();assert not raw.startswith('ERROR'),raw;branch=json.loads(raw)
 v=verify(t,ob,branch,m.codec,runtime,engine,actual=m.agent,clone=clone,capture=True);branch['official']=v
 with gzip.open(a.out/(name+'.json.gz'),'wt') as f:json.dump(branch,f,separators=(',',':'))
 row={'case':name,'shared_prefix':start,'fresh_root_calls':v['counts'].get('real_root_matches',0),'clone_calls':v['counts'].get('clone_matches',0),'official_prefixes':v['prefixes'],'start':start,'conditional_final_cash':branch['final_cash'],'overflow':branch['overflow'],'wave_added':branch['wave_added'],'handoffs':branch['handoffs'],'counts':v['counts']};print(row,flush=True);summ.append(row);ag.close();clone.close();m.reset()
(a.out/'SUMMARY.json').write_text(json.dumps({'new_games':0,'branches':summ,'scope':'warm shared prefixes then exclusively fresh official own-only observations; no opposing orders or future replay'},indent=2))
