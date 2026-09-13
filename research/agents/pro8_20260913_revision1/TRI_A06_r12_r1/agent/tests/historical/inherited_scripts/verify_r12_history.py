"""Observation-only r12 tests; no new games and no realized suffix after divergence.
--feedback is the extracted, already-completed r6 diagnostic archive.
Use matching parent ea075584 and candidate paths; all outputs outside runtimes.
"""
from pathlib import Path
import argparse,ctypes,gzip,hashlib,importlib.util,json,resource,time
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--feedback',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--case',action='append');args=ap.parse_args()
R=Path(__file__).resolve().parents[1];D=args.out;D.mkdir(parents=True,exist_ok=False)
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module(R/'policy/agent.py','r12codec');p=module(args.parent/'policy/agent.py','parent11codec')
cfg=json.loads((R/'policy/config.json').read_text());pcfg=json.loads((args.parent/'policy/config.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,x):
 s=json.dumps(x,ensure_ascii=False,indent=2).encode();path.write_bytes(gzip.compress(s,mtime=0) if path.suffix=='.gz' else s+b'\n')
def funcs(a):
 for fn,ret,argv in [('td_clone',ctypes.c_void_p,[ctypes.c_void_p]),('td_r11_enable',ctypes.c_int,[ctypes.c_void_p,ctypes.c_int]),('td_r12_enable',ctypes.c_int,[ctypes.c_void_p,ctypes.c_int]),('td_contract_json',ctypes.c_char_p,[ctypes.c_void_p]),('td_crop_clock_json',ctypes.c_char_p,[ctypes.c_void_p])]:
  if hasattr(a.lib,fn):f=getattr(a.lib,fn);f.restype=ret;f.argtypes=argv

def clone(a,on):
 z=m.Agent(cfg,R/'policy/a06.so');z.close();z.handle=a.lib.td_clone(a.handle);assert z.handle;z.last=a.last;z.seat=a.seat;funcs(z);assert z.lib.td_r12_enable(z.handle,on)==0;assert z.lib.td_r11_enable(z.handle,1)==0;return z

def valid(a,o):
 assert set(a)=={'farmer','hands','market'};assert len(a['hands'])==len(o['farms'][o['player']]['hands']);assert len(a['market'])<=10
 for atom in [a['farmer'],*a['hands'],*a['market']]:
  assert isinstance(atom,list) and atom[0] in m._OPS
  if len(atom)==3:assert type(atom[2])==int and atom[2]>0

def strip(d):
 # Preserve complete choices separately; summary should not grow with plan keys.
 z={k:v for k,v in d.items() if k!='last_search'};z['selected_id']=d.get('last_search',{}).get('selected_id');return z
cases=args.case or [p.stem.replace('.replay.json','') for p in sorted((args.feedback/'replays').glob('*.replay.json.gz'))]
summary={'scope':'offline legal historical observations; selected clones stop after one call; zero new matches','native':sha(R/'policy/a06.so'),'parent_native':sha(args.parent/'policy/a06.so'),'cases':[],'new_games':0}
for name in cases:
 f=args.feedback/'replays'/(name+'.replay.json.gz');r=json.loads(gzip.decompress(f.read_bytes()));seat=r['result']['seat'];assert len(r['actions'])==719
 historical=m.Agent(cfg|{'a06_r11_recovery':0,'a06_r12_calendar':0},R/'policy/a06.so');funcs(historical)
 parent=p.Agent(pcfg,args.parent/'policy/a06.so');funcs(parent)
 parentoff=m.Agent(cfg|{'a06_r12_calendar':0},R/'policy/a06.so');funcs(parentoff)
 new=m.Agent(cfg,R/'policy/a06.so');funcs(new)
 parent_active=True;new_active=True;records=[];probes=[];first_r12=None;first_parent=None;off_parent_matches=0;historical_matches=0
 # All eight diagnostics included. Per-day legal prefix probes, no seed routing.
 probe_steps={24*d for d in (0,3,6,9,12,15,18,21,24,27)}
 start=time.perf_counter()
 for t in range(719):
  o=r['steps'][t][seat]['observation'];assert o['step']==t
  if t in (0,288):
   extra=dict(o,seed=999,opponent_id='FORBIDDEN',opponent_private={'cash':999999},future_shops=['FORBIDDEN']);assert list(m._pack(o))==list(m._pack(extra))
  clones=[clone(historical,on) for on in (0,1)] if t in probe_steps else []
  a=historical(o);valid(a,o);assert a==r['actions'][t][seat],f'r6-equivalent mismatch {name} {t}';historical_matches+=1
  b=None
  if parent_active:
   b=parent(o);c=parentoff(o);valid(b,o);valid(c,o);assert b==c,f'new OFF != exact parent at {name}:{t}';off_parent_matches+=1
   if b!=a:
    first_parent={'step':t,'historical_action':a,'parent_action':b};parent_active=False
  if new_active:
   z=new(o);valid(z,o)
   # Compare r12 with parent only while all previous real observations match.
   if b is not None and z!=b:
    first_r12={'step':t,'same_historical_prefix':t,'observation':o,'parent_action':b,'r12_action':z,'parent_debug':parent.debug(),'r12_debug':new.debug()};save(D/(name+'_first_r12.json.gz'),first_r12)
   if z!=a or not parent_active:new_active=False
  if not parent_active:parent.close();parentoff.close()
  if not new_active:new.close()
  if clones:
   pair=[]
   for on,z in enumerate(clones):
    before=json.loads(z.lib.td_contract_json(z.handle));a2=z(o);valid(a2,o);dbg=z.debug()
    pair.append({'enabled':on,'action':a2,'debug':dbg,'before_contract':before,'after_contract':json.loads(z.lib.td_contract_json(z.handle)),'conditional_rival_flow':json.loads(z.lib.td_crop_clock_json(z.handle)).get('live_forecast')});z.close()
   probes.append({'step':t,'observation':o,'pair':pair,'different':pair[0]['action']!=pair[1]['action'],'both_cloned_from_r6_exact_prefix':True})
  records.append({'step':t,'historical_action':a,'cash_observed':o['farms'][seat]['money'],'workers_observed':1+len(o['farms'][seat]['hands'])})
  if t%144==0:print(name,t,'parentmatches',off_parent_matches,'first',None if first_r12 is None else first_r12['step'],flush=True)
 for z in (historical,parent,parentoff,new):z.close()
 save(D/(name+'_control_actions.json.gz'),records);save(D/(name+'_probes.json.gz'),probes)
 entry={'id':name,'recorded_cash_not_new':r['result']['cash'],'source_replay_sha256':sha(f),'r6_equivalent_actions':historical_matches,'OFF_matches_exact_parent_on_valid_prefix':off_parent_matches,'parent_first_difference_from_r6':first_parent,'r12_first_difference_from_parent':None if first_r12 is None else first_r12['step'],'one_step_paired_probes':len(probes),'changed_probes':sum(z['different'] for z in probes),'wall_seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 summary['cases'].append(entry);save(D/'SUMMARY.json',summary);print(json.dumps(entry),flush=True)
