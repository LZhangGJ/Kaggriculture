#!/usr/bin/env python3
"""Same saved-state, short OFFICIAL conditional branches. Not full matches.
One native version per process. Opponent private cleared before any branch.
Only pre-start observations warm policy state; all later inputs are generated
by the official interpreter from actions, never read from the saved future.
"""
from pathlib import Path
import argparse,copy,ctypes,gzip,hashlib,importlib.util,json,sys,time,traceback
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--referee',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists();assert not a.out.resolve().is_relative_to(a.root.resolve())
 sys.path.insert(0,str(a.referee));from cpu_runtime import AttrDict,load_engine
 def attr(x):
  if isinstance(x,dict):return AttrDict({k:attr(v) for k,v in x.items()})
  if isinstance(x,list):return [attr(v) for v in x]
  return x
 case=json.loads(a.case.read_text());data=json.loads(gzip.decompress(a.fixture.read_bytes()));seat=data['result']['seat'];other=1-seat;start=case['step'];ticks=case['ticks'];assert ticks<=5 and (start+ticks-1)//24==start//24
 spec=importlib.util.spec_from_file_location('one_version',a.root/'main.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);base=mod.create_agent(a.binary)
 lib=base.lib;lib.td_clone.argtypes=[ctypes.c_void_p];lib.td_clone.restype=ctypes.c_void_p
 before_equal=0;prefix=[];result={'scope':'short official conditional branches; no new or complete matches; future rival inventory/sales explicitly hypothetical','case':case,'fixture_sha256':sha(a.fixture),'binary_sha256':sha(a.binary),'referee_sha256':sha(a.referee/'official/kaggriculture.py'),'new_matches':0,'new_seeds':0,'branches':[],'complete':False}
 try:
  for step in range(start):
   own=base(data['steps'][step][seat]['observation']);equal=own==data['actions'][step][seat];before_equal+=equal
   if not equal:raise AssertionError('Not a common original prefix; do not treat changed-history warmup as on-policy: '+str(step))
   prefix.append(own)
  result['common_prefix_calls']=start;result['common_prefix_actions_equal']=before_equal;result['common_prefix_action_sha256']=hashlib.sha256(canonical(prefix).encode()).hexdigest()
  for label,slot,traffic in [('no_additional_sales',0,False),('traffic_queue0',0,True),('traffic_queue3',3,True),('traffic_queue9',9,True)]:
   agent=copy.copy(base);agent.handle=lib.td_clone(base.handle);assert agent.handle
   engine=load_engine()
   class Branch:pass
   env=Branch();env.configuration=attr(copy.deepcopy(data['configuration']));env.info={'seed':0};env.done=False;env.t=start;env.state=attr(copy.deepcopy(data['steps'][start]));events=[];rows=[];post_units={}
   # Actual opponent-private contents are erased, not consulted for traffic.
   env.state[other].observation.private={'shed':{},'seeds':{},'inventories':[{} for _ in range(1+len(env.state[0].observation.farms[other]['hands']))]}
   env.state[other].observation.farms=env.state[0].observation.farms
   env.state[other].observation.market=env.state[0].observation.market
   env.state[other].observation.town=env.state[0].observation.town
   initial=copy.deepcopy(env.state[seat].observation)
   def wrap_money(name):
    original=getattr(engine,name)
    def call(*args,**kw):
     farm=args[3] if name=='_commit_unit' else args[0]
     player=next(s for s in (0,1) if farm is env.state[0].observation.farms[s]);before=farm['money'];ans=original(*args,**kw)
     events.append({'step':env.t,'player':player,'op':args[0] if name=='_commit_unit' else name,'item':args[1] if name=='_commit_unit' else None,'cash_before':before,'cash_after':farm['money'],'delta':farm['money']-before,'success':bool(ans) if name=='_commit_unit' else before!=farm['money']})
     return ans
    setattr(engine,name,call)
   for name in ['_commit_unit','_do_hire','_do_buy_land']:wrap_money(name)
   original_market=engine._process_market
   def market(state,environment):
    post_units['own_private']=copy.deepcopy(state[seat].observation.private)
    post_units['own_farm']=copy.deepcopy(state[0].observation.farms[seat])
    post_units['market']=copy.deepcopy(state[0].observation.market)
    return original_market(state,environment)
   engine._process_market=market
   try:
    for t in range(ticks):
     step=start+t;before=copy.deepcopy(env.state[seat].observation);clock=time.perf_counter();own=agent(before);seconds=time.perf_counter()-clock;debug=agent.debug()
     if t==0:
      assert own.get('farmer')==data['actions'][start][seat].get('farmer') and own.get('hands')==data['actions'][start][seat].get('hands'), 'Sale mechanism changed unit prefix at common decision'
     assert len(own.get('hands',[]))<=len(before.farms[seat]['hands'])
     rival=[];injected={}
     for item,quantities in case['traffic'].items():
      q=quantities[t] if traffic else 0
      if q:
       env.state[other].observation.private['shed'][item]=env.state[other].observation.private['shed'].get(item,0)+q
       injected[item]=q;rival.append(['SELL',item,q])
     assert len(rival)<=1, 'These isolated window branches declare only one product traffic stream'
     rival=[['PASS']]*slot+rival if rival else []
     env.state[seat].action=copy.deepcopy(own);env.state[other].action={'farmer':['PASS'],'hands':[],'market':rival}
     for s in env.state:s.observation.step=step
     money=[f['money'] for f in env.state[0].observation.farms];event_at=len(events)
     engine.interpreter(env.state,env);env.t+=1
     for s in env.state:s.observation.step=env.t
     after=copy.deepcopy(env.state[seat].observation);residual=[]
     for s in (0,1):
      x=after.farms[s]['money']-money[s]-sum(e['delta'] for e in events[event_at:] if e['player']==s);assert x==0,x;residual.append(x)
     for m in own.get('market',[]):
      if m[0]=='SELL':assert m[2]<=post_units['own_private']['shed'].get(m[1],0), ('Not in actual post-unit warehouse',m)
     rows.append({'step':step,'before_own_legal_observation':before,'own_action':own,'rival_synthetic_orders':rival,'injected_synthetic_inventory':injected,'post_own_units_before_market':copy.deepcopy(post_units),'after_own_legal_observation':after,'seconds':seconds,'cash_identity_residual_both':residual,'debug':debug})
    end=rows[-1]['after_own_legal_observation'];own_delta=end.farms[seat]['money']-initial.farms[seat]['money'];rival_delta=end.farms[other]['money']-initial.farms[other]['money']
    result['branches'].append({'scenario':label,'ticks':ticks,'start_step':start,'end_step':start+ticks,'end_cash_both':[f['money'] for f in end.farms],'own_cash_change':own_delta,'rival_cash_change':rival_delta,'end_private':end.private,'cash_residual_zero':True,'actual_sell_stock_guards':True,'rows':rows,'events':events})
   finally:agent.close()
  result['complete']=True
 except Exception as e:
  result['error']=repr(e);result['traceback']=traceback.format_exc();raise
 finally:
  base.close();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(gzip.compress(json.dumps(result,separators=(',',':')).encode(),mtime=0))
  print(json.dumps({'case':case['id'],'binary':result['binary_sha256'],'complete':result['complete'],'branches':[{'scenario':x['scenario'],'ticks':x['ticks'],'own_cash_change':x['own_cash_change'],'rival_cash_change':x['rival_cash_change']} for x in result['branches']],'error':result.get('error')},indent=2))
if __name__=='__main__':main()
