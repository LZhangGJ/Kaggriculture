"""Official bounded 48/72-transition conditional forks. No actual opponent or saved future.

Warm both production roots only to a verified shared historical prefix; then
replace the missing opponent private state with a declared synthetic scenario.
Use actual successive official observations after the intervention. Endpoints
are conditional cash diagnostics, NEVER match outcomes or recovered game cash.
"""
from pathlib import Path
import copy,gzip,json,importlib.util,sys,time,hashlib,collections,resource,ctypes
w=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(w/'feedback/referee'))
from cpu_runtime import LocalGame,AttrDict

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fork(obs,config,seed,scenario):
 g=LocalGame(seed);g.configuration=AttrDict(copy.deepcopy(config));g.configuration.seed=None;g.info['seed']=seed;g.t=obs['step']
 farms=copy.deepcopy(obs['farms']);market=copy.deepcopy(obs['market']);town=copy.deepcopy(obs['town']);seat=obs['player']
 for i in (0,1):
  private=copy.deepcopy(obs['private']) if i==seat else g.engine._new_private()
  if i!=seat:
   private['inventories']=[{} for _ in range(len(farms[i]['hands'])+1)]
   if scenario=='supply_input_pressure':private['shed'].update(STRAWBERRY=40,TOMATO=40)
  g.state[i].observation=AttrDict(step=g.t,remainingOverageTime=60,farms=farms,market=market,town=town,day=g.t//24,hour=g.t%24,player=i,private=private)
 return g

def rival_action(g,seat,scenario,offset):
 farm=g.state[0].observation.farms[seat];orders=[]
 if scenario=='supply_input_pressure':
  if offset==0:orders=[['BUY_PRODUCT','FERTILIZER',20]]
  elif offset%4==0 and offset<40:orders=[['SELL','STRAWBERRY',5],['SELL','TOMATO',5]]
 return {'farmer':['PASS'],'hands':[['PASS'] for _ in farm['hands']],'market':orders}

def run(case,variant,scenario,start,seed,ticks=48,output_group="bounded",intervention=False):
 tr=json.loads(gzip.decompress((w/f'feedback/own_traces/candidate_r3/{case}.json.gz').read_bytes()));root=load((w/'research/dayfork_agent/main.py') if intervention and variant=='candidate' else ((w/'candidate' if (w/'candidate/main.py').is_file() else w)/'main.py' if variant=='candidate' else w/'parent_r3'/'main.py'),f'{variant}_{case}_{scenario}');t0=time.perf_counter()
 gate=None
 if intervention and variant=='candidate':
  lib=ctypes.CDLL(str(w/'research/dayfork_agent/policy/a06.so'));gate=lib.td_r3_test_gate;gate.argtypes=[ctypes.c_int];gate.restype=ctypes.c_int;assert gate(0)==0
 for k,obs in enumerate(tr['observations'][:start]):
  a=root.agent(copy.deepcopy(obs),copy.deepcopy(tr['configuration']));assert a==tr['own_actions'][k],(variant,case,k,'prefix mismatch')
 if gate is not None:assert gate(1)==0
 initial=copy.deepcopy(tr['observations'][start]);seat=initial['player'];other=1-seat
 if scenario=='no_finance':
  initial['farms'][seat]['money']=0
  initial['private']['shed']={k:0 for k in initial['private']['shed']};initial['private']['seeds']={k:0 for k in initial['private']['seeds']}
 elif scenario=='expensive_fertilizer':
  initial['market']['inventory']['FERTILIZER']=0
  # price uses the exact official function, installed below after fork.
 g=fork(initial,tr['configuration'],seed,scenario)
 if scenario=='expensive_fertilizer':
  g.state[0].observation.market['prices']['FERTILIZER']=g.engine.market_price('FERTILIZER',0)
 e=g.engine;events=[];actions_log=[];commits=[];production=[];calls=[];flow=collections.Counter();quant=collections.Counter();fert_success=0;fert_failed=0
 orig_commit=e._commit_unit
 def commit(op,item,price,farm,private,market,shed_capacity=100):
  who=next(i for i,f in enumerate(g.state[0].observation.farms) if f is farm);before=farm['money'];old=private['shed'].get(item,0)
  ok=orig_commit(op,item,price,farm,private,market,shed_capacity)
  commits.append({'step':g.t,'player':who,'op':op,'item':item,'price':price,'ok':ok,'money_delta':farm['money']-before,'shed_delta':private['shed'].get(item,0)-old})
  if who==seat and ok:flow[f'{op}:{item}']+=farm['money']-before;quant[f'{op}:{item}']+=1
  return ok
 e._commit_unit=commit
 orig_hire=e._do_hire
 def hire(farm,private,board_size,mult=1):
  who=next(i for i,f in enumerate(g.state[0].observation.farms) if f is farm);before=farm['money'];n=len(farm['hands']);out=orig_hire(farm,private,board_size,mult)
  if who==seat:flow['HIRE']+=farm['money']-before;quant['HIRE']+=len(farm['hands'])-n
  return out
 e._do_hire=hire
 orig_land=e._do_buy_land
 def buy_land(farm,board_size):
  who=next(i for i,f in enumerate(g.state[0].observation.farms) if f is farm);before=farm['money'];out=orig_land(farm,board_size)
  if who==seat:flow['BUY_LAND']+=farm['money']-before;quant['BUY_LAND']+=farm['money']<before
  return out
 e._do_buy_land=buy_land
 orig_unit=e._apply_unit_action
 def unit(farm,private,idx,action,board_size,day,turns_per_day,shed_capacity=100):
  nonlocal fert_success,fert_failed
  who=next(i for i,f in enumerate(g.state[0].observation.farms) if f is farm)
  pos=copy.deepcopy(farm['farmer'] if idx==0 else farm['hands'][idx-1]) if idx<=len(farm['hands']) else None
  if who==seat and pos:
   x,y=pos;before=copy.deepcopy(farm['tiles'][y][x]);inv=copy.deepcopy(private['inventories'][idx])
  out=orig_unit(farm,private,idx,action,board_size,day,turns_per_day,shed_capacity)
  if who==seat and pos:
   after=copy.deepcopy(farm['tiles'][y][x]);afterinv=private['inventories'][idx]
   if action and action[0] in ('FERTILIZE','WATER','HARVEST','DROP','PICKUP','PLACE'):
    event={'step':g.t,'unit':idx,'pos':pos,'action':copy.deepcopy(action),'tile_before':before,'tile_after':after,'inventory_before':inv,'inventory_after':copy.deepcopy(afterinv)};events.append(event)
    if action[0]=='FERTILIZE':
     ok=inv.get('FERTILIZER',0)-afterinv.get('FERTILIZER',0)==1;fert_success+=ok;fert_failed+=not ok
  return out
 e._apply_unit_action=unit
 orig_daily=e._daily_refresh_plants
 def daily(farm,day,turns):
  who=next(i for i,f in enumerate(g.state[0].observation.farms) if f is farm);old=copy.deepcopy(farm['tiles']);out=orig_daily(farm,day,turns)
  if who==seat:
   for y,row in enumerate(old):
    for x,tile in enumerate(row):
     if not isinstance(tile,dict) or tile.get('kind')!='PLANT' or tile.get('crop') not in ('STRAWBERRY','TOMATO'):continue
     cd=e.CROPS[tile['crop']];ds=day+1-tile['planted_day']-cd['first_yield_day']
     if ds<0 or ds%cd['interval'] or ds//cd['interval']>=cd['max_yield']:continue
     after=farm['tiles'][y][x];alive=isinstance(after,dict) and after.get('kind')=='PLANT'
     production.append({'day':day,'pos':[x,y],'crop':tile['crop'],'birth':tile['planted_day'],'watered':tile['watered_today'],'fertilizer_until':tile.get('fertilized_until_day',-1),'bonus':bool(alive and tile['watered_today'] and tile.get('fertilized_until_day',-1)>=day),'alive':alive,'produced':max(0,after.get('yield_units',0)-tile['yield_units']) if alive else 0})
  return out
 e._daily_refresh_plants=daily
 observed_initial=g.observation(seat)
 for offset in range(ticks):
  obs=g.observation(seat);tt=time.perf_counter();a=root.agent(obs,copy.deepcopy(g.configuration));calls.append(time.perf_counter()-tt)
  assert len(a['market'])<=10 and len(a['hands'])==len(obs['farms'][seat]['hands'])
  rr=rival_action(g,other,scenario,offset);joint=[None,None];joint[seat]=a;joint[other]=rr
  g.advance(joint)
  after=g.observation(seat)
  assert after['farms'][seat]['money']>=0
  assert sum(after['private']['shed'].values())<=100
  assert all(v>=0 for v in after['private']['shed'].values())
  assert all(v>=0 for inv in after['private']['inventories'] for v in inv.values())
  actions_log.append({'step':obs['step'],'own_observation':obs,'own_action':a,'synthetic_rival_action':rr,'own_cash_after':after['farms'][seat]['money'],'rival_cash_after':after['farms'][other]['money']})
 end=g.observation(seat);root.reset();assert g.done and g.t==719;assert abs(sum(flow.values())-(end["farms"][seat]["money"]-observed_initial["farms"][seat]["money"]))<1e-6
 result={'case':case,'variant':variant,'scenario':scenario,'synthetic_seed':seed,'start_step':start,'steps':ticks,'terminal_reached':g.done,'cash_ledger_reconciled':True,'new_complete_games':0,'scope':('Research-only switch enables the exact R3 preparation formula at a specified current observation after an exact parent prefix; this is NOT the natural R3 historical prefix. ' if intervention else '')+'Official bounded conditional continuation; missing rival private state is explicit synthetic input. No saved future, actual opponent response, match result or recovered historical cash is claimed.','intervention_from_parent_state':intervention,'prefix_matched':start,'own_cash_start':observed_initial['farms'][seat]['money'],'own_cash_end':end['farms'][seat]['money'],'rival_cash_start':observed_initial['farms'][other]['money'],'rival_cash_end':end['farms'][other]['money'],'flow':dict(flow),'units':dict(quant),'fertilize_success':fert_success,'fertilize_failed':fert_failed,'production_total':sum(p['produced'] for p in production),'production_bonus_ticks':sum(p['bonus'] for p in production),'ongoing_production_dead':sum(not p['alive'] for p in production),'elapsed_seconds':time.perf_counter()-t0,'max_call_seconds':max(calls),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 raw={'result':result,'initial_own_observation':observed_initial,'initial_synthetic_rival_private':({'shed':{'STRAWBERRY':40,'TOMATO':40}} if scenario=='supply_input_pressure' else {}),'configuration':dict(g.configuration),'actions':actions_log,'unit_events':events,'production_events':production,'market_commits':commits,'final_own_observation':end}
 out=w/'logs'/output_group/f'{case}_{variant}_{scenario}.json.gz';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(gzip.compress(json.dumps(raw,separators=(',',':')).encode()))
 print(json.dumps(result),flush=True);return result

