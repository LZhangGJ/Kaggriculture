"""Bounded own-route feasibility fixture with official unit/dawn transitions.

The parent context is reproduced through the observed step216. Its executor
and target/service contracts are frozen, not re-optimized. We change only the
preparation order arrangement, settle synthetic current-market stress, compile
from the actual resulting stocks/hands, and execute those prescribed routes
until this day's boundary. There are no opponents, random events, future saved
observations, future revenues, match outcomes or candidate closed-loop claims.
"""
from pathlib import Path
import argparse,collections,copy,ctypes,gzip,importlib.util,json,sys,time
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--audit-lib',type=Path,required=True);p.add_argument('--feedback',type=Path,required=True);p.add_argument('--unit-json',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
runtime=mod(a.feedback/'referee/cpu_runtime.py','fixed_runtime');e=runtime.load_engine();parent=mod(a.parent,'fixed_parent');codec=mod(a.parent.parent/'policy/agent.py','fixed_codec')
tr=json.load(gzip.open(a.feedback/'own_traces/submission_56149565_852484341_seat0.json.gz','rt'));obs=tr['observations'][216]
agent=parent.create_agent(binary_path=a.audit_lib);start=time.monotonic()
for step,o in enumerate(tr['observations'][:217]):assert agent(copy.deepcopy(o),tr['configuration'])==tr['own_actions'][step],('audit parent exact',step)
agent.lib.audit_fixed_routes.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t];agent.lib.audit_fixed_routes.restype=ctypes.c_char_p
unit=json.loads(a.unit_json.read_text());cases=[]

def settle_order(order,farm,priv,market):
 s=e._parse_order(order);op=s['type']
 if op=='HIRE':e._do_hire(farm,priv,10)
 elif op=='BUY_LAND':e._do_buy_land(farm,10)
 else:
  item=s['item']
  for _ in range(s['remaining']):
   if op in ('SELL','BUY_PRODUCT'):price=e.market_price(item,market['inventory'][item]-(op=='BUY_PRODUCT'),market.get('params'))
   elif op=='BUY_SEED':price=e.CROPS[item]['seed']
   elif op=='BUY_ANIMAL':price=e.ANIMALS[item]['cost']
   else:raise AssertionError(op)
   if not e._commit_unit(op,item,price,farm,priv,market,100):break
 e._refresh_prices(market)

def run(orders,shock):
 state=copy.deepcopy(obs);farm=state['farms'][state['player']];priv=state['private'];market=state['market']
 risky=[]
 for y,row in enumerate(farm['tiles']):
  for x,t in enumerate(row):
   if isinstance(t,dict) and t.get('crop') in ('TOMATO','STRAWBERRY') and t.get('consecutive_unwatered',0)>=1 and not t.get('watered_today',False):risky.append([x,y])
 for item,delta in shock.items():market['inventory'][item]+=delta
 e._refresh_prices(market);frames=[]
 for batch in range((len(orders)+9)//10):
  step=216+batch
  # Exactly the known first preparation action, then no credited setup labor.
  ua=tr['own_actions'][216]['farmer'] if batch==0 else ['PASS']
  e._apply_unit_action(farm,priv,0,ua,10,9,24,100)
  for order in orders[batch*10:(batch+1)*10]:settle_order(order,farm,priv,market)
  e._decay_plants(farm,step)
  frames.append({'step':step,'phase':'preparation','farmer':ua,'market':orders[batch*10:(batch+1)*10],'cash':farm['money'],'hands':len(farm['hands']),'shed':copy.deepcopy(priv['shed'])})
 state['step']=216+(len(orders)+9)//10;state['hour']=state['step']%24
 packed=codec._pack(state);raw=agent.lib.audit_fixed_routes(agent.handle,packed,len(packed)).decode();assert not raw.startswith('ERROR'),raw
 compiled=json.loads(raw);routes=compiled['routes'];assert len(routes)==1+len(farm['hands'])
 successful=collections.Counter();noop=[];max_route=max(map(len,routes));work_frames=24-state['hour']
 # compile() may omit infeasible jobs, never extend the day or invent inputs.
 assert max_route<=work_frames,(max_route,work_frames)
 for offset,step in enumerate(range(state['step'],240)):
  acts=[]
  for u,route in enumerate(routes):
   if offset>=len(route):acts.append(['PASS']);continue
   op,item,qty,target=route[offset];name=codec._OPS[op];action=[name]
   if item>=0:action += [codec._ITEMS[item],qty]
   pos=(farm['farmer'] if u==0 else farm['hands'][u-1]).copy();x,y=pos
   before=(copy.deepcopy(farm['tiles'][y][x]),copy.deepcopy(priv['inventories'][u]),copy.deepcopy(priv['shed']),copy.deepcopy(priv['seeds']),pos)
   e._apply_unit_action(farm,priv,u,action,10,9,24,100)
   afterpos=(farm['farmer'] if u==0 else farm['hands'][u-1]).copy()
   after=(farm['tiles'][y][x],priv['inventories'][u],priv['shed'],priv['seeds'],afterpos)
   if name!='PASS':
    if before!=after:successful[name]+=1
    else:noop.append({'step':step,'unit':u,'action':action,'position':pos,'target':target})
   assert all(q>=0 for q in priv['shed'].values()) and all(q>=0 for q in priv['seeds'].values())
   assert all(q>=0 for bag in priv['inventories'] for q in bag.values())
   assert sum(priv['shed'].values())<=100 and farm['money']>=0
   acts.append(action)
  e._decay_plants(farm,step)
  frames.append({'step':step,'phase':'fixed compiled route','units':acts,'cash':farm['money'],'shed':copy.deepcopy(priv['shed']),'inventories':copy.deepcopy(priv['inventories'])})
 water_risky=[p for p in risky if isinstance(farm['tiles'][p[1]][p[0]],dict) and farm['tiles'][p[1]][p[0]].get('watered_today')]
 e._daily_refresh_plants(farm,9,24)
 alive=[p for p in risky if isinstance(farm['tiles'][p[1]][p[0]],dict) and farm['tiles'][p[1]][p[0]].get('crop') in ('TOMATO','STRAWBERRY')]
 return {'hires':len(farm['hands']),'actual_cash_after_preparation':frames[1]['cash'],'at_risk_at_start':risky,'at_risk_watered':water_risky,'at_risk_alive_at_dawn':alive,'successful_actions':dict(successful),'no_effect_actions':noop,'compiled':compiled,'frames':frames,'conditional_only':True}

for delta in (0,5,10,20):
 shock={'MILK':delta,'FERTILIZER':delta//2};row={'synthetic_shock':shock}
 for label,key in [('original','original_orders'),('protected','protected_orders')]:row[label]=run(unit[key],shock)
 cases.append(row)
 print(json.dumps({'shock':shock,**{label:{'hands':row[label]['hires'],'watered_risky':len(row[label]['at_risk_watered']),'surviving_risky':len(row[label]['at_risk_alive_at_dawn']),'noop':len(row[label]['no_effect_actions']),'dropped_jobs':row[label]['compiled']['drop_count']} for label in ('original','protected')}}),flush=True)
agent.close()
result={'kind':'fixed own-plan route feasibility only; no new match, no recovered-profit claim','parent_exact_calls':217,'source_step':216,'synthetic_cases':cases,'seconds':time.monotonic()-start,'new_games':0,'limits':['No unit labor credited on the second preparation frame.','Parent investment/service intentions held fixed; full candidate may choose differently.','After preparation, only compiled unit routes are executed: no market selling, intraday optimization or random events.','No saved observation after source step216 used except the recorded first preparation action.','Dawn is a conditional deterministic unit test, not an observed candidate future.']}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
# Require actual execution, not only a claimed plan.
assert any(len(r['protected']['at_risk_watered'])>len(r['original']['at_risk_watered']) for r in cases),'no demonstrated route-level improvement'
