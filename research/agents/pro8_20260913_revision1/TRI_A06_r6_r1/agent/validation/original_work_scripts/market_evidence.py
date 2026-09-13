"""Current-state primitive evidence; historical audit, not counterfactual games."""
import ctypes as C,json,pathlib,gzip,sys,collections,math
R=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(R/'input/referee'));from cpu_runtime import load_engine,LocalGame,compare_frame
engine=load_engine();lib=C.CDLL(str(R/'diagnostics/market_primitive_probe.so'))
items=list(engine.PRODUCTS)
for name in ('legacy','floor_safe','exact'):
 f=getattr(lib,'probe_'+name);f.argtypes=[C.c_int,C.c_double,C.c_double,C.POINTER(C.c_double)];f.restype=C.c_double
lib.probe_saturation.argtypes=[C.c_int];lib.probe_saturation.restype=C.c_double
lib.probe_price.argtypes=[C.c_int,C.c_double];lib.probe_price.restype=C.c_double

def model(name,i,stock,q):
 out=C.c_double();cash=getattr(lib,'probe_'+name)(i,stock,q,C.byref(out));return {'cash':cash,'stock':out.value}
def official(item,stock,q):
 farm={'money':0};private={'shed':{item:q}};market={'inventory':{item:stock}};cash=0
 for _ in range(q):
  p=engine.market_price(item,market['inventory'][item]);assert engine._commit_unit('SELL',item,p,farm,private,market)
 return {'cash':farm['money'],'stock':market['inventory'][item]}
examples=[];positive=0;legacy_fail=0;negative=0;fractional=0
for i,item in enumerate(items):
 sat=int(lib.probe_saturation(i));assert engine.market_price(item,sat)==1 and engine.market_price(item,sat-1)>1
 for dist in range(-12,13):
  stock=sat-dist
  for q in [1,2,3,5,8,13,21,34,55,89,100]:
   old=model('legacy',i,stock,q);fix=model('floor_safe',i,stock,q)
   if stock+q>sat:
    positive+=1;truth=official(item,stock,q);assert fix==truth,(item,stock,q,fix,truth)
    if old!=truth:legacy_fail+=1
    if dist==2 and q==13:examples.append({'item':item,'saturation':sat,'stock':stock,'quantity':q,'legacy':old,'floor_safe':fix,'official':truth})
   else:negative+=1;assert old==fix
 for stock in [9700,10000,sat-15,sat,sat+15]:
  for q in [-100,-31,-10,-3,-2,-1,0]:
   negative+=1;assert model('legacy',i,stock,q)==model('floor_safe',i,stock,q)
 for dist in [-3.75,-.25,0,.25,.75,1.5,3.5]:
  for q in [.25,.75,1.5,3.75,15.5]:
   stock=sat-dist
   if stock+q>sat:
    fractional+=1;assert model('floor_safe',i,stock,q)==model('exact',i,stock,q)
summary={'scope':'One-sided conditional primitive only, not full-game counterfactual cash. Official _commit_unit oracle; only positive floor-crossing branches changed.', 'positive_official_cases':positive,'legacy_failures_in_positive_cases':legacy_fail,'negative_identical_cases':negative,'fractional_relaxation_cases':fractional,'examples':examples,'historical_cases':[]}
for case in json.loads((R/'input/SELECTED_CASES.json').read_text()):
 name=case['id'];seat=case['row']['seat'];replay=json.load(gzip.open(R/'input/replays'/f'{name}.json.gz','rt'));g=LocalGame(replay['result']['seed'],engine);events=[];floorunits=collections.Counter();rec=[]
 original_commit=engine._commit_unit
 def logged(op,item,price,farm,private,market,shed_capacity=100):
  # Identity only for offline accounting of the already-recorded historical game.
  sid=0 if farm is g.state[0].observation.farms[0] else 1
  before=market['inventory'].get(item)
  ok=original_commit(op,item,price,farm,private,market,shed_capacity)
  if ok and op=='SELL':
   events.append((g.t,sid,item,before,price))
   if sid==seat and price==1:floorunits[item]+=1
  return ok
 engine._commit_unit=logged
 try:
  for t in range(719):g.advance(replay['actions'][t]);compare_frame(g,replay['steps'][t+1])
 finally:engine._commit_unit=original_commit
 batches=collections.defaultdict(list)
 for t,sid,item,stock,price in events:
  if sid==seat:batches[t,item].append((stock,price))
 for (t,item),ev in batches.items():
  i=items.index(item);stock=ev[0][0];q=len(ev)
  if q>0 and stock+q>lib.probe_saturation(i):
   old=model('legacy',i,stock,q);fix=model('floor_safe',i,stock,q);truth=official(item,stock,q);assert fix==truth
   rec.append({'step':t,'day':t//24,'item':item,'stock_before_first_successful_unit':stock,'own_successful_quantity':q,'actual_own_revenue_with_rival_lockstep':sum(x[1] for x in ev),'conditional_legacy':old,'conditional_floor_safe':fix,'conditional_official':truth,'legacy_stock_error':old['stock']-truth['stock']})
 row={'case':name,'scope':'Historical state/order support. Conditional one-sided batch comparison does NOT replace actual simultaneous-order revenue or imply recoverable profit.','verified_transitions':719,'own_floor_sale_units':dict(floorunits),'floor_crossing_successful_batches':rec}
 summary['historical_cases'].append(row);print(name,'floorunits',dict(floorunits),'batches',len(rec),'bad',sum(e['legacy_stock_error']!=0 for e in rec),flush=True)
 (R/'diagnostics/market_floor_evidence.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['examples','historical_cases']}),flush=True)
