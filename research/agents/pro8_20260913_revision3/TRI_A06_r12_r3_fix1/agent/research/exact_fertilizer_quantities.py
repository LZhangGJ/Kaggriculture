from pathlib import Path
import gzip,json,copy,sys,collections
w=Path(__file__).resolve().parents[1];sys.path.insert(0,str(w/'feedback/referee'))
from cpu_runtime import load_engine
e=load_engine();result=[]
for version in ('parent_r2','candidate_r3'):
 tr=json.load(gzip.open(w/f'feedback/own_traces/{version}/submission_56149565_805057947_seat0.json.gz','rt'));seat=tr['seat']
 for step in range(360,719):
  a=tr['own_actions'][step];orders=a['market']
  buys=[x for x in orders if x[:2]==['BUY_PRODUCT','FERTILIZER']]
  if not buys:continue
  assert not any(x[:2]==['SELL','FERTILIZER'] for x in orders),'netting must be handled explicitly'
  o=tr['observations'][step];n=tr['observations'][step+1];p=copy.deepcopy(o['private']);f=copy.deepcopy(o['farms'][seat]);units=[a['farmer']]+a['hands'];dem=collections.Counter(x[1] for x in units if x and x[0]=='PLANT');blocked={k for k,v in dem.items() if v>p['seeds'].get(k,0)}
  for u,act in enumerate(units):
   if act[0]=='PLANT' and act[1] in blocked:act=['PASS']
   e._apply_unit_action(f,p,u,act,10,step//24,24,100)
  q=n['private']['shed'].get('FERTILIZER',0)-p['shed'].get('FERTILIZER',0)
  req=sum(x[2] for x in buys);assert 0<=q<=req
  result.append({'version':version,'step':step,'requested':req,'observed_arrived':q,'post_own_unit_shed_fertilizer':p['shed'].get('FERTILIZER',0),'next_observed_shed_fertilizer':n['private']['shed'].get('FERTILIZER',0),'actual_whole_step_cash_delta':n['farms'][seat]['money']-o['farms'][seat]['money'],'other_orders':orders,'price_scope':'All inputs are actual own-visible. Quantity is observed net fertilizer arrival; individual monetary attribution still requires separating simultaneous other orders.'})
out=w/'logs/exact_fertilizer_arrivals.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
