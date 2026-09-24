"""Rule-only public-supply sale admission on top of the A06 R12 executor.
Unit actions, input reservations and all original purchases are retained.
No opponent program, seed, replay, learned weights or hidden inventory.
"""
from pathlib import Path
import importlib.util, json
ROOT=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_sale_native_entry',ROOT/'native_entry.py')
_native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_native)
CASH=('CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL')
PRODUCT={'GOOSE':'EGG','COW':'MILK','SHEEP':'WOOL'}
FIRST={'WHEAT':2,'CARROT':2,'TOMATO':8,'STRAWBERRY':10,'MELON':10}
DEPOTS=((4,4),(4,5),(5,4),(5,5))
def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def near(a):return min(dist(a,b) for b in DEPOTS)
class SaleAgent:
 def __init__(self,binary_path=None):
  self.base=_native.create_agent(binary_path);self.config=self.base.config.copy();self.config.update(r13_sale_mode=2,r13_sale_from=0,r13_sale_order=0);self.reset()
 def reset(self):
  self.base.config={k:v for k,v in self.config.items() if not k.startswith('r13_sale_')};self.base.reset();self.changed=0;self.added=0
 def close(self):self.base.close()
 def debug(self):
  d=self.base.debug();d.update(r13_sale_changes=self.changed,r13_sale_added=self.added,r13_sale_mode=self.config['r13_sale_mode']);return d
 @staticmethod
 def after_units(obs,a):
  """Conservative available cash commodities after known own unit phase."""
  own=obs['farms'][int(obs['player'])];p=obs['private'];shed=dict(p['shed']);bags=p['inventories'];positions=[own['farmer']]+own['hands'];units=[a.get('farmer',['PASS'])]+a.get('hands',[])
  for u,cmd in enumerate(units):
   if u>=len(bags) or not cmd or tuple(positions[u]) not in DEPOTS:continue
   bag=bags[u];op=cmd[0]
   if op=='PICKUP' and len(cmd)>=3:
    item=cmd[1];shed[item]=max(0,shed.get(item,0)-max(0,int(cmd[2])))
   elif op=='DROP':
    # The referee accepts each commodity subject to the shared warehouse cap.
    for item,n in bag.items():
     qty=min(max(0,int(n)),max(0,100-sum(shed.values())))
     shed[item]=shed.get(item,0)+qty
   elif op=='PLACE' and len(cmd)>=3 and cmd[1] in CASH:
    item=cmd[1];qty=min(max(0,int(cmd[2])),bag.get(item,0),max(0,100-sum(shed.values())))
    shed[item]=shed.get(item,0)+qty
  return shed
 def __call__(self,obs,configuration=None):
  a=self.base(obs,configuration);mode=int(self.config['r13_sale_mode']);step=int(obs['step'])
  if mode<=0 or step<int(self.config['r13_sale_from']):return a
  raw=[list(x) for x in a.get('market',[])];shed=self.after_units(obs,a);public={i:0 for i in CASH};arrival={i:999 for i in CASH};own=obs['farms'][int(obs['player'])];opp=obs['farms'][1-int(obs['player'])];day=step//24
  for r,row in enumerate(opp['tiles']):
   for c,t in enumerate(row):
    if not isinstance(t,dict):continue
    item=PRODUCT.get(t.get('animal'),t.get('crop'));qty=max(0,int(t.get('yield_units',0)))
    if item not in public or qty<=0:continue
    if item in FIRST and day-int(t.get('planted_day',0))<FIRST[item]:continue
    public[item]+=qty
    eta=min(dist(pos,(c,r)) for pos in [opp['farmer']]+opp['hands'])+1+near((c,r))+1
    arrival[item]=min(arrival[item],eta)
  existing={x[1] for x in raw if len(x)>=3 and x[0]=='SELL'}
  sell={}
  for item in CASH:
   q=int(shed.get(item,0));enable=mode==1 or mode==2 and public[item]>0 or mode==3 and arrival[item]<=4 or mode==4 and arrival[item]<=8
   if enable and q>0:sell[item]=q
  if not sell:return a
  changed=False
  for order in raw:
   if len(order)>=3 and order[0]=='SELL' and order[1] in sell and order[2]<sell[order[1]]:
    self.added+=sell[order[1]]-int(order[2]);order[2]=sell[order[1]];changed=True
  extras=[['SELL',i,q] for i,q in sell.items() if i not in existing]
  prices=obs['market']['prices']
  def key(x):
   i=x[1];return (public.get(i,0)*prices.get(i,0) if self.config['r13_sale_order'] else prices.get(i,0),int(x[2]))
  extras.sort(key=key,reverse=True);extras=extras[:max(0,10-len(raw))]
  if extras:self.added+=sum(x[2] for x in extras);changed=True
  if not changed:return a
  # New surplus sales precede all purchases; original relative order retained.
  self.changed+=1
  return dict(a,market=extras+raw)
def create_agent(binary_path=None):return SaleAgent(binary_path)
_seats={}
def reset():
 for p in _seats.values():p.close()
 _seats.clear()
def agent(observation,configuration=None):
 seat=int(observation['player'])
 if seat not in _seats:_seats[seat]=create_agent()
 return _seats[seat](observation,configuration)
