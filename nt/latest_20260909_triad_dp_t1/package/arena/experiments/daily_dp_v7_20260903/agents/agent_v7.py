from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

CROPS={
 'WHEAT':{'seed':10,'first':2,'max_day':4,'interval':0,'max_yield':6,'ongoing':False,'harvest_age':4},
 'CARROT':{'seed':20,'first':2,'max_day':3,'interval':0,'max_yield':4,'ongoing':False,'harvest_age':3},
 'TOMATO':{'seed':50,'first':8,'max_day':8,'interval':1,'max_yield':4,'ongoing':True},
 'STRAWBERRY':{'seed':100,'first':10,'max_day':10,'interval':2,'max_yield':4,'ongoing':True},
 'MELON':{'seed':80,'first':10,'max_day':12,'interval':0,'max_yield':6,'ongoing':False,'harvest_age':10},
}
ANIMALS={
 'GOOSE':{'cost':300,'structure':'COOP','first':4,'interval':1,'max_held':4,'product':'EGG'},
 'COW':{'cost':400,'structure':'PASTURE','first':8,'interval':2,'max_held':6,'product':'MILK'},
 'SHEEP':{'cost':500,'structure':'PASTURE','first':6,'interval':3,'max_held':6,'product':'WOOL'},
}
PRODUCTS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
SHOPS={
 'BAKERY':['EGG','WHEAT'],'PIZZA_SHOP':['MILK','TOMATO','WHEAT'],
 'BRUNCH_SPOT':['EGG','WHEAT','STRAWBERRY'],'YARN_STORE':['WOOL'],
 'ICE_CREAM_SHOP':['STRAWBERRY','MILK','WHEAT'],'PET_CAFE':['CARROT'],
 'SMOOTHIE_SHOP':['STRAWBERRY','MILK'],'FARMERS_MARKET':['WHEAT','CARROT','TOMATO','STRAWBERRY'],
}
MARKET_PARAMS={
 'WHEAT':{'base':25,'I0':10000,'T':400,'below_func':'sqrt','below_target':.8,'above_func':'log','above_target':.2},
 'CARROT':{'base':35,'I0':10000,'T':450,'below_func':'hinge','below_target':1.,'above_func':'sqrt','above_target':.7},
 'TOMATO':{'base':60,'I0':10000,'T':200,'below_func':'hinge','below_target':.4,'above_func':'sqrt','above_target':.6},
 'STRAWBERRY':{'base':120,'I0':10000,'T':100,'below_func':'sqrt','below_target':.7,'above_func':'linear','above_target':1.6},
 'MELON':{'base':250,'I0':10000,'T':300,'below_func':'log','below_target':.2,'above_func':'sq','above_target':3.6},
 'EGG':{'base':50,'I0':10000,'T':332,'below_func':'hinge','below_target':.4,'above_func':'log','above_target':.2},
 'MILK':{'base':160,'I0':10000,'T':122,'below_func':'sqrt','below_target':.6,'above_func':'linear','above_target':1.6},
 'WOOL':{'base':200,'I0':10000,'T':105,'below_func':'log','below_target':.2,'above_func':'sq','above_target':3.2},
 'FERTILIZER':{'base':100,'I0':10000,'T':200,'below_func':'linear','below_target':.4,'above_func':'linear','above_target':.4},
}
LAND_ORDER=('NW','NE','SW','SE'); LAND_COST={1:1000,2:2000,3:4000}
SHED_ACCESS=((4,4),(5,4),(4,5),(5,5)); HIRE_FIB=(1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987)
MOVES={'NORTH':(0,-1),'SOUTH':(0,1),'EAST':(1,0),'WEST':(-1,0)}

def _shape(k,x,T):
 x=max(0.,float(x))
 if k=='linear':return x
 if k=='sq':return x*x
 if k=='sqrt':return math.sqrt(x)
 if k=='log':return math.log1p(x)
 if k=='hinge':
  u=x/T;return u+8*max(0.,u-1.)**2
 return x

def market_price(item,inventory):
 p=MARKET_PARAMS[item];b=p['base'];I=p['I0'];T=p['T']
 if inventory<I:
  f=p['below_func'];v=b+p['below_target']*b/_shape(f,T,T)*_shape(f,I-inventory,T)
 else:
  f=p['above_func'];v=b-p['above_target']*b/_shape(f,T,T)*_shape(f,inventory-I,T)
 return max(1,int(round(v)))

def integral_revenue(item,start_inventory,q):
 q=max(0,int(q));s=0;inv=int(round(start_inventory))
 for _ in range(q):s+=market_price(item,inv);inv+=1
 return s

def quad(c):
 x,y=c;return ('N' if y<5 else 'S')+('W' if x<5 else 'E')

def md(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])

def path(a,b):
 x,y=a;tx,ty=b;o=[]
 while x<tx:o.append(['EAST']);x+=1
 while x>tx:o.append(['WEST']);x-=1
 while y<ty:o.append(['SOUTH']);y+=1
 while y>ty:o.append(['NORTH']);y-=1
 return o

def is_plant(t,crop=None):return isinstance(t,dict) and t.get('kind')=='PLANT' and (crop is None or t.get('crop')==crop)
def is_animal(t,a=None):return isinstance(t,dict) and 'animal' in t and (a is None or t.get('animal')==a)

def radial_cells(quads):
 qs=set(quads);out=[(x,y) for y in range(10) for x in range(10) if quad((x,y)) in qs]
 return sorted(out,key=lambda c:(min(md(c,s) for s in SHED_ACCESS),LAND_ORDER.index(quad(c)),c[1],c[0]))

def snake_key(c):
 x,y=c;q=quad(c)
 if q=='NW':return (0,4-y,(4-x if (4-y)%2==0 else x))
 if q=='NE':return (1,4-y,(x-5 if (4-y)%2==0 else 9-x))
 if q=='SW':return (2,y-5,(4-x if (y-5)%2==0 else x))
 return (3,y-5,(x-5 if (y-5)%2==0 else 9-x))

@dataclass
class Params:
 fix_resources:bool=True
 fix_expiry:bool=True
 fix_values:bool=True
 fix_calendar:bool=True
 fix_liquidity:bool=False
 fix_logistics:bool=False
 max_land:int=3
 max_animals:int=18
 max_cows:int=14
 max_sheep:int=12
 max_geese:int=8
 max_strawberry:int=40
 max_tomato:int=16
 max_melon:int=15
 opening_animals:tuple[str,...]=('COW','COW','SHEEP','SHEEP')
 opening_crops:tuple[int,...]=(-1,-1,-1,-1,-1)
 opening_melon:int=8
 opening_strawberry:int=4
 latest_animal_day:int=15
 timing_discount:float=.84
 action_shadow:float=2.0
 feed_price_mult:float=.95
 capital_fraction:float=.94
 future_shop_weight:float=1.0
 opponent_supply_weight:float=0.0
 own_feed_demand_weight:float=0.0
 temporal_value_weight:float=0.0
 competitive_sell_weight:float=0.0
 sell_horizon_days:int=3
 sell_advantage:float=0.0
 rotate_finite:bool=False
 rotation_margin:float=0.0
 hold_capacity:int=100
 shed_safety:int=1
 max_hands:int=14
 feed_cover_days:int=7
 feed_stock_cap:int=60
 force_min_cows:int=0
 force_min_sheep:int=0
 force_min_strawberry:int=0
 force_min_melon:int=0
 project_bias:dict[str,float]=field(default_factory=lambda:{'COW':1.,'SHEEP':1.,'GOOSE':.9,'STRAWBERRY':1.,'TOMATO':.9,'MELON':1.,'WHEAT':.78,'CARROT':.78})

@dataclass
class Project:
 kind:str; output_item:str; output:int; fert:int; feed:int; seed_cost:int; capital:int; actions:float; value:float=0.

@dataclass
class Job:
 pos:tuple[int,int]; actions:list[list[Any]]; needs:dict[str,int]=field(default_factory=dict); seeds:dict[str,int]=field(default_factory=dict)
 priority:int=20; output:dict[str,int]=field(default_factory=dict); tag:str=''

@dataclass
class Route:
 unit:int
 start:tuple[int,int]
 jobs:list[Job]=field(default_factory=list)
 needs:dict[str,int]=field(default_factory=dict)
 end:tuple[int,int]|None=None
 core_cost:int=0
 has_output:bool=False
 def __post_init__(self):
  if self.end is None:self.end=self.start
 def total_cost(self,return_drop=False):
  cost=self.core_cost
  if return_drop and self.has_output:
   cost+=min(md(self.end,s) for s in SHED_ACCESS)+1
  return cost
 def append_total(self,j,return_drop=False):
  new_keys=sum(1 for k,v in j.needs.items() if v>0 and self.needs.get(k,0)<=0)
  cost=self.core_cost+new_keys+md(self.end,j.pos)+len(j.actions)
  if return_drop and (self.has_output or bool(j.output)):
   cost+=min(md(j.pos,s) for s in SHED_ACCESS)+1
  return cost
 def append(self,j):
  self.core_cost+=sum(1 for k,v in j.needs.items() if v>0 and self.needs.get(k,0)<=0)
  self.core_cost+=md(self.end,j.pos)+len(j.actions)
  for k,v in j.needs.items():self.needs[k]=self.needs.get(k,0)+v
  self.jobs.append(j);self.end=j.pos;self.has_output=self.has_output or bool(j.output)
 def ordered_cost(self,jobs=None,return_drop=False):
  if jobs is None:return self.total_cost(return_drop)
  needs={};cost=0;pos=self.start;has_output=False
  for j in jobs:
   for k,v in j.needs.items():needs[k]=needs.get(k,0)+v
   cost+=md(pos,j.pos)+len(j.actions);pos=j.pos;has_output=has_output or bool(j.output)
  cost+=sum(1 for v in needs.values() if v>0)
  if return_drop and has_output:cost+=min(md(pos,s) for s in SHED_ACCESS)+1
  return cost

class DailyDPController:
 def __init__(self,p=None):self.p=p or Params();self.reset()
 def reset(self):
  self.day=-1;self.last_step=-1;self.phase='NEW';self.queue=[];self.target={};self.plans={};self.targets={};self.pi={};self.debug={};self.expected_output={};self.return_drop=False
 def farm(self,o):return o['farms'][o['player']]
 def tile(self,o,c):return self.farm(o)['tiles'][c[1]][c[0]]
 def counts(self,o):
  ac={a:0 for a in ANIMALS};cc={c:0 for c in CROPS}
  for row in self.farm(o)['tiles']:
   for t in row:
    if is_animal(t):ac[t['animal']]+=1
    elif is_plant(t):cc[t['crop']]+=1
  return ac,cc

 def future_demand(self,o):
  step=int(o.get('step',self.day*24));shops=list(o['town'].get('unlocked_shops',[]));d={x:0. for x in PRODUCTS}
  # Current/future scheduled ticks, including the tick after this action.
  shop_ticks=sum(1 for s in range(step,719) if s%4==0)
  center_ticks=sum(1 for s in range(step,719) if s%24==0)
  for x in PRODUCTS:
   if x!='FERTILIZER':d[x]+=center_ticks
  for sh in shops:
   ps=SHOPS[sh];m=2 if len(ps)==1 else 1
   for x in ps:d[x]+=m*shop_ticks
  known=len(shops)
  per={x:0. for x in PRODUCTS}
  for ps in SHOPS.values():
   m=2 if len(ps)==1 else 1
   for x in ps:per[x]+=m/len(SHOPS)
  unlock_days=(3,6,9,12,15,18,21,24)
  for ud in unlock_days[known:]:
   if ud*24>=719:continue
   ticks=sum(1 for s in range(max(step,ud*24),719) if s%4==0)
   for x in PRODUCTS:d[x]+=self.p.future_shop_weight*per[x]*ticks
  return d

 def crop_project(self,crop,day):
  cd=CROPS[crop]
  if cd['ongoing']:
   pday=day;out=0;fert=0;seeds=0;acts=0
   while pday+cd['first']<=29:
    events=[pday+cd['first']+i*cd['interval'] for i in range(cd['max_yield'])]
    events=[v for v in events if v<=29]
    if not events:break
    seeds+=1;out+=2*len(events);acts+=2 # plant + first-day water
    # daily watering through last production visibility day-1
    acts+=max(0,events[-1]-pday-1)
    # minimal 3-day fertilizer covers event action-days (visible-1)
    ed=[v-1 for v in events];i=0
    while i<len(ed):
     cover=ed[i]+2;fert+=1;acts+=1;i+=1
     while i<len(ed) and ed[i]<=cover:i+=1
    # harvest every two events plus terminal remainder
    acts+=math.ceil(len(events)/2)
    pday=pday+cd['first']+cd['interval']*(cd['max_yield']-1)+1
   return Project(crop,crop,out,fert,0,seeds*cd['seed'],cd['seed'],acts)
  h=cd['harvest_age'];cycles=max(0,((29 if self.p.fix_calendar else 28)-day)//h)
  if cycles<=0:return Project(crop,crop,0,0,0,0,cd['seed'],0)
  per={'WHEAT':4,'CARROT':3,'MELON':6}[crop]
  # water daily, one plant and one harvest per cycle
  acts=cycles*(h+2)
  return Project(crop,crop,cycles*per,0,0,cycles*cd['seed'],cd['seed'],acts)

 def animal_project(self,a,day):
  ad=ANIMALS[a]
  if day+ad['first']>29:return Project(a,ad['product'],0,0,0,0,ad['cost'],0)
  # Exact full-care schedule: capped opening batch, then 1+interval care bonus.
  first=ad['max_held'];n_after=max(0,((29 if self.p.fix_calendar else 28)-(day+ad['first']))//ad['interval'])
  out=first+n_after*(1+ad['interval'])
  service_days=max(0,29-day);fert=max(0,29-day);feed=service_days
  acts=4+service_days*3+n_after+1 # setup + daily service + harvests
  return Project(a,ad['product'],out,fert,feed,0,ad['cost'],acts)

 def existing_supply(self,o):
  # Current bankable stock plus conservative future production from live assets.
  q={x:int(o['private']['shed'].get(x,0)) for x in PRODUCTS}
  for inv in o['private']['inventories']:
   for x,n in inv.items():
    if x in q:q[x]+=int(n)
  feed=0;fert_need=0
  for row in self.farm(o)['tiles']:
   for t in row:
    if is_animal(t):
     a=t['animal'];ad=ANIMALS[a];q[ad['product']]+=int(t.get('yield_units',0))
     # Remaining production from next due events; conservative 1+interval each.
     for visible in range(self.day+1,30):
      ds=visible-int(t['placed_day'])-ad['first']
      if ds>=0 and ds%ad['interval']==0:q[ad['product']]+=1+ad['interval']
     q['FERTILIZER']+=max(0,29-self.day);feed+=max(0,29-self.day)
    elif is_plant(t):
     c=t['crop'];cd=CROPS[c];q[c]+=int(t.get('yield_units',0))
     if cd['ongoing']:
      for visible in range(self.day+1,30):
       ds=visible-int(t['planted_day'])-cd['first']
       if ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<cd['max_yield']:
        q[c]+=2;fert_need+=1 if False else 0
     else:
      age=self.day-int(t['planted_day']);h=cd['harvest_age'];remain=max(0,h-age)
      if self.day+remain<=28:q[c]+=max(0,{'WHEAT':4,'CARROT':3,'MELON':6}[c]-int(t.get('yield_units',0)))
  # Fertilizer used by existing ongoing crops is accounted operationally only;
  # clipping its sale forecast is safer than over-crediting all animal manure.
  q['FERTILIZER']=max(0,q['FERTILIZER']-sum(1 for row in self.farm(o)['tiles'] for t in row if is_plant(t) and t['crop'] in ('STRAWBERRY','TOMATO'))*2)
  return q,feed

 def project_values(self,o,base_q,selected_q):
  if self.p.temporal_value_weight:
   raise NotImplementedError('Temporal forecast is native-only experimental until reference parity is added.')
  demand=self.future_demand(o)
  if self.p.opponent_supply_weight:
   supply=self.opponent_supply(o)
   for x in PRODUCTS:demand[x]-=self.p.opponent_supply_weight*supply[x]
  own_animals=sum(is_animal(t) for row in self.farm(o)['tiles'] for t in row)
  demand['WHEAT']+=self.p.own_feed_demand_weight*own_animals*max(0,29-self.day)
  base_inv={x:float(o['market']['inventory'][x])-demand[x] for x in PRODUCTS}
  def marg(item,qty):
   before=int(base_q.get(item,0)+selected_q.get(item,0));return integral_revenue(item,base_inv[item]+before,qty)
  vals={}
  ac,cc=self.counts(o)
  projects=[self.animal_project(a,self.day) for a in ANIMALS]+[self.crop_project(c,self.day) for c in CROPS]
  feed_price=max(12.,float(o['market']['prices']['WHEAT'])*self.p.feed_price_mult)
  fert_price=max(25.,float(o['market']['prices']['FERTILIZER'])*.72)
  for pr in projects:
   if pr.output<=0:vals[pr.kind]=(-1e9,pr);continue
   gross=marg(pr.output_item,pr.output)
   if pr.fert>0 and (not self.p.fix_values or pr.kind in ANIMALS):gross+=marg('FERTILIZER',pr.fert)
   cost=pr.seed_cost+pr.feed*feed_price+pr.fert*0
   if self.p.fix_values and pr.kind in ANIMALS:cost+=pr.capital
   if pr.kind in ('STRAWBERRY','TOMATO'):cost+=pr.fert*fert_price
   net=self.p.timing_discount*gross-cost-self.p.action_shadow*pr.actions
   net*=self.p.project_bias.get(pr.kind,1.)
   vals[pr.kind]=(net,pr)
  return vals

 def opponent_supply(self,o):
  # Conditional public-asset forecast. No hidden inventory or future actions.
  q={x:0. for x in PRODUCTS};feed=0;fert_need=0
  for row in o['farms'][1-o['player']]['tiles']:
   for t in row:
    if is_animal(t):
     a=ANIMALS[t['animal']];q[a['product']]+=int(t.get('yield_units',0))
     for day in range(self.day+1,30):
      ds=day-int(t['placed_day'])-a['first']
      if ds>=0 and ds%a['interval']==0:q[a['product']]+=1+a['interval']
     q['FERTILIZER']+=max(0,29-self.day);feed+=max(0,29-self.day)
    elif is_plant(t):
     c=t['crop'];cd=CROPS[c];q[c]+=int(t.get('yield_units',0))
     if cd['ongoing']:
      for day in range(self.day+1,30):
       ds=day-int(t['planted_day'])-cd['first']
       if ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<4:q[c]+=2
      fert_need+=2
     elif self.day+max(0,cd['harvest_age']-(self.day-int(t['planted_day'])))<=29:
      q[c]+=max(0,{'WHEAT':4,'CARROT':3,'MELON':6}[c]-int(t.get('yield_units',0)))
  q['WHEAT']-=feed;q['FERTILIZER']=max(0,q['FERTILIZER']-fert_need)
  return q

 def competition_sales(self,o):
  sell={x:0 for x in PRODUCTS}
  if self.p.competitive_sell_weight<=0 or self.day>=29:return sell
  end=min(719,int(o['step'])+24*self.p.sell_horizon_days);last_day=min(29,(end-1)//24)
  demand={x:0. for x in PRODUCTS};supply={x:0. for x in PRODUCTS}
  for step in range(int(o['step']),end):
   if step%24==0:
    for x in PRODUCTS[:8]:demand[x]+=1
   if step%4==0:
    for sh in o['town']['unlocked_shops']:
     for x in SHOPS[sh]:demand[x]+=2 if len(SHOPS[sh])==1 else 1
  for seat in (o['player'],1-o['player']):
   for row in o['farms'][seat]['tiles']:
    for t in row:
     if is_animal(t):
      a=ANIMALS[t['animal']];supply[a['product']]+=int(t.get('yield_units',0))
      for day in range(self.day+1,last_day+1):
       ds=day-int(t['placed_day'])-a['first']
       if ds>=0 and ds%a['interval']==0:supply[a['product']]+=1+a['interval']
     elif is_plant(t):
      x=t['crop'];cd=CROPS[x];supply[x]+=int(t.get('yield_units',0))
      if cd['ongoing']:
       for day in range(self.day+1,last_day+1):
        ds=day-int(t['planted_day'])-cd['first']
        if ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<4:supply[x]+=2
      elif int(t['planted_day'])+cd['harvest_age']<=last_day:
       supply[x]+=max(0,{'WHEAT':4,'CARROT':3,'MELON':6}[x]-int(t.get('yield_units',0)))
  for x in PRODUCTS[1:8]:
   n=int(o['private']['shed'].get(x,0))
   if n<=0:continue
   inv=float(o['market']['inventory'][x]);future=inv-demand[x]+self.p.competitive_sell_weight*supply[x]
   if integral_revenue(x,inv,n)-integral_revenue(x,future,n)>self.p.sell_advantage*n:sell[x]=n
  return sell

 def choose_target(self,o):
  farm=self.farm(o);ac,cc=self.counts(o);current_land=len(farm['unlocked_quadrants'])
  # One land option at a time. It enters target early enough to purchase today.
  planned_land=current_land
  earliest_land_day={1:9,2:12,3:99}.get(current_land,99)
  if current_land<self.p.max_land and self.day>=earliest_land_day:
   lc=LAND_COST[current_land]
   # Existing stock can finance land, but do not liquidate the whole farm for it.
   liquid=farm['money']+sum(int(o['private']['shed'].get(x,0))*int(o['market']['prices'][x])*.55 for x in PRODUCTS)
   if liquid>=lc+350:planned_land=current_land+1
  quads=list(LAND_ORDER[:planned_land]);cells=radial_cells(quads);target={}
  available=[]
  # Protect sunk assets. Harvest-ready finite crops may change replacement type.
  for c in cells:
   t=self.tile(o,c)
   if is_animal(t):target[c]=t['animal']
   elif is_plant(t):
    crop=t['crop'];cd=CROPS[crop]
    target[c]=crop
   else:available.append(c)
  # Explicit day-0 hedge derived from expected-value/capital constraints.
  if self.day==0 and sum(ac.values())==0:
   if self.p.opening_crops[0]>=0:
    counts=dict(zip(CROPS,self.p.opening_crops))
    seq=list(self.p.opening_animals)+[x for x in ('MELON','STRAWBERRY','TOMATO','CARROT','WHEAT') for _ in range(counts[x])]
   else:seq=list(self.p.opening_animals)+['MELON']*self.p.opening_melon+['STRAWBERRY']*self.p.opening_strawberry
   for c,k in zip(sorted(available,key=lambda z:(min(md(z,s) for s in SHED_ACCESS),snake_key(z))),seq):target[c]=k
   for c in available:
    if c not in target:target[c]=None
   self.target=target;self.debug['planned_land']=planned_land;self.debug['selection']=seq;return
  base_q,base_feed=self.existing_supply(o);selected_q={x:0 for x in PRODUCTS}
  existing_counts={**ac,**cc};chosen=[]
  # Immediate capital envelope. Daily operating inputs and hires remain reserved.
  liquid=farm['money']
  for x,n in o['private']['shed'].items():
   if x in PRODUCTS:liquid+=n*o['market']['prices'][x]*.72
  land_reserve=LAND_COST[current_land] if planned_land>current_land else 0
  animal_n=sum(ac.values());mandatory_feed=max(0,animal_n-int(o['private']['shed'].get('WHEAT',0)))*o['market']['prices']['WHEAT']
  budget=max(0.,self.p.capital_fraction*liquid-land_reserve-mandatory_feed-450)
  spent=0.
  limits={'COW':self.p.max_cows,'SHEEP':self.p.max_sheep,'GOOSE':self.p.max_geese,'STRAWBERRY':self.p.max_strawberry,'TOMATO':self.p.max_tomato,'MELON':self.p.max_melon,'WHEAT':75,'CARROT':75}
  # Desired lower bounds avoid a myopic late-shop outcome; they are economic
  # diversification floors, not a fixed action route.
  floors={'COW':self.p.force_min_cows,'SHEEP':self.p.force_min_sheep,'STRAWBERRY':self.p.force_min_strawberry,'MELON':self.p.force_min_melon}
  while len(chosen)<len(available):
   vals=self.project_values(o,base_q,selected_q);options=[]
   total_animals=animal_n+sum(1 for x in chosen if x in ANIMALS)
   for k,(v,pr) in vals.items():
    cnt=existing_counts.get(k,0)+chosen.count(k)
    if cnt>=limits.get(k,99):continue
    if k in ANIMALS:
     if self.day>self.p.latest_animal_day or total_animals>=self.p.max_animals:continue
    cap=pr.capital
    if spent+cap>budget:continue
    # Strong but finite diversification bonus until floor is reached.
    if cnt<floors.get(k,0):v+=3500 if k in ANIMALS else 1800
    # Late fast crops remain available; long projects require bankable output.
    if v<=0:continue
    options.append((v,k,pr))
   if not options:break
   v,k,pr=max(options,key=lambda z:(z[0],-z[2].capital,z[1]))
   chosen.append(k);spent+=pr.capital
   selected_q[pr.output_item]=selected_q.get(pr.output_item,0)+pr.output
   if pr.fert:selected_q['FERTILIZER']=selected_q.get('FERTILIZER',0)+(pr.fert if not self.p.fix_values or pr.kind in ANIMALS else -pr.fert)
  # Service-heavy projects closest to shed; crops occupy deterministic radial lanes.
  intensity={'COW':4,'SHEEP':4,'GOOSE':4,'STRAWBERRY':2,'TOMATO':2,'MELON':1,'WHEAT':1,'CARROT':1}
  chosen.sort(key=lambda k:(-intensity[k],k))
  acells=sorted(available,key=lambda z:(min(md(z,s) for s in SHED_ACCESS),snake_key(z)))
  for c,k in zip(acells,chosen):target[c]=k
  for c in available:
   if c not in target:target[c]=None
  self.target=target;self.debug.update(planned_land=planned_land,selection=chosen,budget=budget,spent=spent)

 def fertilize_due(self,t):
  c=t['crop'];cd=CROPS[c]
  if not cd['ongoing']:return False
  nextday=self.day+1;ds=nextday-int(t['planted_day'])-cd['first']
  due=ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<cd['max_yield']
  return due and int(t.get('fertilized_until_day',-1))<self.day

 def rotate_targets(self,o):
  if not self.p.rotate_finite or self.day>=29:return
  base,_=self.existing_supply(o);selected={x:0 for x in PRODUCTS};ac,live=self.counts(o)
  limits={'WHEAT':75,'CARROT':75,'TOMATO':self.p.max_tomato,'STRAWBERRY':self.p.max_strawberry,'MELON':self.p.max_melon}
  for pos,want in list(self.target.items()):
   t=self.tile(o,pos)
   if not is_plant(t) or CROPS[t['crop']]['ongoing']:continue
   current=t['crop'];expiry=int(t.get('yield_units',0))>0 and 0<=int(t.get('max_lifespan_step',-1))<=(self.day+1)*24
   if self.day-int(t['planted_day'])<CROPS[current]['harvest_age'] and not expiry:continue
   vals=self.project_values(o,base,selected);best=vals[current][0];choice=current;chosen=None
   for c,(v,pr) in vals.items():
    if c not in CROPS or c==current or live[c]>=limits[c] or pr.output<=0:continue
    extra=max(0,CROPS[c]['seed']-CROPS[current]['seed'])
    if o['private']['seeds'].get(c,0)<=0 and extra>self.farm(o)['money']:continue
    if v>best+self.p.rotation_margin:best=v;choice=c;chosen=pr
   if choice!=current:
    self.target[pos]=choice;live[current]-=1;live[choice]+=1
    selected[chosen.output_item]+=chosen.output;selected['FERTILIZER']-=chosen.fert

 def build_jobs(self,o,anticipate=False):
  jobs=[];day=self.day
  for c,want in self.target.items():
   if want is None:continue
   t=self.tile(o,c)
   if t=='LOCKED':
    if anticipate:t=None
    else:continue
   if want in ANIMALS:
    if is_animal(t):
     a=t['animal'];ad=ANIMALS[a];acts=[];out={}
     y=int(t.get('yield_units',0));nextday=day+1;ds=nextday-int(t['placed_day'])-ad['first'];due=ds>=0 and ds%ad['interval']==0
     incoming=(1+int(t.get('pending_care_bonus',0))) if due and not t.get('fed_today',False) else 0
     if y>0 and (day>=29 or y+incoming>ad['max_held'] or y>=ad['max_held']):acts.append(['HARVEST']);out[ad['product']]=y
     if t.get('fertilizer_available',False):acts.append(['COLLECT_FERTILIZER']);out['FERTILIZER']=1
     needs={}
     if day<29:
      if not t.get('fed_today',False):acts.append(['FEED']);needs['WHEAT']=1
      if not t.get('cared_today',False):acts.append(['CARE'])
     if acts:jobs.append(Job(c,acts,needs=needs,priority=0 if day<29 else -5,output=out,tag='animal'))
    elif t is None or (isinstance(t,dict) and t.get('kind') in ('WEED','COOP','PASTURE')):
     acts=[]
     if isinstance(t,dict) and t.get('kind')=='WEED':acts.append(['DIG']);t=None
     st=ANIMALS[want]['structure']
     if t is None:acts.append(['BUILD_'+st])
     elif isinstance(t,dict) and t.get('kind')!=st:acts += [['DIG'],['BUILD_'+st]]
     acts += [['PLACE',want],['FEED'],['CARE']]
     jobs.append(Job(c,acts,needs={want:1,'WHEAT':1},priority=6,tag='new_animal'))
   elif want in CROPS:
    if is_animal(t):continue
    acts=[];needs={};seeds={};out={}
    if isinstance(t,dict) and t.get('kind') in ('WEED','COOP','PASTURE'):
     if day>=29:continue
     acts.append(['DIG']);t=None
    desired=want
    if is_plant(t):
     c0=t['crop'];cd=CROPS[c0];y=int(t.get('yield_units',0))
     if cd['ongoing']:
      bonus=2 if self.fertilize_due(t) else 1
      expiry_due=self.p.fix_expiry and 0<=int(t.get('max_lifespan_step',-1))<=(day+1)*24
      if y>0 and (day>=29 or expiry_due or y>=cd['max_yield'] or (self.fertilize_due(t) and y+bonus>cd['max_yield'])):
       acts.append(['HARVEST']);out[c0]=y
      if day<29:
       if self.fertilize_due(t):acts.append(['FERTILIZE']);needs['FERTILIZER']=1
       if not t.get('watered_today',False):acts.append(['WATER'])
      # Existing ongoing crop never changes before decay.
      desired=c0
     else:
      age=day-int(t['planted_day'])
      expiry_due=self.p.fix_expiry and y>0 and 0<=int(t.get('max_lifespan_step',-1))<=(day+1)*24
      if (age>=cd['harvest_age'] or expiry_due) and day<=28:
       if not t.get('watered_today',False):acts.append(['WATER'])
       acts.append(['HARVEST']);out[c0]=max(y,{'WHEAT':4,'CARROT':3,'MELON':6}[c0])
       t=None
      elif day>=29:
       if y>0:acts.append(['HARVEST']);out[c0]=y
      elif not t.get('watered_today',False):acts.append(['WATER'])
    if t is None and day<29:
     cd=CROPS[desired];bank_day=day+(cd.get('harvest_age',cd['first']))
     if bank_day<=(29 if self.p.fix_calendar else 28):
      acts += [['PLANT',desired],['WATER']];seeds[desired]=1
    if acts:
     pr=0 if is_plant(self.tile(o,c)) else 12
     if out:pr=-1
     jobs.append(Job(c,acts,needs=needs,seeds=seeds,priority=pr,output=out,tag='crop'))
  return jobs

 def purchase_orders(self,o,jobs):
  shed=o['private']['shed'];seeds=o['private']['seeds'];need={};sneed={}
  for j in jobs:
   for k,v in j.needs.items():need[k]=need.get(k,0)+v
   for k,v in j.seeds.items():sneed[k]=sneed.get(k,0)+v
  q=[];current=len(self.farm(o)['unlocked_quadrants']);desired=int(self.debug.get('planned_land',current))
  if desired>current:q.append(['BUY_LAND'])
  for a in ('COW','SHEEP','GOOSE'):
   n=max(0,need.get(a,0)-int(shed.get(a,0)))
   if n:q.append(['BUY_ANIMAL',a,n])
  daily_wheat=int(need.get('WHEAT',0));have_wheat=int(shed.get('WHEAT',0))
  n=max(0,daily_wheat-have_wheat)
  if n:q.append(['BUY_PRODUCT','WHEAT',n,'OPERATING'])
  # Inventory DP for feed: buy several future service-days while wheat is still
  # cheap, but cap physical storage and never outrank land/current production.
  horizon=max(0,min(self.p.feed_cover_days,29-self.day))
  target=min(self.p.feed_stock_cap,daily_wheat*horizon)
  future_price=market_price('WHEAT',int(o['market']['inventory']['WHEAT']-self.future_demand(o)['WHEAT']))
  if self.day<22 and future_price>=int(o['market']['prices']['WHEAT'])+5:
   extra=max(0,target-max(have_wheat,daily_wheat))
   if extra:q.append(['BUY_PRODUCT','WHEAT',extra,'STOCK'])
  else:target=daily_wheat
  self.debug['feed_stock_target']=target
  n=max(0,need.get('FERTILIZER',0)-int(shed.get('FERTILIZER',0)))
  if n and self.day<15:q.append(['BUY_PRODUCT','FERTILIZER',n])
  for c in ('MELON','STRAWBERRY','TOMATO','CARROT','WHEAT'):
   n=max(0,sneed.get(c,0)-int(seeds.get(c,0)))
   if n:q.append(['BUY_SEED',c,n])
  self.debug['daily_need']=need
  return q

 def order_cost(self,o,x):
  if x[0]=='BUY_LAND':return LAND_COST[len(self.farm(o)['unlocked_quadrants'])]
  if x[0]=='BUY_ANIMAL':return ANIMALS[x[1]]['cost']*x[2]
  if x[0]=='BUY_SEED':return CROPS[x[1]]['seed']*x[2]
  if x[0]=='BUY_PRODUCT':
   item=x[1];q=int(x[2]);inv=int(o['market']['inventory'][item]);cost=0
   for _ in range(q):inv-=1;cost+=market_price(item,inv)
   return cost
  return 0

 def forecast_daily_output(self,jobs):
  q={x:0 for x in PRODUCTS}
  for j in jobs:
   for x,n in j.output.items():q[x]+=n
  return q

 def hold_scores(self,o):
  d=self.future_demand(o);scores={}
  for x in PRODUCTS:
   now=float(o['market']['prices'][x]);future=market_price(x,int(o['market']['inventory'][x]-d[x]))
   scores[x]=(future-now, future, now)
  return scores

 def sale_orders(self,o,force_cash=0.,terminal=False,extra_drop=None):
  shed=dict(o['private']['shed']);extra_drop=extra_drop or {}
  for x,n in extra_drop.items():shed[x]=shed.get(x,0)+n
  if terminal:return [['SELL',x,int(n)] for x,n in sorted(shed.items(),key=lambda z:o['market']['prices'].get(z[0],0),reverse=True) if x in PRODUCTS and n>0]
  # Reserve today's feed/fertilizer inputs.
  need=self.debug.get('daily_need',{});reserve={'WHEAT':max(int(need.get('WHEAT',0)),min(int(o['private']['shed'].get('WHEAT',0)),int(self.debug.get('feed_stock_target',0)))),'FERTILIZER':int(need.get('FERTILIZER',0))}
  avail={x:max(0,int(shed.get(x,0))-reserve.get(x,0)) for x in PRODUCTS}
  scores=self.hold_scores(o);total_hold=sum(avail.values());sell={x:0 for x in PRODUCTS}
  # No-demand/weak-appreciation goods are released first. Keep at most hold_capacity.
  incoming=sum(self.expected_output.values()) if self.expected_output else 0
  hold_limit=max(0,min(self.p.hold_capacity,100-self.p.shed_safety-incoming))
  excess=max(0,total_hold-hold_limit)
  for x in sorted(PRODUCTS,key=lambda z:(scores[z][0],scores[z][1])):
   n=avail[x]
   take=min(n,excess);sell[x]+=take;avail[x]-=take;excess-=take
  # Raise required cash by sacrificing the lowest expected appreciation units.
  cash=float(self.farm(o)['money']);inv=dict(o['market']['inventory'])
  for x in sorted(PRODUCTS,key=lambda z:(scores[z][0],scores[z][1])):
   while avail[x]>0 and cash<force_cash:
    pr=market_price(x,int(inv[x]+sell[x]));sell[x]+=1;avail[x]-=1;cash+=pr
  # Fertilizer and melon cannot appreciate through shops; sell remaining unless
  # they are being retained for today's fertilizer use.
  for x in ('FERTILIZER','MELON'):
   sell[x]+=avail[x];avail[x]=0
  return [['SELL',x,n] for x,n in sorted(sell.items(),key=lambda z:o['market']['prices'][z[0]],reverse=True) if n>0]

 def funding_sales(self,o,force_cash,required_slots=0):
  """Sell only what is needed to fund today's irreversible setup.

  Capacity liquidation is deliberately postponed to the last hour of the day,
  after the town has consumed six more times.  Fertilizer is the exception:
  shops never consume it, so retaining yesterday's manure has no option value.
  """
  shed=dict(o['private']['shed']);need=self.debug.get('daily_need',{})
  reserve={'WHEAT':max(int(need.get('WHEAT',0)),min(int(o['private']['shed'].get('WHEAT',0)),int(self.debug.get('feed_stock_target',0)))),'FERTILIZER':int(need.get('FERTILIZER',0))}
  avail={x:max(0,int(shed.get(x,0))-reserve.get(x,0)) for x in PRODUCTS}
  sell={x:0 for x in PRODUCTS};cash=float(self.farm(o)['money']);inv=dict(o['market']['inventory'])
  # Manure has no exogenous consumer.  Realise it immediately for cash and to
  # make room for tonight's collection.
  n=avail['FERTILIZER'];sell['FERTILIZER']=n;avail['FERTILIZER']=0
  for _ in range(n):
   cash+=market_price('FERTILIZER',inv['FERTILIZER']);inv['FERTILIZER']+=1
  # Unit-level opportunity-cost liquidation.  Recompute the cheapest next unit
  # because each sale moves its own nonlinear market curve.  Market purchases of
  # feed/animals land in the shed before workers can pick them up, so setup also
  # reserves physical slots, not just cash.
  demand=self.future_demand(o);free=max(0,100-sum(int(v) for v in o['private']['shed'].values())+sum(sell.values()))
  slots_left=max(0,int(required_slots)-free)
  tactical=self.competition_sales(o)
  for x in PRODUCTS[1:8]:
   n=min(avail[x],tactical[x])
   for k in range(n):cash+=market_price(x,inv[x]+sell[x]+k)
   sell[x]+=n;avail[x]-=n;slots_left=max(0,slots_left-n)
  while cash<force_cash or slots_left>0:
   best=None
   for x,n in avail.items():
    if n<=0:continue
    now=market_price(x,inv[x]+sell[x])
    future=market_price(x,inv[x]+sell[x]-demand[x])
    loss=future-now
    key=(loss,future,now,x)
    if best is None or key<best[0]:best=(key,x,now)
   if best is None:break
   _,x,pr=best;sell[x]+=1;avail[x]-=1;cash+=pr;slots_left=max(0,slots_left-1)
  return [['SELL',x,n] for x,n in sorted(sell.items(),key=lambda z:o['market']['prices'][z[0]],reverse=True) if n>0]

 def pending_eod_incoming(self,o):
  """Exact inventory already carried plus conservative output on this hour."""
  total=sum(sum(int(v) for v in inv.values()) for inv in o['private']['inventories'])
  farm=self.farm(o);poslist=[farm['farmer']]+farm['hands']
  for u,pos0 in enumerate(poslist):
   plan=self.plans.get(u,[]);i=self.pi.get(u,0)
   if i>=len(plan):continue
   a=plan[i];pos=tuple(pos0);tg=self.targets.get(u,[None]*len(plan))[i]
   if tg is not None and pos!=tg:continue
   t=farm['tiles'][pos[1]][pos[0]];op=a[0]
   if op=='HARVEST' and isinstance(t,dict):total+=max(0,int(t.get('yield_units',0)))
   elif op=='COLLECT_FERTILIZER' and is_animal(t) and t.get('fertilizer_available',False):total+=1
  return total

 def capacity_sales(self,o,incoming=None):
  """Latest-feasible daily storage DP.

  At hour 23, retain the units with the largest expected waiting gain and sell
  only enough to make tonight's auto-deposit fit in the 100-slot shed.
  """
  shed={x:int(o['private']['shed'].get(x,0)) for x in PRODUCTS}
  if incoming is None:incoming=self.pending_eod_incoming(o)
  cap=max(0,min(self.p.hold_capacity,100-self.p.shed_safety-incoming))
  total=sum(shed.values());must=max(0,total-cap);sell={x:0 for x in PRODUCTS}
  locked_wheat=min(shed['WHEAT'],int(self.debug.get('feed_stock_target',0)))
  shed['WHEAT']-=locked_wheat
  scores=self.hold_scores(o)
  # Melon has only one center tick per day and fertilizer none; liquidate both
  # at the latest daily price even when there is spare storage.
  for x in ('FERTILIZER','MELON'):
   sell[x]=shed[x];must=max(0,must-shed[x]);shed[x]=0
  for x in sorted(PRODUCTS,key=lambda z:(scores[z][0],scores[z][1],scores[z][2],z)):
   if must<=0:break
   n=min(shed[x],must);sell[x]+=n;shed[x]-=n;must-=n
  if must>0 and locked_wheat>0:
   n=min(locked_wheat,must);sell['WHEAT']+=n;must-=n
  return [['SELL',x,n] for x,n in sorted(sell.items(),key=lambda z:o['market']['prices'][z[0]],reverse=True) if n>0]

 def simulate_hire_starts(self,n_hands):
  starts=[(4,4)];occ={s:0 for s in SHED_ACCESS};occ[(4,4)]=1
  for _ in range(n_hands):
   s=min(SHED_ACCESS,key=lambda z:(occ[z],SHED_ACCESS.index(z)));occ[s]+=1;starts.append(s)
  return starts

 def pack(self,jobs,starts,budget,return_drop=False):
  """Fast state-feedback route packing.

  Jobs are globally ordered by hard priority and a deterministic farm-space
  sweep.  Each job is appended to the route with the smallest exact marginal
  action cost.  Unlike the earlier all-insertion search this is O(J*U), so the
  hand-count DP can be evaluated every day without timing out.
  """
  routes=[Route(i,s) for i,s in enumerate(starts)];dropped=[]
  for j in sorted(jobs,key=lambda z:(z.priority,snake_key(z.pos),-len(z.actions))):
   best=None
   for r in routes:
    cost=r.append_total(j,return_drop)
    if cost<=budget:
     inc=cost-r.total_cost(return_drop);key=(inc,cost,len(r.jobs),r.unit)
     if best is None or key<best[0]:best=(key,r)
   if best is None:dropped.append(j)
   else:best[1].append(j)
  return routes,dropped

 def pack_insertion(self,jobs,starts,budget,return_drop=False):
  """Exact priority-preserving insertion fallback.

  The daily fast path remains O(J*U).  This quadratic fallback is invoked only
  when that fast packer leaves work unassigned, recovering geometry-induced
  drops without charging for another Fibonacci-priced worker.
  """
  routes=[Route(i,s) for i,s in enumerate(starts)];dropped=[]
  for j in sorted(jobs,key=lambda z:(z.priority,-len(z.actions),snake_key(z.pos))):
   best=None
   for r in routes:
    old=r.ordered_cost(r.jobs,return_drop)
    for at in range(len(r.jobs)+1):
     js=r.jobs[:at]+[j]+r.jobs[at:];cost=r.ordered_cost(js,return_drop)
     if cost<=budget:
      key=(cost-old,cost,len(r.jobs),r.unit,at)
      if best is None or key<best[0]:best=(key,r,at)
   if best is None:dropped.append(j)
   else:best[1].jobs.insert(best[2],j)
  # Rebuild cached route metadata used by compilation/debugging.
  rebuilt=[]
  for r in routes:
   nr=Route(r.unit,r.start)
   for j in r.jobs:nr.append(j)
   rebuilt.append(nr)
  return rebuilt,dropped

 def estimate_hands(self,jobs,base_orders):
  for h in range(0,self.p.max_hands+1):
   setup=math.ceil((base_orders+h)/10) if base_orders+h else 0
   compile_hour=max(1,setup);end=22 if self.day>=29 else 23;budget=end-compile_hour+1
   routes,drop=self.pack(jobs,self.simulate_hire_starts(h),budget,self.day>=29)
   mandatory=[j for j in drop if j.priority<=0]
   if not mandatory and not drop:return h,compile_hour,0
  return self.p.max_hands,math.ceil((base_orders+self.p.max_hands)/10),999

 def projected_sale_cash(self,o,sales):
  cash=float(self.farm(o)['money']);inv=dict(o['market']['inventory'])
  for x in sales:
   if x[0]!='SELL':continue
   for _ in range(int(x[2])):cash+=market_price(x[1],inv[x[1]]);inv[x[1]]+=1
  return cash

 def admit_buys(self,o,orders,cash):
  out=[];money=float(cash)
  def rk(z):
   if z[0]=='BUY_PRODUCT' and len(z)>3 and z[3]=='OPERATING':return 0
   if z[0]=='BUY_LAND':return 1
   if z[0]=='BUY_ANIMAL':return 2
   if z[0]=='BUY_SEED':return 3
   if z[0]=='BUY_PRODUCT':return 4
   return 9
  local_inv=dict(o['market']['inventory'])
  for x in sorted(orders,key=lambda z:(rk(z),0 if len(z)>1 and z[1] in ('WHEAT','MELON','STRAWBERRY') else 1)):
   if x[0]=='BUY_LAND':
    c=self.order_cost(o,x)
    if c<=money:out.append(x);money-=c
   elif x[0]=='BUY_PRODUCT':
    item=x[1];want=int(x[2]);n=0;cost=0;inv=int(local_inv[item])
    while n<want:
     pr=market_price(item,inv-1)
     if cost+pr>money:break
     cost+=pr;inv-=1;n+=1
    if n>0:
     out.append([x[0],item,n]+(x[3:] if len(x)>3 else []));money-=cost;local_inv[item]=inv
   else:
    unit=self.order_cost(o,[x[0],x[1],1]);n=min(int(x[2]),int(money//max(1,unit)))
    if n>0:out.append([x[0],x[1],n]+(x[3:] if len(x)>3 else []));money-=unit*n
  return out

 def new_day(self,o):
  self.day=int(o['day']);self.phase='SETUP';self.queue=[];self.plans={};self.targets={};self.pi={};self.debug={}
  self.choose_target(o);self.rotate_targets(o);anticipated=self.build_jobs(o,True);self.expected_output=self.forecast_daily_output(anticipated);buys0=self.purchase_orders(o,anticipated)
  if self.day>=29:
   # Existing shed stock can be sold concurrently once work starts.  Keeping it
   # out of the setup queue saves a full terminal hour in the common <=10-hire
   # case and prevents two-hour plant decay before harvest.
   sales=[];buys=[];cash=float(self.farm(o)['money'])
   h,setup,dropped=self.estimate_hands(anticipated,0);hire_cost=sum(HIRE_FIB[:h])
  else:
   # First estimate labour, then raise exactly the cash needed for purchases and
   # that labour.  A second pass handles the small order-count circularity.
   h,setup,dropped=self.estimate_hands(anticipated,len(buys0)+2)
   target=sum(self.order_cost(o,x) for x in buys0)+sum(HIRE_FIB[:h])+8
   slots=sum(int(x[2]) for x in buys0 if x[0] in ('BUY_PRODUCT','BUY_ANIMAL'))
   sales=self.funding_sales(o,target,slots);cash=self.projected_sale_cash(o,sales)
   buys=self.admit_buys(o,buys0,max(0,cash-sum(HIRE_FIB[:h])-8))
   h,setup,dropped=self.estimate_hands(anticipated,len(sales)+len(buys))
   target=sum(self.order_cost(o,x) for x in buys)+sum(HIRE_FIB[:h])+8
   slots=sum(int(x[2]) for x in buys0 if x[0] in ('BUY_PRODUCT','BUY_ANIMAL'))
   sales=self.funding_sales(o,target,slots);cash=self.projected_sale_cash(o,sales)
   buys=self.admit_buys(o,buys0,max(0,cash-sum(HIRE_FIB[:h])-8))
   hire_cost=sum(HIRE_FIB[:h])
   while h>0 and hire_cost+sum(self.order_cost(o,x) for x in buys)>cash:
    h-=1;hire_cost=sum(HIRE_FIB[:h])
  self.queue=list(sales)+buys+[['HIRE'] for _ in range(h)]
  self.debug.update(target_hands=h,estimated_drop=dropped,setup_orders=len(self.queue),anticipated_jobs=len(anticipated))
  if not self.queue:self.phase='COMPILE_NEXT'

 def reserve_jobs(self,o,jobs):
  shed=dict(o['private']['shed']);seeds=dict(o['private']['seeds']);out=[]
  for j in sorted(jobs,key=lambda z:(z.priority,snake_key(z.pos))):
   ok=all(shed.get(k,0)>=v for k,v in j.needs.items()) and all(seeds.get(k,0)>=v for k,v in j.seeds.items())
   if not ok and self.p.fix_resources and (is_plant(self.tile(o,j.pos)) or is_animal(self.tile(o,j.pos))):
    # Retain independent service actions; unavailable optional successors do not
    # cancel legal collection. Rebuild the exact resource demand of retained work.
    acts=[];needs={};sneed={};plant_present=is_plant(self.tile(o,j.pos))
    for a in j.actions:
     op=a[0]
     if op=='PLANT':
      if seeds.get(a[1],0)-sneed.get(a[1],0)<=0:continue
      sneed[a[1]]=sneed.get(a[1],0)+1;plant_present=True
     elif op in ('FEED','FERTILIZE'):
      item='WHEAT' if op=='FEED' else 'FERTILIZER'
      if shed.get(item,0)-needs.get(item,0)<=0:continue
      needs[item]=needs.get(item,0)+1
     elif op=='WATER' and not plant_present:continue
     elif op=='HARVEST' and plant_present:
      if not CROPS[self.tile(o,j.pos)['crop']]['ongoing']:plant_present=False
     acts.append(a)
    if acts:
     j=replace(j,actions=acts,needs=needs,seeds=sneed);ok=True
     self.debug['resource_degraded_jobs']=self.debug.get('resource_degraded_jobs',0)+1
   if not ok:
    # Existing service degrades only optional fertilizer collection/use; feed and
    # water remain hard priorities.
    if j.tag=='crop' and j.needs.get('FERTILIZER',0):
     acts=[a for a in j.actions if a[0]!='FERTILIZE'];needs={k:v for k,v in j.needs.items() if k!='FERTILIZER'}
     if all(shed.get(k,0)>=v for k,v in needs.items()) and all(seeds.get(k,0)>=v for k,v in j.seeds.items()):j=Job(j.pos,acts,needs,j.seeds,j.priority,j.output,j.tag);ok=True
   if not ok:continue
   for k,v in j.needs.items():shed[k]-=v
   for k,v in j.seeds.items():seeds[k]-=v
   out.append(j)
  return out

 def compile(self,o):
  jobs=self.reserve_jobs(o,self.build_jobs(o,False));self.expected_output=self.forecast_daily_output(jobs)
  current_shed=sum(o['private']['shed'].values());daily=sum(self.expected_output.values())
  # Normal-day inventories auto-deposit at EOD.  Explicit depot returns are
  # reserved for the terminal day, where cash must be banked before scoring.
  self.return_drop=self.day>=29
  starts=[tuple(self.farm(o)['farmer'])]+[tuple(x) for x in self.farm(o)['hands']]
  end=22 if self.day>=29 else 23;budget=end-int(o['hour'])+1
  routes,drop=self.pack(jobs,starts,budget,self.return_drop)
  if drop:
   iroutes,idrop=self.pack_insertion(jobs,starts,budget,self.return_drop)
   if len(idrop)<len(drop):routes,drop=iroutes,idrop
  self.debug.update(actual_jobs=len(jobs),actual_drop=len(drop),budget=budget,route_costs={r.unit:r.ordered_cost(return_drop=self.return_drop) for r in routes})
  deposit_units=set()
  if self.p.fix_logistics and not self.return_drop:
   # Add only feasible early returns needed to reduce tonight's excess. No
   # mandatory depot tour for every worker and no fixed-day liquidation rule.
   excess=max(0,daily+sum(sum(inv.values()) for inv in o['private']['inventories'])-100+self.p.shed_safety)
   options=[]
   for r in routes:
    qty=sum(sum(j.output.values()) for j in r.jobs)
    extra=min(md(r.end,s) for s in SHED_ACCESS)+1
    if qty>0 and r.ordered_cost(return_drop=True)<=budget:
     options.append((extra/max(1,qty),extra,r.unit,qty))
   for _,extra,u,qty in sorted(options):
    if excess<=0:break
    deposit_units.add(u);excess-=qty
   self.debug['early_deposit_units']=sorted(deposit_units)
   self.debug['unresolved_predicted_overflow']=max(0,excess)
  for r in routes:
   p=[];tg=[];pos=r.start;needs={}
   for j in r.jobs:
    for k,v in j.needs.items():needs[k]=needs.get(k,0)+v
   for x,n in sorted(needs.items()):
    if n>0:p.append(['PICKUP',x,n]);tg.append(pos)
   for j in r.jobs:
    for a in path(pos,j.pos):p.append(a);tg.append(None);dx,dy=MOVES[a[0]];pos=(pos[0]+dx,pos[1]+dy)
    for a in j.actions:p.append(a);tg.append(j.pos)
    pos=j.pos
   if (self.return_drop or r.unit in deposit_units) and any(j.output for j in r.jobs):
    dest=min(SHED_ACCESS,key=lambda s:md(pos,s))
    for a in path(pos,dest):p.append(a);tg.append(None);dx,dy=MOVES[a[0]];pos=(pos[0]+dx,pos[1]+dy)
    p.append(['DROP']);tg.append(dest)
   self.plans[r.unit]=p;self.targets[r.unit]=tg;self.pi[r.unit]=0
  self.phase='WORK'

 def valid(self,o,u,a,tg,res):
  farm=self.farm(o);poslist=[farm['farmer']]+farm['hands']
  if u>=len(poslist):return None
  pos=tuple(poslist[u]);op=a[0];inv=o['private']['inventories'][u]
  if op in MOVES or op=='PASS':return a
  if tg is not None and pos!=tg:
   pp=path(pos,tg);return pp[0] if pp else None
  t=farm['tiles'][pos[1]][pos[0]]
  if op=='PICKUP':
   x=a[1];av=int(o['private']['shed'].get(x,0))-res['pickup'].get(x,0)
   if pos not in SHED_ACCESS or av<=0:return None
   n=min(int(a[2]),av);res['pickup'][x]=res['pickup'].get(x,0)+n;return ['PICKUP',x,n]
  if op=='HARVEST':return a if isinstance(t,dict) and int(t.get('yield_units',0))>0 else None
  if op=='COLLECT_FERTILIZER':return a if is_animal(t) and t.get('fertilizer_available',False) else None
  if op=='FEED':
   av=int(inv.get('WHEAT',0))-res['feed'].get(u,0)
   if not is_animal(t) or t.get('fed_today',False) or av<=0:return None
   res['feed'][u]=res['feed'].get(u,0)+1;return a
  if op=='CARE':return a if is_animal(t) and not t.get('cared_today',False) else None
  if op=='DIG':return a if t is not None and not is_animal(t) else None
  if op=='BUILD_COOP':return a if t is None else None
  if op=='BUILD_PASTURE':return a if t is None else None
  if op=='PLACE':
   x=a[1];av=int(inv.get(x,0))-res['place'].get((u,x),0)
   if av<=0 or not isinstance(t,dict) or t.get('kind')!=ANIMALS[x]['structure'] or 'animal' in t:return None
   res['place'][(u,x)]=res['place'].get((u,x),0)+1;return a
  if op=='PLANT':
   x=a[1];av=int(o['private']['seeds'].get(x,0))-res['plant'].get(x,0)
   if t is not None or av<=0:return None
   res['plant'][x]=res['plant'].get(x,0)+1;return a
  if op=='WATER':return a if is_plant(t) and not t.get('watered_today',False) else None
  if op=='FERTILIZE':
   av=int(inv.get('FERTILIZER',0))-res['fert'].get(u,0)
   if not is_plant(t) or av<=0:return None
   res['fert'][u]=res['fert'].get(u,0)+1;return a
  if op=='DROP':return a if pos in SHED_ACCESS and bool(inv) else None
  return a

 def unit_action(self,o,u,res):
  p=self.plans.get(u,[]);tg=self.targets.get(u,[]);i=self.pi.get(u,0)
  while i<len(p):
   if self.p.fix_logistics and p[i][0]=='DROP':
    positions=[self.farm(o)['farmer']]+self.farm(o)['hands']
    if tuple(positions[u]) in SHED_ACCESS:
     qty=sum(o['private']['inventories'][u].values())
     room=100-sum(o['private']['shed'].values())+sum(res['pickup'].values())-res.get('drop_qty',0)
     if qty>room:return ['PASS'] # keep pointer; this turn's market may clear room
     res['drop_qty']=res.get('drop_qty',0)+qty
   a=self.valid(o,u,p[i],tg[i],res);i+=1;self.pi[u]=i
   if a is not None:return a
  return ['PASS']

 def market_work(self,o):
  extra={}
  # On a terminal DROP turn, orders may safely include the inventory that unit
  # actions deposit before market processing.
  if self.day>=29:
   farm=self.farm(o)
   for u,pl in self.plans.items():
    i=self.pi.get(u,0)
    if i<len(pl) and pl[i][0]=='DROP':
     for x,n in o['private']['inventories'][u].items():extra[x]=extra.get(x,0)+n
   return self.sale_orders(o,terminal=True,extra_drop=extra)[:10]
  if int(o.get('hour',0))>=23:return self.capacity_sales(o)[:10]
  if self.p.fix_logistics:
   incoming=0
   for u,pl in self.plans.items():
    i=self.pi.get(u,0)
    if i<len(pl) and pl[i][0]=='DROP':incoming+=sum(o['private']['inventories'][u].values())
   if incoming>0:return self.capacity_sales(o,incoming=incoming)[:10]
  sell=self.competition_sales(o)
  return [['SELL',x,n] for x,n in sorted(sell.items(),key=lambda z:o['market']['prices'][z[0]],reverse=True) if n>0][:10]

 def feed_shortfall(self,o):
  if self.day>=29:return 0
  need=sum(1 for row in self.farm(o)['tiles'] for t in row if is_animal(t) and not t.get('fed_today',False))
  stock=int(o['private']['shed'].get('WHEAT',0))+sum(int(i.get('WHEAT',0)) for i in o['private']['inventories'])
  return max(0,need-stock)

 def liquidity_recovery(self,o):
  if not self.p.fix_liquidity or self.phase!='COMPILE_NEXT':return None
  shortage=self.feed_shortfall(o)
  if shortage<=0:return None
  farm=self.farm(o);positions=[farm['farmer']]+farm['hands']
  acts=[['PASS'] for _ in positions];deposit={}
  room=max(0,100-sum(o['private']['shed'].values()))
  for u,pos in enumerate(positions):
   inv=o['private']['inventories'][u];qty=sum(inv.values())
   if tuple(pos) in SHED_ACCESS and inv.get('FERTILIZER',0)>0 and 0<qty<=room:
    acts[u]=['DROP'];room-=qty
    for item,n in inv.items():deposit[item]=deposit.get(item,0)+n
  wheat=int(o['market']['inventory']['WHEAT']);cash=float(farm['money'])
  fert=max(0,int(o['private']['shed'].get('FERTILIZER',0))+deposit.get('FERTILIZER',0)-int(self.debug.get('daily_need',{}).get('FERTILIZER',0)))
  sell=0;finv=int(o['market']['inventory']['FERTILIZER'])
  desired=sum(market_price('WHEAT',wheat-i-1) for i in range(shortage))
  while cash<desired and sell<fert:
   cash+=market_price('FERTILIZER',finv+sell);sell+=1
  buy=0;spent=0
  while buy<shortage:
   price=market_price('WHEAT',wheat-buy-1)
   if spent+price>cash:break
   spent+=price;buy+=1
  if buy<=0:return None
  market=([['SELL','FERTILIZER',sell]] if sell else [])+[['BUY_PRODUCT','WHEAT',buy]]
  self.debug['liquidity_recoveries']=self.debug.get('liquidity_recoveries',0)+1
  # Wait for the actual fill, then compile from real resources on the next step.
  return {'farmer':acts[0],'hands':acts[1:],'market':market}

 def act(self,o,configuration=None):
  step=int(o.get('step',0));day=int(o.get('day',0))
  if step<=self.last_step or day<self.day:self.reset()
  self.last_step=step
  if day!=self.day:self.new_day(o)
  recovery=self.liquidity_recovery(o)
  if recovery is not None:return recovery
  market=[]
  if self.phase=='SETUP':
   market=self.queue[:10];self.queue=self.queue[10:]
   if not self.queue:self.phase='COMPILE_NEXT'
  elif self.phase=='COMPILE_NEXT':
   self.compile(o);market=self.market_work(o)
  else:market=self.market_work(o)
  n=1+len(self.farm(o)['hands']);res={'pickup':{},'feed':{},'place':{},'plant':{},'fert':{}}
  acts=[self.unit_action(o,u,res) for u in range(n)] if self.phase=='WORK' else [['PASS'] for _ in range(n)]
  if self.p.fix_liquidity and self.phase!='WORK' and self.feed_shortfall(o)>0:
   positions=[self.farm(o)['farmer']]+self.farm(o)['hands']
   for u,pos in enumerate(positions):
    t=self.tile(o,tuple(pos))
    if tuple(pos) in SHED_ACCESS and is_animal(t) and t.get('fertilizer_available',False):acts[u]=['COLLECT_FERTILIZER']
  return {'farmer':acts[0] if acts else ['PASS'],'hands':acts[1:],'market':market}

_CONTROLLERS={}
def agent(obs,configuration=None):
 p=int(obs.get('player',0));c=_CONTROLLERS.get(p)
 if c is None:c=DailyDPController();_CONTROLLERS[p]=c
 return c.act(obs,configuration)
