#!/usr/bin/env python3
"""Actual-native route constraints + frozen official unit-phase verification.
No new games, opponent private state or saved post-divergence futures are used.
"""
import argparse,ctypes,copy,gzip,hashlib,importlib.util,json,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
codec=load(ROOT/'policy/agent.py','delivery_codec')
OPS=codec._OPS;ITEMS=codec._ITEMS
MOVES={1:(0,-1),2:(0,1),3:(1,0),4:(-1,0)}
def atom(a):
 op,i,q,_=a;out=[OPS[op]]
 if i>=0:
  out.append(ITEMS[i])
  if op in (6,7,20,21,22,23):out.append(q)
 return out
def dist(a,b):return abs(a%10-b%10)+abs(a//10-b//10)
def walk(pos,d):
 out=[]
 while pos%10!=d%10:
  op=3 if pos%10<d%10 else 4;out.append([op,-1,1,-1]);pos+=1 if op==3 else -1
 while pos//10!=d//10:
  op=2 if pos//10<d//10 else 1;out.append([op,-1,1,-1]);pos+=10 if op==2 else -10
 return out

def field_order(plans):
 events=collections.defaultdict(list)
 for u,p in enumerate(plans):
  for k,(op,i,q,pos) in enumerate(p):
   if pos>=0 and op not in (*MOVES,5,6):events[pos].append((k,u,op,i,q))
 return {pos:[tuple(e[2:]) for e in sorted(es)] for pos,es in events.items()}

def product_count(farm,private):
 out=collections.Counter(private['shed'])
 for bag in private['inventories']:out.update(bag)
 return +out

def units_only(engine,o,plans):
 """Official own unit operations, no market, until pre-refresh day endpoint."""
 farm=copy.deepcopy(o['farms'][o['player']]);p=copy.deepcopy(o['private']);events=[];effects=collections.Counter();before=product_count(farm,p)
 for k in range(24-o['hour']):
  acts=[atom(pl[k]) if k<len(pl) else ['PASS'] for pl in plans]
  demand=collections.Counter(a[1] for a in acts if a[0]=='PLANT')
  for u,a in enumerate(acts):
   pos=farm['farmer'] if u==0 else farm['hands'][u-1];cell=pos[1]*10+pos[0]
   if k<len(plans[u]) and plans[u][k][3]>=0:assert cell==plans[u][k][3],('wrong target',k,u,cell,plans[u][k])
   if a[0]=='PLANT' and demand[a[1]]>p['seeds'].get(a[1],0):a=['PASS']
   inv0=copy.deepcopy(p['inventories'][u]);shed0=sum(p['shed'].values());tile0=copy.deepcopy(farm['tiles'][pos[1]][pos[0]])
   engine._apply_unit_action(farm,p,u,a,10,o['day'],24,100)
   if a[0] not in ('PASS','NORTH','SOUTH','EAST','WEST','DROP','PICKUP') and (inv0!=p['inventories'][u] or tile0!=farm['tiles'][pos[1]][pos[0]]):effects[(cell,tuple(a))]+=1
   if a[0]=='DROP':events.append({'tick':o['step']+k,'u':u,'cargo_before':inv0,'deposited':sum(p['shed'].values())-shed0,'discarded':sum(inv0.values())-(sum(p['shed'].values())-shed0)})
  engine._decay_plants(farm,o['step']+k)
 return {'farm':farm,'private':p,'counts':product_count(farm,p),'drops':events,'effects':effects}

def synthetic(engine,hour=5,cargo=10,reflection=0):
 day=20
 def mirror(c):
  x,y=c%10,c//10
  if reflection&1:x=9-x
  if reflection&2:y=9-y
  return y*10+x
 def xy(c):c=mirror(c);return [c%10,c//10]
 f={'money':50000,'farmer':xy(46),'hands':[xy(1)],'unlocked_quadrants':['NW','NE','SW','SE'],'hires_today':1,'tiles':[[None]*10 for _ in range(10)]}
 for pos,t in [(49,engine._new_animal('COW',10)),(1,engine._new_animal('COW',10)),(9,engine._new_plant('STRAWBERRY',0,24)),(0,engine._new_plant('STRAWBERRY',0,24))]:
  q=mirror(pos);t['max_lifespan_step']=10000;t['yield_units']=0;f['tiles'][q//10][q%10]=t
 # Background cargo is real carried inventory, not a price/seed lookup.
 priv={'shed':{},'seeds':{},'inventories':[{'MILK':cargo},{'EGG':95,'WHEAT':1}]}
 market={'inventory':{i:10000 for i in ITEMS[:9]},'prices':{i:engine.market_price(i,10000) for i in ITEMS[:9]}}
 o={'step':day*24+hour,'day':day,'hour':hour,'player':0,'farms':[f,copy.deepcopy(f)],'private':priv,'market':market,'town':{'unlocked_shops':[]}}
 pp=[];cur=mirror(46)
 for target,op in [(49,17),(9,9),(0,9)]:
  d=mirror(target);pp+=walk(cur,d)+[[op,-1,1,d]];cur=d
 return o,[pp,[[15,-1,1,mirror(1)]]]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,default=ROOT/'policy/tri_a08_r11_r3.so');ap.add_argument('--referee',type=Path,default=ROOT/'evidence/feedback/referee');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 runtime=load(a.referee/'cpu_runtime.py','delivery_official');engine=runtime.load_engine();lib=ctypes.CDLL(str(a.library.resolve()))
 f=lib.td_delivery_proposal_snapshot;f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,ctypes.c_double,ctypes.c_int];f.restype=ctypes.c_char_p
 count=0;records=[];positives=0;negatives=[]
 def check(v,msg):
  nonlocal count
  count+=1
  if not v:raise AssertionError(msg)
 def proposal(o,plans,shadow=4,authorize=1):
  raw=[len(plans)]
  for p in plans:raw += [len(p)]+[x for act in p for x in act]
  pack=codec._pack(o);rr=(ctypes.c_int32*len(raw))(*raw);r=json.loads(f(pack,len(pack),rr,len(rr),shadow,authorize))
  check('error' not in r,r);return r
 def validate(o,plans,r,tag):
  nonlocal positives
  if not r['selected']:return
  positives+=1;best=r['best'];u=r['donor'];v=r['receiver']
  check(u!=v,(tag,'donor receiver'))
  qty=sum(o['private']['inventories'][u].values());check(qty==r['quantity'],(tag,'chosen cargo/quantity consistency'))
  extra=max(1,len(best[u])+len(best[v])-len(plans[u])-len(plans[v]));check(extra==r['extra_actions'],(tag,'chosen route/cost consistency'))
  quote=0
  for item in ITEMS[1:8]:
   inv=o['market']['inventory'][item]
   for _ in range(o['private']['inventories'][u].get(item,0)):
    price=engine.market_price(item,inv);quote+=price
    if price>1:inv+=1
  quote*=min(qty,r['overflow_before'])/qty
  check(abs(quote-r['conditional_quote'])<1e-7,(tag,'chosen route/quote consistency'))
  check(abs((quote-4*extra)/extra-r['score'])<1e-7,(tag,'score from chosen route, not another option'))
  check(field_order(plans)==field_order(best),(tag,'field order'))
  check(all(len(p)<=24-o['hour'] for p in best),(tag,'day deadline'))
  for j in range(len(plans)):
   if j not in (u,v):check(plans[j]==best[j],(tag,'unrelated plan changed'))
  old=units_only(engine,o,plans);new=units_only(engine,o,best)
  check(old['farm']['tiles']==new['farm']['tiles'],(tag,'field endpoint mismatch'))
  check(old['private']['seeds']==new['private']['seeds'],(tag,'seed consumption'))
  check(old['effects']==new['effects'],(tag,'actually successful field operations'))
  check(old['counts']==new['counts'],(tag,'inventory conservation',old['counts'],new['counts']))
  deposits=[d for d in new['drops'] if d['u']==u and d['tick']==o['step']+r['arrival']]
  check(len(deposits)==1,(tag,'arrival'))
  check(deposits[0]['deposited']==r['quantity'] and deposits[0]['discarded']==0,(tag,'drop capacity',deposits))
  # Forecast salvage is not cash: with no market clearance, total goods are
  # conserved before the night and early DROP alone is not a recovered return.
  def night(state):
   p=copy.deepcopy(state['private']);before=sum(state['counts'].values());engine._drop_inventories_to_shed(p,100);return before-sum(p['shed'].values())
  check(night(old)==night(new),(tag,'no-market negative control'))
  records.append({'tag':tag,**{k:v for k,v in r.items() if k not in ('base','best')},'official_drop':deposits[0],'pre_night_production_maintenance_equal':True,'no_market_loss_base':night(old),'no_market_loss_candidate':night(new)})
 for reflection in range(4):
  for h in (3,4,5):
   for qty in (6,10,15,25):
    o,p=synthetic(engine,h,qty,reflection);r=proposal(o,p);check(r['selected'],('positive synthetic',h,qty,reflection,r));validate(o,p,r,f'synthetic/{reflection}/{h}/{qty}')
 # Recipient still has an actual warehouse pickup and later FEED. The moved
 # resource-free job must be inserted AFTER pickup, preserving input access.
 o,p=synthetic(engine);o['farms'][0]['hands'][0]=[4,4];o['private']['shed']={'WHEAT':1};o['private']['inventories'][1]={'EGG':88};o['farms'][0]['tiles'][0][1]['yield_units']=6
 p[1]=[[6,0,1,44]]+walk(44,1)+[[15,-1,1,1],[10,-1,1,1]]
 r=proposal(o,p);check(r['selected'],('recipient pickup positive',r));validate(o,p,r,'synthetic/recipient_pickup')
 semantics=[a for a in r['best'][r['receiver']] if a[0] not in MOVES]
 check(semantics[0][0]==6,('recipient pickup reordered',semantics))
 o,p=synthetic(engine)
 variants=[]
 q=copy.deepcopy(o);q['private']['inventories'][1]={'WHEAT':1};variants.append(('no_overflow',q,p,4,1))
 variants.append(('not_real_observation',o,p,4,0));variants.append(('negative_opportunity_return',o,p,100000,1))
 q=copy.deepcopy(o);q['private']['shed']={'WHEAT':100};variants.append(('no_free_warehouse_capacity',q,p,4,1))
 q=copy.deepcopy(o);q['private']['inventories'][0]['COW']=1;variants.append(('held_animal_commitment',q,p,4,1))
 # Replace a real existing service with a feasible held-input obligation,
 # rather than extending the day and accidentally testing only the deadline.
 for op,item,target,replaced in [(15,'WHEAT',49,17),(11,'FERTILIZER',9,9)]:
  q=copy.deepcopy(o);q['private']['inventories'][0][item]=1;plans=copy.deepcopy(p)
  for a0 in plans[0]:
   if a0[0]==replaced and a0[3]==target:a0[0]=op;break
  variants.append((f'donor_reserved_input_{op}',q,plans,4,1))
 # The baseline harvest is before expiry; a later detour reaches it too late.
 q=copy.deepcopy(o);plans=copy.deepcopy(p);q['farms'][0]['tiles'][0][9]['yield_units']=1;q['farms'][0]['tiles'][0][9]['max_lifespan_step']=o['step']+10
 for a0 in plans[0]:
  if a0[0]==9 and a0[3]==9:a0[0]=10;break
 variants.append(('delayed_harvest_expiry',q,plans,4,1))
 q=copy.deepcopy(o);q['private']['inventories'][1]['EGG']=90
 for i in ITEMS[:9]:q['market']['inventory'][i]=100000;q['market']['prices'][i]=1
 variants.append(('floor_price_unprofitable_detour',q,p,4,1))
 q=copy.deepcopy(o);q.update(day=29,step=29*24+5);variants.append(('final_day_unchanged',q,p,4,1))
 for name,q,plans,shadow,auth in variants:
  r=proposal(q,plans,shadow,auth);check(not r['selected'],(name,r));negatives.append(name)
 fixtures=json.loads(gzip.decompress((ROOT/'tests/fixtures/delivery_snapshots.json.gz').read_bytes()))
 for fxt in fixtures:
  o=fxt['observation'];p=fxt['plans'];r=proposal(o,p);validate(o,p,r,fxt['case']+'/'+str(fxt['step']))
 report={'status':'PASS','native_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'checks':count,'positive_proposals':positives,'historical_saved_snapshot_calls':len(fixtures),'synthetic_positive_cases':49,'negative_controls':negatives,'results':records,'new_games':0,'scope':'official own-unit prefixes; no market clearing and no competitor rollout; snapshots not counterfactual trajectories'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
if __name__=='__main__':main()
