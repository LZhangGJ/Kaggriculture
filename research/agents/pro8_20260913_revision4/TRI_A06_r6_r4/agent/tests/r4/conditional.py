"""Independent official-rule continuation. No replay future is consulted.
Only own orders; today's observed shops held fixed; no new random weeds/shops.
The returned quotes (when present in native input) are not actual cash.
"""
import copy,ctypes
from collections import Counter
from official_prefix import decode_action,own_signature

def verify(trace, initial, branch, codec, runtime, engine, actual=None, clone=None, capture=False):
 ob=copy.deepcopy(initial);seat=ob['player'];env=runtime.AttrDict(configuration=runtime.AttrDict(trace['configuration']))
 overflow=0;counts=Counter();effects=Counter();entries=[];generated=[];market_cash=Counter();market_units=Counter()
 for row in branch['ticks']:
  assert ob['step']==row['step'];act=decode_action(row['action'],codec)
  if actual:
   predicted=actual(copy.deepcopy(ob),trace['configuration']);assert predicted==act,('real root / audit disagreement',ob['step'],predicted,act);counts['real_root_matches']+=1
  if clone:
   alternate=clone(copy.deepcopy(ob),trace['configuration']);assert alternate==act,('clone / original disagreement',ob['step']);counts['clone_matches']+=1
  if capture:generated.append(copy.deepcopy(ob))
  farm,pr=ob['farms'][seat],ob['private'];units=[act['farmer'],*act['hands']];plants=Counter(x[1] for x in units if x[0]=='PLANT');blocked={c for c,n in plants.items() if n>pr['seeds'].get(c,0)}
  for u,x in enumerate(units):
   if x[0]=='PLANT' and x[1] in blocked:continue
   pos=farm['farmer'] if u==0 else farm['hands'][u-1];col,r=pos;tile=copy.deepcopy(farm['tiles'][r][col]);iv=pr['inventories'][u].copy();shed=pr['shed'].copy()
   engine._apply_unit_action(farm,pr,u,x,10,ob['day'],24,100)
   nt=farm['tiles'][r][col];nv=pr['inventories'][u];n=0
   if x[0] in ('HARVEST','COLLECT_FERTILIZER'):n=max(0,sum(nv.values())-sum(iv.values()))
   elif x[0] in ('FEED','FERTILIZE'):n=max(0,iv.get('WHEAT' if x[0]=='FEED' else 'FERTILIZER',0)-nv.get('WHEAT' if x[0]=='FEED' else 'FERTILIZER',0))
   elif x[0] in ('CARE','WATER') and isinstance(nt,dict):
    key='cared_today' if x[0]=='CARE' else 'watered_today';n=int(not (isinstance(tile,dict) and tile.get(key,False)) and nt.get(key,False))
   counts[x[0]+'_attempts']+=1
   if n:counts[x[0]+'_completed_units']+=n;effects[f'{r*10+col}:{x[0]}']+=n
   if x[0] in ('PLACE','DROP'):
    for i,q in pr['shed'].items():counts['deposited_'+i]+=max(0,q-shed.get(i,0))
  # The official routine is called once per actual order in order. With no
  # opposing orders this is equivalent to its loop, including worker cost growth.
  assert len(act['market'])<=10
  for order in act['market']:
   oldcash=farm['money'];oldstock=pr['shed'].copy();oldhands=len(farm['hands'])
   states=[runtime.AttrDict(observation=runtime.AttrDict(farms=ob['farms'],market=ob['market'],town=ob['town'],private=pr if q==seat else {}),action={'market':[order]} if q==seat else {}) for q in range(2)]
   engine._process_market(states,env)
   market_cash[order[0]]+=farm['money']-oldcash
   if order[0]=='HIRE':counts['HIRES_arrived']+=len(farm['hands'])-oldhands
   if len(order)>1:market_units[order[0]+':'+order[1]]+=abs(pr['shed'].get(order[1],0)-oldstock.get(order[1],0))
  sig=own_signature(ob,codec)
  assert sig==row['prefix_signature'],('official prefix mismatch',ob['step'],[(i,x,y) for i,(x,y) in enumerate(zip(sig,row['prefix_signature'])) if x!=y][:8])
  states=[runtime.AttrDict(observation=runtime.AttrDict(farms=ob['farms'],market=ob['market'],town=ob['town'],private=pr if q==seat else {}),action={}) for q in range(2)]
  engine._town_consume(env,states,ob['step'])
  for f in ob['farms']:engine._decay_plants(f,ob['step'])
  if (ob['step']+1)%24==0:
   before=copy.deepcopy(farm['tiles']);engine._daily_refresh_plants(farm,ob['day'],24);engine._daily_refresh_animals(farm,ob['day'])
   for y,line in enumerate(before):
    for x,t in enumerate(line):
     now=farm['tiles'][y][x]
     if isinstance(t,dict) and t.get('kind')=='PLANT' and not (isinstance(now,dict) and now.get('kind')=='PLANT'):counts['plants_died_at_boundary']+=1
   pre=sum(pr['shed'].values())+sum(sum(iv.values()) for iv in pr['inventories']);engine._drop_inventories_to_shed(pr,100);loss=pre-sum(pr['shed'].values());overflow+=loss;assert loss==row['overflow'];farm['farmer']=[4,4];farm['hands']=[];farm['hires_today']=0;pr['inventories']=[{}]
  assert farm['money']==row['cash'];entries.append({'step':ob['step'],'cash':farm['money'],'pass':True});ob['step']+=1;ob['day']=ob['step']//24;ob['hour']=ob['step']%24
 assert farm['money']==branch['final_cash'] and overflow==branch['overflow']
 return {'prefixes':len(entries),'actual_conditional_cash':farm['money'],'overflow':overflow,'counts':dict(counts),'successful_field_units_by_tile':dict(effects),'actual_cash_by_order':dict(market_cash),'actual_units_by_order':dict(market_units),'rows':entries,'generated_own_observations':generated if capture else [],'final_own_tiles':farm['tiles']}
