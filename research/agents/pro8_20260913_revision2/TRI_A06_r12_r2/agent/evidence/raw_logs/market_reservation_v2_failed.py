"""Official atomic-market fixtures, not matches or replay counterfactuals.

Starts from one actually visible own snapshot. Public inventory shocks below
are explicitly synthetic stress inputs, not inferred opponent orders. Only
this farm/private is passed to official commit/hire functions: no opponent
private state, random generator, game runner or saved future is used.
"""
from pathlib import Path
import argparse,copy,gzip,importlib.util,json,sys
p=argparse.ArgumentParser();p.add_argument('--feedback',type=Path,required=True);p.add_argument('--unit-json',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
s=importlib.util.spec_from_file_location('atomic_runtime',a.feedback/'referee/cpu_runtime.py');runtime=importlib.util.module_from_spec(s);s.loader.exec_module(runtime);e=runtime.load_engine()
case='submission_56149565_852484341_seat0';tr=json.load(gzip.open(a.feedback/'own_traces'/f'{case}.json.gz','rt'));obs=tr['observations'][216]
q=json.loads(a.unit_json.read_text());assert q['failed']==0
original=q['original_orders'];protected=q['protected_orders']
# Exact first batch emitted by the parent on the selected observed decision.
assert original[:10]==tr['own_actions'][216]['market']
checks=0

def settle(orders,shock):
 farm=copy.deepcopy(obs['farms'][obs['player']]);private=copy.deepcopy(obs['private']);market=copy.deepcopy(obs['market'])
 for item,delta in shock.items():market['inventory'][item]+=delta
 e._refresh_prices(market)
 events=[];first=None
 for i,order in enumerate(orders):
  before=farm['money'];state=e._parse_order(order);op=state['type'];units=0;hire0=len(farm['hands']);shed0=copy.deepcopy(private['shed'])
  if op=='HIRE':e._do_hire(farm,private,10);units=len(farm['hands'])-hire0
  elif op=='BUY_LAND':
   old=len(farm['unlocked_quadrants']);e._do_buy_land(farm,10);units=len(farm['unlocked_quadrants'])-old
  else:
   item=state['item']
   for _ in range(state['remaining']):
    if op in ('SELL','BUY_PRODUCT'):price=e.market_price(item,market['inventory'][item]-(op=='BUY_PRODUCT'),market.get('params'))
    elif op=='BUY_SEED':price=e.CROPS[item]['seed']
    elif op=='BUY_ANIMAL':price=e.ANIMALS[item]['cost']
    else:raise AssertionError(op)
    if not e._commit_unit(op,item,price,farm,private,market,100):break
    units+=1
  e._refresh_prices(market);assert farm['money']>=0 and sum(private['shed'].values())<=100
  events.append({'index':i,'batch':i//10,'order':order,'filled':units,'cash_before':before,'cash_after':farm['money'],'hands_after':len(farm['hands'])})
  if i==9:first={'cash':farm['money'],'hands':len(farm['hands'])}
 return {'cash':farm['money'],'hands':len(farm['hands']),'first_batch':first,'shed':private['shed'],'seeds':private['seeds'],'events':events}

rows=[];improved=[]
for delta in [0,1,2,3,5,8,10,13,20,30,50,80,120,200]:
 shock={'MILK':delta,'FERTILIZER':delta//2}
 old,new=settle(original,shock),settle(protected,shock)
 # Same buys/sales within an unchanging synthetic market; if money is short,
 # capex cannot take wages already reserved by the policy.
 assert new['hands']>=old['hands'];checks+=1
 assert new['events'][2]['filled']==old['events'][2]['filled'];checks+=1
 if new['hands']>old['hands']:improved.append(delta)
 rows.append({'synthetic_public_inventory_additions':shock,'original':old,'protected':new})
assert improved;checks+=1
assert rows[0]['original']['hands']==7==rows[0]['protected']['hands'];checks+=1
# Negative example: no queue ordering can guarantee all hires when even feed
# and wages cannot be funded at actual executable quotes. Never print a fake
# reserve or borrow from future strawberries.
stressed={'MILK':200,'FERTILIZER':100,'WHEAT':-650}
negative={'shock':stressed,'original':settle(original,stressed),'protected':settle(protected,stressed)}
assert negative['protected']['hands']<7;checks+=1
assert negative['protected']['cash']>=0;checks+=1
result={'kind':'single-own-farm official atomic settlements; no elapsed game and no opponent orders','source_case':case,'source_step':216,'assertions':checks,'new_games':0,'order_source':'compiled production preparation_with_wage_reserve helper via unit executable','synthetic_scenarios':rows,'insufficient_feed_and_wage_cash_negative':negative,'hire_improvement_synthetic_milk_deltas':improved}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
print(json.dumps({'assertions':checks,'scenarios':len(rows)+1,'improvement_deltas':improved,'no_shock_hires':[rows[0][k]['hands'] for k in ('original','protected')],'extreme_negative_hires':negative['protected']['hands'],'new_games':0}))
