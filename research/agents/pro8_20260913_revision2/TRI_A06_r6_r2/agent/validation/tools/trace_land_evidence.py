"""Audit only supplied own-visible histories. No new opponents or episodes.
Transactions and terminal results retain the complete supplied denominator.
Unit-phase events use the unchanged official primitive on the current observation;
no future random state is generated. Fungible goods are NOT attributed to land cash.
"""
import argparse,collections,copy,gzip,importlib.util,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--audits',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('own_unit_official',a.input/'referee/cpu_runtime.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);engine=runtime.load_engine()
quad=lambda x,y:('NW','NE','SW','SE')[(x>=5)+2*(y>=5)]
kind=lambda t:t.get('crop',t.get('animal',t.get('kind','EMPTY'))) if isinstance(t,dict) else t
productive=lambda t:isinstance(t,dict) and ('crop' in t or 'animal' in t)
rows=json.loads((a.input/'FULL64_ROWS.json').read_text());assert len(rows)==1536 and len({x['id'] for x in rows})==1536
counts=lambda rr:{'games':len(rr),'wins':sum(x['margin']>0 for x in rr),'ties':sum(x['margin']==0 for x in rr),'losses':sum(x['margin']<0 for x in rr),'all_719':all(x['steps']==719 for x in rr),'all_terminal':all(x['terminal'] for x in rr),'errors':sum(x['error'] is not None for x in rr)}
full={'scope':'Supplied r1 historical full64, NOT new r2 results','overall':counts(rows),'R2':counts([x for x in rows if x['opponent']=='submission_56149565']),'public11':counts([x for x in rows if x['opponent']!='submission_56149565']),'by_opponent':{o:counts([x for x in rows if x['opponent']==o]) for o in sorted({x['opponent'] for x in rows})},'development_seeds':sorted({x['seed'] for x in rows}),'new_games':0}
assert full['overall']['wins']==1196 and full['R2']['wins']==118 and full['public11']['wins']==1078
(a.out/'FULL64_RECOUNT.json').write_text(json.dumps(full,indent=2)+'\n');summary=[];started=time.perf_counter()
for f in sorted((a.input/'own_traces').glob('*.gz')):
 t=json.load(gzip.open(f,'rt'));obs=t['observations'];acts=t['own_actions'];seat=t['seat'];name=t['game_id'];assert len(obs)==720 and len(acts)==719
 ledger=json.loads((a.input/'own_ledgers'/f'{name}.json').read_text())['own_player'];days=ledger['days'];cashchecks=[]
 for d in days:
  cashchecks.append({'day':d['day'],'start':d['cash_start'],'end':d['cash_end'],'sum_flows':sum(d['flow'].values()),'residual':d['cash_end']-d['cash_start']-sum(d['flow'].values())})
 assert len(days)==30 and all(x['residual']==0 for x in cashchecks)
 assert days[0]['cash_start']==obs[0]['farms'][seat]['money'] and days[-1]['cash_end']==obs[-1]['farms'][seat]['money']==ledger['cash']
 unit_events=[];maturation=[];purchases=[];audit={x['step']:x for x in json.loads((a.audits/f'{name}.json').read_text())}
 for step,(ob,act) in enumerate(zip(obs,acts)):
  original=ob['farms'][seat];farm=copy.deepcopy(original);priv=copy.deepcopy(ob['private']);nextfarm=obs[step+1]['farms'][seat]
  extra=[q for q in nextfarm['unlocked_quadrants'] if q not in farm['unlocked_quadrants']]
  if extra:
   assert len(extra)==1 and ['BUY_LAND'] in act['market'];day=ob['day'];parent=audit[step];la=parent['land_audit'];cands=parent['debug']['last_search']['candidates']
   purchases.append({'step':step,'day':day,'quadrant':extra[0],'cost':-days[day]['flow'].get('_do_buy_land',0),'cash_before':original['money'],'selected_initial_projects':[{'pos':x['pos'],'kind':x['kind'],'path_length':x['length'],'successor':x['successor']} for x in la['targets'] if x['locked'] and x['kind']>=0],'original_candidates':len(cands),'original_candidates_predicting_land':sum(any(o[0]==19 for o in c['first_prediction_market']) for c in cands),'conditional_remove_initial_bundle_model_delta':la['value_with_land']-la['value_strip_newland_paths_and_refund'],'model_delta_scope':'Conditional diagnostic only: removing initial new plots plus refunding land. Not a cash recovery, marginal realized profit, or full future rolling policy.'})
  units=[act['farmer'],*act['hands']];counts=collections.Counter(x[1] for x in units if x[0]=='PLANT');blocked={c for c,n in counts.items() if n>priv['seeds'].get(c,0)}
  for u,action in enumerate(units):
   pos=farm['farmer'] if u==0 else farm['hands'][u-1];x,y=pos;tile_before=copy.deepcopy(farm['tiles'][y][x]);bag_before=priv['inventories'][u].copy();shed_before=priv['shed'].copy();seeds_before=priv['seeds'].copy()
   if not(action[0]=='PLANT' and action[1] in blocked):engine._apply_unit_action(farm,priv,u,action,10,ob['day'],24,t['configuration'].get('shedCapacity',100))
   tile_after=farm['tiles'][y][x];bag_after=priv['inventories'][u];e={'step':step,'day':ob['day'],'unit':u,'pos':y*10+x,'quadrant':quad(x,y),'action':action}
   if action[0]=='PLANT' and productive(tile_after) and priv['seeds']!=seeds_before:e.update(type='planted',crop=tile_after['crop'])
   elif action[0]=='PLACE' and productive(tile_after) and not productive(tile_before):e.update(type='placed_animal',animal=tile_after.get('animal'))
   elif action[0]=='HARVEST':
    gained={k:v-bag_before.get(k,0) for k,v in bag_after.items() if v>bag_before.get(k,0)}
    if gained:e.update(type='harvested',goods=gained)
   elif action[0] in ('WATER','FEED','CARE','FERTILIZE','COLLECT_FERTILIZER') and tile_after!=tile_before:e.update(type='maintenance_effect',operation=action[0],tile_kind=kind(tile_before))
   elif action[0] in ('DROP','PLACE'):
    gained={k:v-shed_before.get(k,0) for k,v in priv['shed'].items() if v>shed_before.get(k,0)}
    if gained:e.update(type='warehouse_arrival',goods=gained,source_land='Not attributable: carried and warehouse goods are fungible')
   if 'type' in e:unit_events.append(e)
  # The next own-visible frame is used only as historical evidence. Never
  # reuse it as the result of a candidate action or of this unit projection.
  for y in range(10):
   for x in range(10):
    old=farm['tiles'][y][x];new=nextfarm['tiles'][y][x]
    if isinstance(new,dict) and new.get('kind')=='PLANT' and isinstance(old,dict) and old.get('kind')=='PLANT' and (old.get('crop'),old.get('planted_day'))==(new.get('crop'),new.get('planted_day')) and new.get('yield_units',0)>old.get('yield_units',0):
     maturation.append({'observed_at_step':step+1,'pos':y*10+x,'quadrant':quad(x,y),'crop':new['crop'],'new_visible_yield':new['yield_units']-old.get('yield_units',0)})
 for buy in purchases:
  q=buy['quadrant'];start=buy['step'];events=[x for x in unit_events if x['step']>=start and x['quadrant']==q];used=[x for x in events if x['type'] in ('planted','placed_animal')]
  harvest=collections.Counter();plants=collections.Counter();care=collections.Counter()
  for e in events:
   if e['type']=='harvested':harvest.update(e['goods'])
   if e['type']=='planted':plants[e['crop']]+=1
   if e['type']=='placed_animal':plants[e['animal']]+=1
   if e['type']=='maintenance_effect':care[e['operation']]+=1
  occupied=[]
  for ob in obs[start+1:]:occupied.append(sum(productive(ti) for y,row in enumerate(ob['farms'][seat]['tiles']) for x,ti in enumerate(row) if quad(x,y)==q))
  buy.update(first_productive_event=used[0] if used else None,same_day_used=bool(used and used[0]['day']==buy['day']),max_productive_occupancy=max(occupied),successful_plant_or_animal_placements=dict(plants),actual_harvest_units=dict(harvest),actual_maintenance_effects=dict(care),visible_maturations=[x for x in maturation if x['quadrant']==q and x['observed_at_step']>start])
  buy['warehouse_delivery_scope']='Harvest origin is observed per plot. Direct warehouse arrivals and ledger sale proceeds are whole-farm only; no plot-specific realized net profit is inferred.'
  buy['whole_farm_post_purchase_sell_cash']=sum(v for d in days if d['day']>=buy['day'] for k,v in d['flow'].items() if k.startswith('SELL:'))
  buy['whole_farm_post_purchase_direct_warehouse_arrivals']=dict(sum((collections.Counter(e['goods']) for e in unit_events if e['step']>=start and e['type']=='warehouse_arrival'),collections.Counter()))
 report={'game_id':name,'scope':'Historical r1 own-visible facts only. No r2 closed-loop episodes.','purchases':purchases,'cash_conservation':cashchecks,'all_unit_events':unit_events,'visible_maturation_events':maturation,'new_games':0}
 (a.out/f'{name}.json').write_text(json.dumps(report,separators=(',',':'))+'\n');summary.append({'id':name,'purchases':len(purchases),'land_spend':sum(x['cost'] for x in purchases),'all_same_day_used':all(x['same_day_used'] for x in purchases),'cash_days_checked':len(cashchecks),'unit_events':len(unit_events),'last_purchase':{k:v for k,v in purchases[-1].items() if k not in ('visible_maturations',)}});print(name,'cash days',len(cashchecks),'land',[(x['day'],x['quadrant'],x['first_productive_event']['step']) for x in purchases],flush=True)
(a.out/'SUMMARY.json').write_text(json.dumps({'cases':summary,'total_land_purchases':sum(x['purchases'] for x in summary),'all_same_day_used':all(x['all_same_day_used'] for x in summary),'cash_days_checked':sum(x['cash_days_checked'] for x in summary),'seconds':time.perf_counter()-started,'new_games':0},indent=2)+'\n')
