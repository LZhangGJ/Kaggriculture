"""Forensic own-unit audit. Market attribution is labelled, never unseen rival reconstruction."""
from pathlib import Path
import sys,json,gzip,copy,collections
w=Path(__file__).resolve().parents[1];sys.path.insert(0,str(w/'feedback/referee'))
from cpu_runtime import load_engine,AttrDict
engine=load_engine();summaries=[]
for path in sorted((w/'feedback/own_traces').glob('*/*.gz')):
 tr=json.load(gzip.open(path,'rt'));seat=tr['seat'];daily=[];events=[];total=collections.Counter();buys=[];prod=[];market_rows=[];observations=tr['observations']
 for step,o in enumerate(observations[:719]):
  a=tr['own_actions'][step];nxt=observations[step+1];farm=copy.deepcopy(o['farms'][seat]);pr=copy.deepcopy(o['private']);phase={}
  units=[a['farmer']]+a['hands'];demand=collections.Counter(x[1] for x in units if x and x[0]=='PLANT');blocked={k for k,v in demand.items() if v>pr['seeds'].get(k,0)}
  for uid,act in enumerate(units):
   if act[0]=='PLANT' and act[1] in blocked:act=['PASS']
   pos=copy.deepcopy(farm['farmer'] if uid==0 else farm['hands'][uid-1]);x,y=pos;tile=copy.deepcopy(farm['tiles'][y][x]);before=copy.deepcopy(pr['inventories'][uid]);engine._apply_unit_action(farm,pr,uid,act,10,step//24,24,100);after=pr['inventories'][uid]
   if act[0] in ('FERTILIZE','HARVEST','COLLECT_FERTILIZER'):
    deltas={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after)}
    event={'step':step,'unit':uid,'position':pos,'action':act,'tile_before':tile,'inventory_delta':deltas};events.append(event)
    if act[0]=='FERTILIZE':total['fertilizer_success']+=deltas.get('FERTILIZER',0)==-1;total['fertilizer_failed']+=deltas.get('FERTILIZER',0)!=-1
    if act[0]=='HARVEST':
     for k,q in deltas.items():
      if q>0:total['harvest_'+k]+=q
  # One-step own-only market quotation. This is NOT the opponent's order tape.
  fs=copy.deepcopy(o['farms']);fs[seat]=farm;other=1-seat
  privs=[engine._new_private(),engine._new_private()];privs[seat]=pr
  state=[AttrDict(observation=AttrDict(farms=fs,market=copy.deepcopy(o['market']),private=privs[i]),action=(a if i==seat else {'market':[]})) for i in range(2)]
  for x in state:x.observation.market=state[0].observation.market
  env=AttrDict(configuration=AttrDict(tr['configuration']));transactions=[];orig=engine._commit_unit
  def commit(op,item,price,ff,pp,mm,shed_capacity=100):
   cash=ff['money'];ok=orig(op,item,price,ff,pp,mm,shed_capacity)
   if ff is fs[seat] and ok:transactions.append({'op':op,'item':item,'price':price,'delta':ff['money']-cash})
   return ok
  engine._commit_unit=commit;engine._process_market(state,env);engine._commit_unit=orig
  quote=sum(x['delta'] for x in transactions);actual_delta=nxt['farms'][seat]['money']-o['farms'][seat]['money'];reconciles=abs(fs[seat]['money']-nxt['farms'][seat]['money'])<1e-6
  market_rows.append({'step':step,'actual_cash_delta':actual_delta,'own_only_quoted_cash_delta':fs[seat]['money']-o['farms'][seat]['money'],'cash_reconciles':reconciles,'quoted_transactions':transactions})
  if any(x[:2]==['BUY_PRODUCT','FERTILIZER'] for x in a['market']):
   f=[x for x in transactions if x['op']=='BUY_PRODUCT' and x['item']=='FERTILIZER'];entry={'step':step,'requested':sum(x[2] for x in a['market'] if x[:2]==['BUY_PRODUCT','FERTILIZER']),'quoted_filled':len(f),'quoted_fertilizer_cost':sum(x['price'] for x in f),'whole_step_cash_reconciles':reconciles,'actual_whole_step_cash_delta':actual_delta,'other_orders':a['market']};buys.append(entry)
  # Successful production refresh is checkable from own-visible states alone.
  if step%24==23:
   engine._decay_plants(farm,step);before=copy.deepcopy(farm['tiles']);engine._daily_refresh_plants(farm,step//24,24)
   for y in range(10):
    for x in range(10):
     t=before[y][x];end=nxt['farms'][seat]['tiles'][y][x]
     if isinstance(t,dict) and t.get('kind')=='PLANT':
      assert farm['tiles'][y][x]==end,(path.name,step,x,y,'plant refresh')
      if t.get('crop') in ('STRAWBERRY','TOMATO'):
       cd=engine.CROPS[t['crop']];ds=step//24+1-t['planted_day']-cd['first_yield_day']
       if ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<cd['max_yield']:
        q=max(0,end.get('yield_units',0)-t['yield_units']) if isinstance(end,dict) and end.get('kind')=='PLANT' else 0
        prod.append({'day':step//24,'pos':[x,y],'crop':t['crop'],'birth':t['planted_day'],'produced':q,'watered':t['watered_today'],'fertilizer_until':t.get('fertilized_until_day',-1),'yield_before_refresh':t['yield_units']});total['produced_'+t['crop']]+=q
 # quantity of sales per nonbuyable item inferred from full lifetime own output,
 # with terminal inventory/discard treated explicitly rather than presumed sold.
 result={'trace':str(path.relative_to(w)),'case':tr['game_id'],'version':tr['candidate'],'source_replay_sha256':tr['source_replay_sha256'],'actual_terminal_cash':observations[-1]['farms'][seat]['money'],'actual_rival_cash':observations[-1]['farms'][1-seat]['money'],'counts':dict(total),'fertilizer_buys':buys,'production':prod,'unit_events':events,'market_quotations':market_rows,'warning':'Per-item monetary attribution is a same-observation own-only quote; the complete actual own cash delta is observed. Whole-step equality is a reconciliation, not proof of unique unobserved opponent orders.'}
 out=w/'logs'/('own_audit_'+path.parent.name+'_'+tr['game_id']+'.json.gz');out.write_bytes(gzip.compress(json.dumps(result).encode()));summary={k:result[k] for k in ['trace','case','version','actual_terminal_cash','actual_rival_cash','counts','fertilizer_buys']};summaries.append(summary)
 print(path.parent.name,tr['game_id'],dict(total),'late_f_buy',[(x['step'],x['quoted_filled'],x['quoted_fertilizer_cost'],x['whole_step_cash_reconciles']) for x in buys if x['step']>=360],flush=True)
(w/'logs/own_feedback_audit_summary.json').write_text(json.dumps(summaries,indent=2))
