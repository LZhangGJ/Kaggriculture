"""Audit recorded own-player transitions only; never a counterfactual game."""
from pathlib import Path
import sys, json, gzip, copy, collections, time
import argparse
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--feedback',type=Path,default=W/'evidence/input');p.add_argument('--out',type=Path,default=W/'build/historical_audit');a=p.parse_args();IN=a.feedback;OUT=a.out;OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(IN/'referee'))
from cpu_runtime import load_engine
E=load_engine(); events=[]; summaries=[]
def ongoing(t): return isinstance(t,dict) and t.get('crop') in ('STRAWBERRY','TOMATO')
def same(a,b): return ongoing(a) and ongoing(b) and (a['crop'],a['planted_day'])==(b['crop'],b['planted_day'])
for case in json.loads((IN/'SELECTED_CASES.json').read_text()):
 trace=json.loads(gzip.decompress((IN/case['trace']).read_bytes())); seat=trace['seat']; obs=trace['observations']; stats=collections.Counter(); lives={}; discrepancies=[]
 for step,(o,act,nxt) in enumerate(zip(obs,trace['own_actions'],obs[1:])):
  f=copy.deepcopy(o['farms'][seat]); private=copy.deepcopy(o['private']); day=o['day']; coords=[f['farmer']]+f['hands']; units=[act['farmer']]+act['hands']
  for u,(xy,a) in enumerate(zip(coords,units)):
   x,y=xy; tile=copy.deepcopy(f['tiles'][y][x]); bag=private['inventories'][u].copy()
   E._apply_unit_action(f,private,u,a,10,day,24,100)
   after=f['tiles'][y][x]
   if ongoing(tile):
    k=tile['crop']; ident=(x,y,k,tile['planted_day']); life=lives.setdefault(ident,collections.Counter())
    event={'case':case['id'],'step':step,'day':day,'pos':[x,y],'crop':k,'birth':tile['planted_day'],'unit':u,'action':a}
    if a[0]=='HARVEST':
     q=private['inventories'][u].get(k,0)-bag.get(k,0); stats[f'{k}_harvest']+=q;life['harvest']+=q
     if q:event['harvest_units']=q;events.append(event)
    elif a[0]=='DIG':
     stats[f'{k}_dig']+=1;life['dig_age']=day-tile['planted_day'];event['yield_destroyed']=tile['yield_units'];event['remaining_ticks']=sum(tile['planted_day']+E.CROPS[k]['first_yield_day']+i*E.CROPS[k]['interval']>day for i in range(4));events.append(event)
     stats[f'{k}_dig_yield']+=tile['yield_units'];stats[f'{k}_dig_future_ticks']+=event['remaining_ticks']
    elif a[0]=='WATER' and not tile['watered_today'] and after['watered_today']:life['water']+=1;stats[f'{k}_water']+=1
    elif a[0]=='FERTILIZE' and private['inventories'][u].get('FERTILIZER',0)<bag.get('FERTILIZER',0):life['fertilize']+=1;stats[f'{k}_fertilize']+=1
   elif a[0]=='PLANT' and ongoing(after):
    k=after['crop'];stats[f'{k}_planted']+=1;lives.setdefault((x,y,k,day),collections.Counter())['plant_step']=step
  # Own units are independent of rival private state and action. Deterministic
  # crop refresh is checked against the next actually delivered own observation.
  E._decay_plants(f,step)
  if (step+1)%24==0:
   before=copy.deepcopy(f['tiles']);E._daily_refresh_plants(f,day,24)
   for y,row in enumerate(before):
    for x,t in enumerate(row):
     if not ongoing(t):continue
     k=t['crop']; cd=E.CROPS[k]; ds=day+1-t['planted_day']-cd['first_yield_day']; tick=ds>=0 and ds%cd['interval']==0 and ds//cd['interval']<4
     new=f['tiles'][y][x];ident=(x,y,k,t['planted_day']);life=lives.setdefault(ident,collections.Counter())
     if not same(t,new):
      stats[f'{k}_drought_death']+=1;life['drought_death']+=1;events.append({'case':case['id'],'step':step,'day':day,'pos':[x,y],'crop':k,'birth':t['planted_day'],'event':'drought_death','yield_lost':t['yield_units'],'dry':t['consecutive_unwatered']});continue
     if tick:
      bonus=int(t['watered_today'] and t['fertilized_until_day']>=day);newq=new['yield_units']-t['yield_units'];stats[f'{k}_production_ticks']+=1;stats[f'{k}_bonus_ticks']+=bonus;stats[f'{k}_produced']+=newq;stats[f'{k}_capacity_loss']+=1+bonus-newq;life['production_ticks']+=1;life['produced']+=newq
      events.append({'case':case['id'],'step':step,'day':day,'pos':[x,y],'crop':k,'birth':t['planted_day'],'event':'production','pre_yield':t['yield_units'],'produced':newq,'bonus':bonus,'watered':t['watered_today'],'fertilized_until_day':t['fertilized_until_day']})
  for y,row in enumerate(f['tiles']):
   for x,t in enumerate(row):
    actual=nxt['farms'][seat]['tiles'][y][x]
    if (ongoing(t) or ongoing(actual)) and t!=actual:discrepancies.append({'step':step,'pos':[x,y],'projected':t,'observed':actual})
 # Actual own ledger records avoid interpreting attempted action as paid cash.
 totals=case['own_totals'];summary={'case':case['id'],'parent_margin':case['row']['margin'],'scope':'historical own-player audit, zero new games','steps':len(trace['own_actions']),'stats':dict(stats),'own_crop_transition_mismatch_count':len(discrepancies),'mismatches':discrepancies[:5],'ledger':{k:v for k,v in totals['flow'].items() if 'STRAWBERRY' in k or 'TOMATO' in k},'lives':[dict(pos=[x,y],crop=k,birth=b,**dict(v)) for (x,y,k,b),v in lives.items()]}
 for k in ('STRAWBERRY','TOMATO'):assert stats[f'{k}_harvest']==totals['units'].get(f'HARVEST:{k}',0),(case['id'],k)
 assert not discrepancies,(case['id'],len(discrepancies),discrepancies[:1])
 summaries.append(summary);print(case['id'],json.dumps(dict(stats)),flush=True)
(OUT/'own_crop_audit.json').write_text(json.dumps(summaries,indent=2))
with gzip.open(OUT/'own_crop_events.jsonl.gz','wt') as z:
 for row in events:z.write(json.dumps(row)+'\n')
print('AUDIT OK',len(summaries),'traces',sum(s['steps'] for s in summaries),'own transitions')
