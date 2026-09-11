"""Trace-based loss localization; does not change action or assert causality."""
from pathlib import Path
import gzip,json
E=Path(__file__).resolve().parents[1];src=E/'receipts/s5c_obligations_v1';out=E/'receipts/s5c_obligation_summary_v1';out.mkdir(exist_ok=False)
accept=json.loads((src/'acceptance.json').read_text());rows=[];detail=[]
def animal(t):return isinstance(t,dict) and 'animal' in t
def tile(obs,seat,pos):return obs['farms'][seat]['tiles'][pos//10][pos%10]
def jobs_have(js,pos,op=15):return any(j['pos']==pos and any(a[0]==op for a in j['actions']) for j in js)
def plans_at(plans,pos,op=15):return [(u,k) for u,p in enumerate(plans) for k,a in enumerate(p) if a[0]==op and a[3]==pos]
for record in accept['rows']:
 data=json.load(gzip.open(src/record['trace'],'rt'));case=data['case'];seat=case['seat'];trace=data['trace'];lost=[]
 for f in trace:
  for pos in range(100):
   if animal(tile(f['before'],seat,pos)) and not animal(tile(f['after'],seat,pos)):lost.append((f['step'],pos))
 assert len(lost)==case['escaped']
 for step,pos in lost:
  for day in range(step//24-2,step//24+1):
   frames=[f for f in trace if f['step']//24==day];start=frames[0];compile_frames=[f for f in frames if 'compile_plans' in f['pre']]
   c=compile_frames[0] if compile_frames else None;events=[]
   for f in frames:
    farm=f['before']['farms'][seat];units=[farm['farmer']]+farm['hands'];acts=[f['own']['farmer']]+f['own']['hands']
    for u,(xy,action) in enumerate(zip(units,acts)):
     if xy[1]*10+xy[0]==pos and action[0] not in ('PASS','NORTH','SOUTH','EAST','WEST'):events.append((f['step'],u,action))
   t=tile(start['before'],seat,pos)
   row=dict(label=case['label'],opponent=case['opponent'],seed=case['seed'],seat=seat,pos=pos,day=day,animal=t.get('animal') if animal(t) else None,
     consecutive_unfed=t.get('consecutive_unfed') if animal(t) else None,cash=start['before']['farms'][seat]['money'],
     compile_step=c['step'] if c else None,raw_feed=jobs_have(c['pre'].get('jobs',[]),pos) if c else None,
     reserved_feed=jobs_have(c['pre'].get('reserved',[]),pos) if c else None,
     compile_feed=plans_at(c['pre']['compile_plans'],pos) if c else [],
     compile_drop=c['pre']['compile_drop'] if c else None,compile_degraded=c['pre']['compile_degraded'] if c else None,
     actions=events)
   rows.append(row)
  # Preserve whole missing animal plan timeline, not just end-of-day aggregate.
  timeline=[]
  for f in trace:
   b=tile(f['before'],seat,pos)
   if not animal(b):continue
   timeline.append(dict(step=f['step'],tile=b,cash=f['before']['farms'][seat]['money'],private=f['before']['private'],
    before_feed=plans_at(f['pre']['plans'],pos),after_feed=plans_at(f['post']['plans'],pos),raw_feed=jobs_have(f['post'].get('jobs',[]),pos),
    own=f['own'],positions=[f['before']['farms'][seat]['farmer']]+f['before']['farms'][seat]['hands']))
  detail.append(dict(case=case,pos=pos,escape_step=step,timeline=timeline))
(out/'rows.json').write_text(json.dumps(rows,indent=2))
with gzip.open(out/'timelines.json.gz','wt') as f:json.dump(detail,f)
(out/'acceptance.json').write_text(json.dumps(dict(status='PASS_TRACE_LOCALIZATION',games=len(accept['rows']),lost_animals=len(detail),rows=len(rows),source=str(src/'acceptance.json')),indent=2))
for r in rows:
 if r['seat']==0:print(json.dumps(r),flush=True)
