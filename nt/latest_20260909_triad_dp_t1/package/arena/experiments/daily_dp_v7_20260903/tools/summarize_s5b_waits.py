"""Read-only timing-intent transitions; not a claim that all postponement is bad."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[1];path=E/'receipts/s5b_runtime_v2/acceptance.json';data=json.loads(path.read_text());rows=[]
for game in data['rows']:
 if not game['label'].endswith('_timing'):continue
 events=[];active_days=0
 for before,after in zip(game['states'],game['states'][1:]):
  day=after['step']//24;a=before['stats'];b=after['stats']
  active_days+=any(x>day for x in b['plant_not_before'])
  for pos,(ka,kb) in enumerate(zip(a['deferred_kind'],b['deferred_kind'])):
   ta=a['plant_not_before'][pos];tb=b['plant_not_before'][pos]
   if ka>=0 and ka==kb and tb>ta and tb>day:
    events.append(dict(day=day,pos=pos,kind=kb,old_start=ta,new_start=tb))
 rows.append(dict(label=game['label'],seat=game['seat'],seed=game['seed'],repeated_postponements=len(events),active_wait_days=active_days,events=events))
out=E/'receipts/s5b_wait_intent_audit_v1';out.mkdir(exist_ok=False)
result=dict(status='COMPLETE_READ_ONLY_INTENT_AUDIT',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,caveat='Same known development seed both seats; pending intent transitions only. Need actual planting/resource results to distinguish useful delay from churn.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in rows],indent=2))
