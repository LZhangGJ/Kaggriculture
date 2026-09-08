"""Locate observed service/worker events in the diagnostic live replays."""
from pathlib import Path
import gzip,hashlib,json
EXP=Path(__file__).resolve().parents[1];root=EXP/'receipts/s4l_service_trace_v1'
meta=json.loads((root/'acceptance.json').read_text());assert meta['status']=='PASS_MATCHED_LIVE_TRACE_NOT_STRENGTH'
out=EXP/'receipts/s4l_service_trace_summary_v1';out.mkdir(exist_ok=False);summaries=[];events=[]
for entry in meta['rows']:
    path=root/entry['trace'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    with gzip.open(path,'rt',encoding='utf8') as f:r=json.load(f)
    seat=r['seat'];days=[]
    for d in range(3):
        ticks=[t for t in r['trace'] if t['before']['day']==d]
        services={key:sum(t['service_after'][key]-t['service_before'][key] for t in ticks) for key in ticks[0]['service_before']}
        actions=[a for t in ticks for a in [t['actions'][seat]['farmer'],*t['actions'][seat]['hands']]]
        day=dict(day=d,services=services,issued_feed=sum(a[0]=='FEED' for a in actions),issued_pickup=sum(a[0]=='PICKUP' for a in actions),
            cash=ticks[-1]['after']['farms'][seat]['money'])
        days.append(day)
        for t in ticks:
            a=t['actions'][seat];b=t['before'];change={k:t['service_after'][k]-t['service_before'][k] for k in services if k!='checks'}
            if not any(change.values()):continue
            own=b['farms'][seat]
            events.append(dict(label=r['label'],seed=r['seed'],seat=seat,day=d,hour=b['hour'],change=change,
                cash_before=own['money'],shed=b['private']['shed'],inventories=b['private']['inventories'],
                positions=[own['farmer'],*own['hands']],actions=a,phase=t['debug']['phase']))
    summaries.append(dict(label=r['label'],seed=r['seed'],seat=seat,days=days))
result=dict(status='COMPLETED_SERVICE_TRACE_ACCOUNTING',rows=summaries,events=events,
    trace_acceptance_sha256=hashlib.sha256((root/'acceptance.json').read_bytes()).hexdigest(),
    caveat='Issued event counts do not equal completion; reconcile with per-unit effects and daily production audit.')
(out/'summary.json').write_text(json.dumps(result,indent=2))
for e in summaries:
    if e['seed']==20262701 and e['seat']==0:print(json.dumps(e))
for e in events:
    if e['seed']==20262701 and e['seat']==0 and e['day']<=1:print(json.dumps(e))
