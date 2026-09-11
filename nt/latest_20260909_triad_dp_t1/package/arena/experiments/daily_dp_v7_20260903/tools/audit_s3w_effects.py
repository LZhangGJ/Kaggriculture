"""Paired economic consequences; cash identities, not independent causes."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def measures(r):
    seat=r['seat'];cash=r['cash_ledger'];prod=r['production'];own=[d[seat] for d in cash];rival=[d[1-seat] for d in cash]
    x=dict(cash=r['money'][seat],rival_cash=r['money'][1-seat],
        d11_cash=own[11]['end'],d11_income=sum(sum(d['sales']) for d in own[:12]),
        d11_invest=sum(sum(d['seeds'])+sum(d['animals'])+d['land'] for d in own[:12]),
        total_income=sum(sum(d['sales']) for d in own),total_products=sum(sum(d['products']) for d in own),
        wages=sum(d['hired'] for d in own),
        shortage_days=sum(d['resource_dropped_jobs']>0 for d in r['admissions']),
        no_effect=sum(sum(d[seat]['no_effect']) for d in prod),escapes=sum(sum(d[seat]['escaped']) for d in prod))
    x['margin']=x['cash']-x['rival_cash']
    for i,name in enumerate(['W','C','T','S','M']):x['plant_'+name]=sum(d[seat]['planted'][i] for d in prod)
    for i,name in enumerate(['goose','cow','sheep']):x['place_'+name]=sum(d[seat]['placed'][i] for d in prod)
    return x
def main():
    src=EXP/'receipts/s3w_factorial_audit_A50_v1';out=EXP/'receipts/s3w_factorial_effects_v1';out.mkdir(exist_ok=False)
    meta=json.loads((src/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    result=[];hashes={}
    for item in meta['summary']:
        label,op=item['label'],item['opponent'];p=src/f'{label}_{op}.json.gz';hashes[p.name]=sha(p)
        rows=load(p)['rows'];vals=[measures(r) for r in rows]
        avg={k:st.fmean(x[k] for x in vals) for k in vals[0]}
        result.append(dict(label=label,opponent=op,games=len(rows),means=avg))
    summary=dict(status='COMPLETE_PAIRED_ECONOMIC_DIAGNOSTICS',build=meta['build'],source_hashes=hashes,results=result,
        caveat='Treatment changes production and market jointly. Component cash deltas are identities, not separately recoverable gains. No-effect checks cover emitted actions, not every possible omitted intention.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    for op in ['pass','g001','g003','boatlee_v29']:
        for x in result:
            if x['opponent']==op:print(json.dumps(dict(label=x['label'],opponent=op,**{k:round(v,2) for k,v in x['means'].items()})))
if __name__=='__main__':main()
