"""Preserve the observed lower-cash case without fitting a policy exception."""
from pathlib import Path
import gzip,hashlib,json
EXP=Path(__file__).resolve().parents[1]
def load(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def main():
    out=EXP/'receipts/s3v_loss_fixture_analysis_v1';out.mkdir(exist_ok=False)
    paths=[EXP/f'receipts/{name}/20261416_seat0.json.gz' for name in ('old_s3v_dispatch_loss_official_v1','idle_s3v_dispatch_loss_official_v1')]
    a,b=map(load,paths);first=next(i for i,(x,y) in enumerate(zip(a['trace'],b['trace'])) if x['actions']!=y['actions'])
    assert all(a['trace'][i]==b['trace'][i] for i in range(first))
    assert a['trace'][first]['observation']==b['trace'][first]['observation']
    changed=[(i,x,y) for i,(x,y) in enumerate(zip([a['trace'][first]['actions'][0]['farmer'],*a['trace'][first]['actions'][0]['hands']],
        [b['trace'][first]['actions'][0]['farmer'],*b['trace'][first]['actions'][0]['hands']])) if x!=y]
    rawpaths=[EXP/f'receipts/s3v_idle_audit_A50_v1/{label}_g003.json.gz' for label in ('old','idle_only')]
    rows=[next(r for r in load(p)['rows'] if r['seed']==20261416 and r['seat']==0) for p in rawpaths]
    difference=[]
    for seat in (0,1):
        result={}
        for source,keys in [('production',('generated','acquired','eod_loss','planted','placed','fed','cared')),('cash_ledger',('sales','products','seeds','animals'))]:
            for key in keys:
                result[key]=[sum(d[seat][key][i] for d in rows[1][source])-sum(d[seat][key][i] for d in rows[0][source]) for i in range(len(rows[0][source][0][seat][key]))]
        difference.append(result)
    cash=[r['final']['farms'][0]['money'] for r in (a,b)];rival=[r['final']['farms'][1]['money'] for r in (a,b)]
    receipt=dict(status='VERIFIED_OFFICIAL_LOWER_CASH_CASE',seed=20261416,seat=0,first_action_change=first,changed_units=changed,
        own_cash=cash,rival_cash=rival,own_cash_change=cash[1]-cash[0],margin_change=(cash[1]-rival[1])-(cash[0]-rival[0]),
        quantity_and_cash_differences=difference,input_hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+rawpaths},
        caveat='Offline explanatory example, not an opponent or seed exception. Less physical overflow need not mean more cash; margin and cash can move in opposite directions.')
    (out/'summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('quantity_and_cash_differences','input_hashes')},ensure_ascii=False))
if __name__=='__main__':main()
