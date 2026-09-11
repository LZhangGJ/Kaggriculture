"""Pure post-hoc paired cash/production attribution. Does not run a policy."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics
EXP=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--audit',default=str(EXP/'receipts/s3m_seed_audit_A50_v1'))
parser.add_argument('--out',default=str(EXP/'receipts/s3m_economic_review_v1'))
parser.add_argument('--base',default='seed_base');parser.add_argument('--variant',default='seed_reconcile');args=parser.parse_args()
audit=Path(args.audit);out=Path(args.out);out.mkdir(exist_ok=False);labels=(args.base,args.variant)
data=json.loads((audit/'summary.json').read_text());assert data['all_ledgers_reconciled']
summary=[];witnesses=[]
ops=['pass','g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7']
for op in ops:
    pair={}
    for label in labels:
        path=audit/f'{label}_{op}.json.gz'
        with gzip.open(path,'rt') as f:content=json.load(f)
        pair[label]=content
    old,new=(pair[x]['summary'] for x in labels)
    def entry(s):
        p,c=s['own_production'],s['own_cash']
        return dict(cash=c['cash'],wages=c['hired'],seed_costs=c['seeds'],product_costs=c['products'],sales=c['sales'],
            planted=p['planted'],acquired=p['acquired'],overflow=[a+b for a,b in zip(p['drop_loss'],p['eod_loss'])],
            environment_loss=p['environment_loss'],end_private=p['end_private'],end_field=p['end_field'],
            unassigned=s['planner_mean_unassigned_tasks'],degraded=s['planner_mean_degraded_tasks'])
    rows0={(r['seed'],r['seat']):r for r in pair[args.base]['rows']}
    rows1={(r['seed'],r['seat']):r for r in pair[args.variant]['rows']}
    assert rows0.keys()==rows1.keys()
    deltas=[rows1[k]['money'][k[1]]-rows0[k]['money'][k[1]] for k in rows0]
    summary.append(dict(opponent=op,games=len(rows0),base=entry(old),repair=entry(new),
        cash_increased=sum(v>0 for v in deltas),cash_unchanged=sum(v==0 for v in deltas),cash_decreased=sum(v<0 for v in deltas)))
    if op=='g001':
        for key in ((20261429,0),(20261433,0),(20261434,0)):
            rec=dict(seed=key[0],seat=key[1],variants={})
            for label,rows in [(args.base,rows0),(args.variant,rows1)]:
                r=rows[key];days=[]
                for day in range(12,30):
                    c,p,plan=r['cash_ledger'][day][key[1]],r['production'][day][key[1]],r['planning'][day]
                    days.append(dict(day=day,cash=c['end'],hires=c['hires'],wages=c['hired'],planted=p['planted'],acquired=p['acquired'],
                        sold=c['sold'],sales=c['sales'],product_costs=c['products'],seed_costs=c['seeds'],
                        target=plan['target'],live=plan['live'],unassigned=plan['compile_drop'],degraded=plan['degraded'],
                        overflow=[a+b for a,b in zip(p['drop_loss'],p['eod_loss'])]))
                rec['variants'][label]=days
            witnesses.append(rec)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_OFFLINE_ATTRIBUTION',args=vars(args),summary=summary,witnesses=witnesses,
    source_sha256=hashlib.sha256((audit/'summary.json').read_bytes()).hexdigest(),
    caveat='Paired configuration effect is causal within frozen simulator games; individual cash-flow differences are not separately recoverable gains.'),indent=2))
for s in summary:
    b,n=s['base'],s['repair']
    print(json.dumps(dict(opponent=s['opponent'],delta_cash=n['cash']-b['cash'],delta_wages=n['wages']-b['wages'],
        delta_planted=[x-y for x,y in zip(n['planted'],b['planted'])],delta_acquired=[x-y for x,y in zip(n['acquired'],b['acquired'])],
        delta_sales=[x-y for x,y in zip(n['sales'],b['sales'])],increased=s['cash_increased'],decreased=s['cash_decreased'])))
for w in witnesses:
    print('WITNESS '+str(w['seed']))
    for b,n in zip(w['variants'][args.base],w['variants'][args.variant]):
        if b['day']>=17 and (b['planted']!=n['planted'] or b['target']!=n['target']):
            print(json.dumps(dict(day=b['day'],old_target=b['target'],new_target=n['target'],old_planted=b['planted'],new_planted=n['planted'],cash_delta=n['cash']-b['cash'],old_hires=b['hires'],new_hires=n['hires'])))
