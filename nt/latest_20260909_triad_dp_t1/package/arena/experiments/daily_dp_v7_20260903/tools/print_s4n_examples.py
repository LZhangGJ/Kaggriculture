from pathlib import Path
from collections import Counter
import gzip,json
root=Path(__file__).resolve().parents[1]/'receipts/s4n_timing_analysis_v1'
with gzip.open(root/'examples.json.gz','rt',encoding='utf8') as f:rows=json.load(f)
counts=Counter()
for r in rows:
    p=r['first']['market_action']
    if p is None:continue
    w=next(x for x in r['first_windows'] if x['pivot']==p)
    a=next(x for x in w['old'] if x['step']==p);b=next(x for x in w['new'] if x['step']==p)
    counts[(tuple(tuple(x) for x in a['actions']['market']),tuple(tuple(x) for x in b['actions']['market']))]+=1
print('FIRST_MARKET_DIFFERENCES',json.dumps([dict(old=k[0],new=k[1],count=n) for k,n in counts.most_common(10)]))
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    r=next(r for r in rows if r['opponent']==opponent and r['seed']==20262701 and r['seat']==0)
    print(json.dumps(dict(opponent=opponent,first=r['first'],cash_delta=r['cash_delta'],margin_delta=r['margin_delta']),indent=2))
    for p in sorted({r['first']['unit_action'],r['first']['market_action'],r['first']['target']}):
        if p is None:continue
        w=next(x for x in r['first_windows'] if x['pivot']==p)
        a=next(x for x in w['old'] if x['step']==p);b=next(x for x in w['new'] if x['step']==p)
        compact=lambda x:dict(cash=x['cash'],positions=x['positions'],inventories=x['inventories'],
            shed={k:v for k,v in x['shed'].items() if v},seeds={k:v for k,v in x['seed'].items() if v},actions=x['actions'])
        print(json.dumps(dict(step=p,old=compact(a),new=compact(b),
            target_old_only=sorted(set(map(tuple,a['target']))-set(map(tuple,b['target']))),
            target_new_only=sorted(set(map(tuple,b['target']))-set(map(tuple,a['target']))))))
