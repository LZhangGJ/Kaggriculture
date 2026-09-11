"""Aggregate all opponents, including losses. Accounting is not causality."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.audit);src=root/'summary.json';audit=json.loads(src.read_text());assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);result=[]
    for entry in audit['summary']:
        if entry['label']!='log1_own0_new0':continue
        path=root/f"{entry['label']}_{entry['opponent']}.json.gz"
        with gzip.open(path,'rt',encoding='utf-8') as f:rows=json.load(f)['rows']
        stage=[]
        for stop in (0,2,5,8,11,17,23,29):
            def get(r,side):
                seat=r['seat'] if side==0 else 1-r['seat'];prod=[day[seat] for day in r['production'][:stop+1]]
                cash=[day[seat] for day in r['cash_ledger'][:stop+1]]
                return dict(cash=cash[-1]['end'],wheat_planted=sum(d['planted'][0] for d in prod),strawberry_planted=sum(d['planted'][3] for d in prod),
                    wheat_harvested=sum(d['acquired'][0] for d in prod),strawberry_harvested=sum(d['acquired'][3] for d in prod),
                    wheat_purchased=sum(d['bought_products'][0] for d in cash),wages=sum(d['hired'] for d in cash),land_purchases=sum(d['lands'] for d in cash))
            sides=[]
            for side in (0,1):
                values=[get(r,side) for r in rows];sides.append({k:statistics.fmean(v[k] for v in values) for k in values[0]})
            stage.append(dict(day_zero_based=stop,own=sides[0],opponent=sides[1]))
        result.append(dict(opponent=entry['opponent'],stages=stage))
    (out/'stages.json').write_text(json.dumps(dict(status='COMPLETE_OFFLINE_ATTRIBUTION',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),results=result),indent=2))
    for r in result:
        if r['opponent'] in ('pass','g001','g003','lynn_v5','ecobot_v7'):
            print(json.dumps(r))

if __name__=='__main__':main()
