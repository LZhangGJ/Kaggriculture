"""Actual early investment, production and sale timing. Never edits the agent."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,statistics as st


def mean_vector(xs):return [st.fmean(x[i] for x in xs) for i in range(len(xs[0]))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--label',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.audit);meta=json.loads((src/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    result=[];hashes={}
    for opponent in ('g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7'):
        path=src/f'{a.label}_{opponent}.json.gz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt',encoding='utf8') as f:rows=json.load(f)['rows']
        sides=[]
        for side in (0,1):
            opening_crops=[];opening_animals=[];days={d:[] for d in (5,11,19,29)};first_sales=collections.defaultdict(list)
            for r in rows:
                seat=r['seat'] if side==0 else 1-r['seat']
                opening_crops.append(r['production'][0][seat]['planted']);opening_animals.append(r['production'][0][seat]['placed'])
                for i in range(9):
                    ds=[d for d,record in enumerate(r['cash_ledger']) if record[seat]['sold'][i]>0]
                    if ds:first_sales[i].append(ds[0])
                for day in days:
                    cash=[d[seat] for d in r['cash_ledger'][:day+1]]
                    days[day].append(dict(cash=cash[-1]['end'],sales=sum(sum(d['sales']) for d in cash),
                        investment=sum(sum(d['seeds'])+sum(d['animals'])+d['land'] for d in cash),
                        wages=sum(d['hired'] for d in cash),products=sum(sum(d['products']) for d in cash),
                        sales_by_item=[sum(d['sales'][i] for d in cash) for i in range(9)],
                        cumulative_planted=[sum(d[seat]['planted'][i] for d in r['production'][:day+1]) for i in range(5)],
                        cumulative_placed=[sum(d[seat]['placed'][i] for d in r['production'][:day+1]) for i in range(3)]))
            sides.append(dict(side='own' if side==0 else 'opponent',opening_crops=mean_vector(opening_crops),opening_animals=mean_vector(opening_animals),
                first_sale_day={k:dict(games=len(ds),median=st.median(ds),histogram=dict(collections.Counter(ds))) for k,ds in first_sales.items()},
                checkpoints={d:{k:mean_vector([r[k] for r in rs]) if isinstance(rs[0][k],list) else st.fmean(r[k] for r in rs) for k in rs[0]} for d,rs in days.items()}))
        harvests=collections.defaultdict(list);n=collections.Counter()
        for r in rows:
            for c in r['investments']['cohorts']:
                if c['start_step']//24!=0:continue
                n[c['kind']]+=1
                if c['first_harvest_step']>=0:harvests[c['kind']].append(c['first_harvest_step']//24)
        result.append(dict(opponent=opponent,games=len(rows),sides=sides,own_day0_cohort_first_harvest={k:dict(cohorts=n[k],harvested_cohorts=len(ds),median_day=st.median(ds),histogram=dict(collections.Counter(ds))) for k,ds in harvests.items()}))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='COMPLETE_ACTUAL_EARLY_CASH_AUDIT',label=a.label,input_hashes=hashes,results=result,
        caveat='Day numbers are zero-based. Production batches and market sales are separately observed; pooled inventory sales are not uniquely attributable to one cohort. This describes differences, not the isolated causal value of copying an opening.')
    (out/'summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],opponents=len(result),games=sum(r['games'] for r in result))))


if __name__=='__main__':main()
