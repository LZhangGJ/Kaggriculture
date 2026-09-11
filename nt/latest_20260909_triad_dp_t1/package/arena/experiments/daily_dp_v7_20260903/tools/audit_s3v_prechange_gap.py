"""Cash identity for ordinary days from existing S3U audit, not a causal oracle."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
NAMES=['wheat','carrot','tomato','strawberry','melon','egg','milk','wool','fertilizer']
def main():
    src=EXP/'receipts/s3u_terminal_audit_A50_v1';out=EXP/'receipts/s3v_prechange_cash_gap_v1';out.mkdir(exist_ok=False)
    summary=json.loads((src/'summary.json').read_text());assert summary['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    results=[];hashes={}
    for entry in summary['summary']:
        if entry['label']!='terminal_only':continue
        path=src/f"terminal_only_{entry['opponent']}.json.gz";hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt') as f:rows=json.load(f)['rows']
        sides=[]
        for index in (0,1):
            individuals=[]
            for r in rows:
                seat=r['seat'] if index==0 else 1-r['seat'];days=[d[seat] for d in r['cash_ledger'][:29]];prod=[d[seat] for d in r['production'][:29]]
                d={k:[sum(x[k][i] for x in days) for i in range(len(days[0][k]))] for k in ('sales','sold','products','seeds','animals')}
                d.update({k:sum(x[k] for x in days) for k in ('hired','land')});d['cash']=days[-1]['end']
                d['eod_loss']=[sum(x['eod_loss'][i] for x in prod) for i in range(12)]
                assert 3000+sum(d['sales'])-sum(d['products'])-sum(d['seeds'])-sum(d['animals'])-d['hired']-d['land']==d['cash']
                individuals.append(d)
            sides.append({k:[st.fmean(x[k][i] for x in individuals) for i in range(len(v))] if isinstance(v,list) else st.fmean(x[k] for x in individuals) for k,v in individuals[0].items()})
        own,rival=sides;products=[]
        for i,item in enumerate(NAMES):
            q0,q1=own['sold'][i],rival['sold'][i];p0=own['sales'][i]/q0 if q0 else (rival['sales'][i]/q1 if q1 else 0);p1=rival['sales'][i]/q1 if q1 else p0
            quantity=(q0-q1)*(p0+p1)/2;quote=(p0-p1)*(q0+q1)/2
            assert abs(quantity+quote-(own['sales'][i]-rival['sales'][i]))<1e-7
            products.append(dict(item=item,own_quantity=q0,rival_quantity=q1,own_unit_receipt=p0,rival_unit_receipt=p1,
                revenue_gap=own['sales'][i]-rival['sales'][i],quantity_accounting_component=quantity,price_accounting_component=quote,
                own_eod_loss=own['eod_loss'][i],rival_eod_loss=rival['eod_loss'][i]))
        results.append(dict(opponent=entry['opponent'],games=len(rows),days='0..28',own=own,rival=rival,products=products))
    (out/'summary.json').write_text(json.dumps(dict(status='PASS_CASH_IDENTITY_NOT_CAUSAL',source_hashes=hashes,source_build=summary['build'],results=results,
        caveat='Opponent private ledgers are offline evidence only. Quantity/price decomposition is an accounting identity, not a recoverable profit guarantee. Repeated analysis adds no independent games.'),indent=2))
    lines=['# 普通生产日现金链（0..28日）','','| 对手 | 我方现金 | 对手现金 | 收入差 | 采购支出差 | 工资差 | 我方日末丢货 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        x,y=r['own'],r['rival'];lines.append(f"| {r['opponent']} | {x['cash']:,.0f} | {y['cash']:,.0f} | {sum(x['sales'])-sum(y['sales']):+,.0f} | {sum(x['products'])-sum(y['products']):+,.0f} | {x['hired']-y['hired']:+,.0f} | {sum(x['eod_loss']):.1f} |")
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
