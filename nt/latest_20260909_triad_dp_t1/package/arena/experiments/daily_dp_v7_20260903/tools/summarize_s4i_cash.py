"""Account for observed full-game changes, not hypothetical suffix outcomes."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st
from summarize_startup_cash import summarize
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='s4i2');a=p.parse_args();r=a.round
root=EXP/f'receipts/{r}_pool_audit_N50_v1';baseline=EXP/'receipts/s4e1_pool_audit_N50_v1'
meta=json.loads((root/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
out=EXP/f'receipts/{r}_cash_chain_N50_v1';out.mkdir(exist_ok=False);rows=[];hashes={};cache={}
for entry in meta['summary']:
    label=entry['label'];context='all_intraday' if label.startswith('all_intraday') else 'intraday_funded';opp=entry['opponent']
    pair=[]
    for directory,k in ((baseline,context),(root,label)):
        path=directory/f'{k}_{opp}.json.gz'
        if str(path) not in cache:
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path,'rt',encoding='utf8') as f:records=json.load(f)['rows']
            counters={key:st.fmean(x['planning'][-1].get(key,0) for x in records) for key in ('portfolio_generated','portfolio_evaluated','portfolio_switches','portfolio_rejected')}
            cache[str(path)]=(summarize(records),counters)
            del records
        pair.append(cache[str(path)])
    before,after=pair[0][0],pair[1][0]
    delta={d:{k:[y-x for x,y in zip(before['snapshots'][d][k],value)] if isinstance(value,list) else value-before['snapshots'][d][k]
              for k,value in snapshot.items()} for d,snapshot in after['snapshots'].items()}
    counters=pair[1][1]
    rows.append(dict(label=label,baseline=context,opponent=opp,before=before,after=after,delta=delta,counters=counters,
                     no_effect=entry['own_production']['no_effect'],escaped=entry['own_production']['escaped']))
report=dict(status='COMPLETE_ACTUAL_CASH_CHAIN_COMPARISON',rows=rows,input_hashes=hashes,holdout_used=False,
            caveat='Old baseline audited on S4E; S4F, S4G and this panel must reproduce its live controls. Whole-policy causal contrast does not isolate any one changed product or action.')
(out/'summary.json').write_text(json.dumps(report,indent=2))
lines=['# S4I真实现金链差额','','|配置|对手|现金|销售|供料成本|种子成本|动物成本|工资|土地|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    if x['opponent'] not in ('pass','g001','g003'):continue
    d=x['delta'][29];lines.append('|'+x['label']+'|'+x['opponent']+'|'+'|'.join(f'{d[k]:,.0f}' for k in ('cash','sales','supplies','seed_cost','animal_cost','wages','land'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
