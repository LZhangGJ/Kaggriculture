"""Actual cash-chain differences after a whole live-policy change."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st
from run_cash_audit import totals

def summarize(rows):
    cash=[totals(r['cash_ledger'],r['seat']) for r in rows]
    snapshots={}
    for d in (0,5,11,19,29):
        xs=[]
        for r in rows:
            seat=r['seat'];daily=[x[seat] for x in r['cash_ledger'][:d+1]]
            xs.append(dict(cash=daily[-1]['end'],sales=sum(sum(x['sales']) for x in daily),
                seed_cost=sum(sum(x['seeds']) for x in daily),animal_cost=sum(sum(x['animals']) for x in daily),
                supplies=sum(sum(x['products']) for x in daily),wages=sum(x['hired'] for x in daily),land=sum(x['land'] for x in daily),
                sales_by_item=[sum(x['sales'][i] for x in daily) for i in range(9)],
                planted=[sum(x[seat]['planted'][i] for x in r['production'][:d+1]) for i in range(5)],
                placed=[sum(x[seat]['placed'][i] for x in r['production'][:d+1]) for i in range(3)]))
        snapshots[d]={k:[st.fmean(x[k][i] for x in xs) for i in range(len(v))] if isinstance(v,list) else st.fmean(x[k] for x in xs) for k,v in xs[0].items()}
    return dict(games=len(rows),snapshots=snapshots,terminal_cash=st.fmean(x['cash'] for x in cash))

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--baseline-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    paths=(Path(a.audit),Path(a.baseline_audit));meta=[json.loads((p/'summary.json').read_text()) for p in paths]
    assert all(x['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' for x in meta)
    hashes={str(p/'summary.json'):hashlib.sha256((p/'summary.json').read_bytes()).hexdigest() for p in paths}
    stats={};pairs=[]
    for e in meta[0]['summary']:
        label=e['label'];base=label.split('_auto')[0];opp=e['opponent'];loaded=[]
        for root,k in ((paths[1],base),(paths[0],label)):
            key=(k,opp)
            path=root/f'{k}_{opp}.json.gz'
            if key not in stats:
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                with gzip.open(path,'rt',encoding='utf8') as f:rows=json.load(f)['rows']
                stats[key]=summarize(rows)
            loaded.append(stats[key])
        b,t=loaded
        deltas={d:{k:[y-x for x,y in zip(b['snapshots'][d][k],v)] if isinstance(v,list) else v-b['snapshots'][d][k] for k,v in t['snapshots'][d].items()} for d in t['snapshots']}
        pairs.append(dict(baseline=base,trial=label,opponent=opp,actual_delta=deltas,
            no_effect_count=sum(e['own_production']['no_effect'])*e['games'],escape_count=sum(e['own_production']['escaped'])*e['games'],
            mean_degraded=e['planner_mean_degraded_tasks'],mean_unassigned=e['planner_mean_unassigned_tasks']))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='COMPLETE_ACTUAL_CASH_CHAIN_AUDIT',input_hashes=hashes,
        rows=[dict(label=k[0],opponent=k[1],**v) for k,v in stats.items()],pairs=pairs,
        caveat='Same seeds and seats, whole-policy changes. Day indices are zero-based. Accounting is exact; correlation of a product delta with victory is not an isolated causal effect. No future information enters any live policy.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# S4G真实现金链差异','','相对于同一上下文的固定开局；差额包含全部下游变化。','','|候选|对手|终局现金变化|销售变化|供料支出变化|种子支出变化|动物支出变化|工资变化|土地变化|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in pairs:
        if e['opponent'] not in ('g001','g003','pass'):continue
        d=e['actual_delta'][29];lines.append('|'+e['trial']+'|'+e['opponent']+'|'+'|'.join(f'{d[k]:,.0f}' for k in ('cash','sales','supplies','seed_cost','animal_cost','wages','land'))+'|')
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))

if __name__=='__main__':main()
