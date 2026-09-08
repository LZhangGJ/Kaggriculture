"""Attribute actual money gaps, not hypothetical replacement revenue."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'receipts/s4l_pool_audit_N50_v1/summary.json'
data=json.loads(src.read_text());assert data['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
out=EXP/'receipts/s4l_competitive_cash_gap_v1';out.mkdir(exist_ok=False)
items=['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER']
rows=[]
for e in data['summary']:
    if e['opponent']=='pass':continue
    own,rival=e['own_cash'],e['rival_cash']
    parts={k:sum(own[k])-sum(rival[k]) for k in ('sales','products','seeds','animals')}
    parts.update({k:own[k]-rival[k] for k in ('hired','land')})
    gap=own['cash']-rival['cash']
    assert abs(parts['sales']-sum(v for k,v in parts.items() if k!='sales')-gap)<1e-6
    product=[]
    for i,name in enumerate(items):
        oq,rq=own['sold'][i],rival['sold'][i];oc,rc=own['sales'][i],rival['sales'][i]
        product.append(dict(item=name,own_sold=oq,rival_sold=rq,own_sales=oc,rival_sales=rc,
            own_realized_price=oc/oq if oq else None,rival_realized_price=rc/rq if rq else None,
            sales_gap=oc-rc,own_generated=e['own_production']['generated'][i],
            rival_generated=e['rival_production']['generated'][i],
            own_storage_loss=e['own_production']['drop_loss'][i]+e['own_production']['eod_loss'][i],
            rival_storage_loss=e['rival_production']['drop_loss'][i]+e['rival_production']['eod_loss'][i]))
    rows.append(dict(label=e['label'],opponent=e['opponent'],games=e['games'],wins=e['wins'],
        own_cash=own['cash'],rival_cash=rival['cash'],cash_gap=gap,ledger_difference=parts,products=product,
        own_unassigned=e['planner_mean_unassigned_tasks'],own_degraded=e['planner_mean_degraded_tasks']))
result=dict(status='COMPLETED_ACTUAL_COMPETITIVE_CASH_GAP',input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),rows=rows,
    caveat='Realized price is sales divided by actual sold quantity. Accounting decomposition is exact but is not a causal value of copying the rival or switching products. No hidden rival data enters the policy.')
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# S4L 实际竞争现金缺口','','同局双方真实账本；不是事后策略选择。成本差为正表示我方花得更多。','','|版本|对手|现金差|销售差|饲料/肥料采购差|种子差|动物投资差|工资差|土地差|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:
    d=x['ledger_difference'];lines.append('|'+x['label']+'|'+x['opponent']+'|'+f"{x['cash_gap']:.0f}|"+'|'.join(f'{d[k]:.0f}' for k in ('sales','products','seeds','animals','hired','land'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(status=result['status'],groups=len(rows))))
