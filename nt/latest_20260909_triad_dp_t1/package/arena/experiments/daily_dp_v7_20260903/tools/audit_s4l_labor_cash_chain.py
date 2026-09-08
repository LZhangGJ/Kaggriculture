"""Joint daily labour/production/cash accounting from existing real games.

Issued PASS is not claimed to equal spare usable capacity. Rival omitted
commands may not be recorded; neither movement nor service count is a reward.
"""
from pathlib import Path
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
root=EXP/'receipts/s4l_pool_audit_N50_v1';meta=json.loads((root/'summary.json').read_text())
assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
out=EXP/'receipts/s4l_labor_cash_chain_v1';out.mkdir(exist_ok=False);groups=[];hashes={}
categories={'pass':[0],'move':[1,2,3,4],'logistics':[5,6],'place':[7],'plant_build_dig':[8,12,13,14],
            'water':[9],'harvest':[10],'fertilize':[11],'feed':[15],'collect_fert':[16],'care':[17]}
for entry in meta['summary']:
    src=root/f"{entry['label']}_{entry['opponent']}.json.gz";hashes[src.name]=hashlib.sha256(src.read_bytes()).hexdigest()
    with gzip.open(src,'rt',encoding='utf8') as f:data=json.load(f)
    daily=[]
    for d in range(30):
        sides={}
        for side in ('own','rival'):
            observations=[]
            for row in data['rows']:
                seat=row['seat'] if side=='own' else 1-row['seat'];prod=row['production'][d][seat]
                cash=row['cash_ledger'][d][seat];attempt=prod['attempts'];fail=prod['no_effect']
                values={k:sum(attempt[i]-fail[i] for i in ids) for k,ids in categories.items()}
                values.update(cash=cash['end'],sales=sum(cash['sales']),supplies=sum(cash['products']),
                    wages=cash['hired'],hired=cash['hires'],seed_cost=sum(cash['seeds']),animal_cost=sum(cash['animals']),
                    land_cost=cash['land'],storage_loss=sum(prod['drop_loss'])+sum(prod['eod_loss']),
                    acquired=sum(prod['acquired']),sold=sum(prod['sold']),no_effect=sum(fail))
                if side=='own':values.update(unfed=row['planning'][d]['unfed'],unwatered=row['planning'][d]['unwatered'],
                    unassigned=row['planning'][d]['compile_drop'],degraded=row['planning'][d]['degraded'])
                observations.append(values)
            sides[side]={k:st.fmean(x[k] for x in observations) for k in observations[0]}
        daily.append(dict(day=d,**sides))
    groups.append(dict(label=entry['label'],opponent=entry['opponent'],games=entry['games'],days=daily,
        season={side:{k:(daily[-1][side][k] if k=='cash' else sum(x[side][k] for x in daily)) for k in daily[0][side]} for side in ('own','rival')}))
result=dict(status='COMPLETED_REAL_LABOUR_AND_CASH_CHAIN',groups=groups,input_hashes=hashes,
    caveat='Actual completed unit effects and accounting only. Counts differ across production structures; idle is not necessarily waste, unwatered/unfed is not automatically a bug. No rival private values enter runtime decisions.')
(out/'summary.json').write_text(json.dumps(result,indent=2))
lines=['# S4L 工人、生产与现金联合复盘','','每格50seed双座位。成功动作次数不是优化目标；应结合生产结构和收入。','','|版本|对手|我方/对方移动|我方/对方维护动作|我方/对方收获|我方/对方工资|我方/对方销售额|我方/对方现金|','|---|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    a,b=g['season']['own'],g['season']['rival']
    service=lambda s:sum(s[k] for k in ('water','fertilize','feed','care','collect_fert'))
    pairs=[(a['move'],b['move']),(service(a),service(b)),(a['harvest'],b['harvest']),(a['wages'],b['wages']),(a['sales'],b['sales']),(a['cash'],b['cash'])]
    lines.append('|'+g['label']+'|'+g['opponent']+'|'+'|'.join(f'{x:.0f}/{y:.0f}' for x,y in pairs)+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(status=result['status'],groups=len(groups))))
