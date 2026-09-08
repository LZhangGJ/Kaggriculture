"""Paired actual labour -> production -> cash chains; no hypothetical rollout."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
from summarize_declared_value_trial import ci
EXP=Path(__file__).resolve().parents[1];root=EXP/'receipts/s4m1_pool_audit_N50_v1'
meta=json.loads((root/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
panel=json.loads((EXP/'receipts/s4m1_eightway_N50_v1/results.json').read_text())
freeze=json.loads((EXP/'profiles/s4m1/freeze.json').read_text())
refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
categories={'move':[1,2,3,4],'logistics':[5,6],'water':[9],'harvest':[10],'fertilize':[11],'feed':[15],'collect_fert':[16],'care':[17]}
counter_keys=['shared_insertion_checks','shared_insertion_applied','intraday_activated','intraday_purchase_orders','service_recovered_feed','service_recovered_fertilize']
hashes={};cache={};pairs=[]
def load(label,opp):
    if label in cache and opp in cache[label]:return cache[label][opp]
    if label in freeze['prior_mapping']:
        src=EXP/'receipts/s4l_pool_audit_N50_v1'/f"{freeze['prior_mapping'][label]}_{opp}.json.gz"
    else:src=root/f'{label}_{opp}.json.gz'
    hashes[str(src.relative_to(EXP))]=hashlib.sha256(src.read_bytes()).hexdigest()
    with gzip.open(src,'rt',encoding='utf8') as f:data=json.load(f)
    rows=[]
    for raw in data['rows']:
        ref=refs[label,opp,raw['seed'],raw['seat']];seat=raw['seat']
        assert raw['money'][seat]==ref['cash'] and raw['money'][1-seat]==ref['opponent_cash'] and raw['overflow']==ref['overflow']
        days=[]
        for d in range(30):
            prod=raw['production'][d][seat];money=raw['cash_ledger'][d][seat];plan=raw['planning'][d]
            daily={k:sum(prod['attempts'][i]-prod['no_effect'][i] for i in ids) for k,ids in categories.items()}
            daily.update(cash=money['end'],sales=sum(money['sales']),supplies=sum(money['products']),wages=money['hired'],
                seed_cost=sum(money['seeds']),animal_cost=sum(money['animals']),land_cost=money['land'],
                no_effect=sum(prod['no_effect']),storage_loss=sum(prod['drop_loss'])+sum(prod['eod_loss']),
                acquired=sum(prod['acquired']),sold=sum(prod['sold']),unassigned=plan['compile_drop'],unfed=plan['unfed'],unwatered=plan['unwatered'])
            days.append(daily)
        totals={k:(days[-1][k] if k=='cash' else sum(x[k] for x in days)) for k in days[0]}
        totals.update({k:raw['planning'][-1].get(k,0) for k in counter_keys})
        rows.append(dict(seed=raw['seed'],seat=seat,totals=totals,days=days))
    group=dict(rows=rows,mean={k:st.fmean(x['totals'][k] for x in rows) for k in rows[0]['totals']},
        days=[{k:st.fmean(x['days'][d][k] for x in rows) for k in rows[0]['days'][d]} for d in range(30)])
    cache.setdefault(label,{})[opp]=group;return group
for base in ('no_intraday','all_intraday','auto_portfolio','auto_portfolio_procure'):
    for opp in panel['identities']:
        before,after=load(base,opp),load(base+'_insert',opp)
        rows1={(r['seed'],r['seat']):r for r in before['rows']};rows2={(r['seed'],r['seat']):r for r in after['rows']}
        estimates={}
        for k in before['mean']:
            ds=[st.fmean(rows2[seed,s]['totals'][k]-rows1[seed,s]['totals'][k] for s in (0,1)) for seed in range(20262701,20262751)]
            estimates[k]=ci(ds)
        pairs.append(dict(context=base,opponent=opp,before=before['mean'],after=after['mean'],
            delta={k:after['mean'][k]-before['mean'][k] for k in before['mean']},estimates=estimates,
            days=[{k:after['days'][d][k]-before['days'][d][k] for k in before['days'][d]} for d in range(30)]))
out=EXP/'receipts/s4m1_cash_labour_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_LABOUR_CASH_CHAIN',pairs=pairs,input_hashes=hashes,
    caveat='Full-policy paired effects with responsive opponents; differences are not independent additive causal effects. Scheduling counter deltas across rolling steps are overlapping plans, not realized work savings. N50 development only.'),indent=2))
lines=['# S4M1 多人调度与产销联合变化','','每格100局；以下为开启插入减去关闭插入的均值。成功动作数不是奖励。','','|背景|对手|现金差|销售差|采购差|工资差|移动差|收获差|新增项目差|采用插入/局|无效动作/局|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in pairs:
    d=p['delta'];a=p['after'];lines.append('|'+p['context']+'|'+p['opponent']+'|'+'|'.join(f'{d[k]:.1f}' for k in ('cash','sales','supplies','wages','move','harvest','intraday_activated'))+f"|{a['shared_insertion_applied']:.1f}|{a['no_effect']:.1f}|")
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
