"""Whole-policy paired labour, output and cash effects, not hypothetical labels."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
from summarize_declared_value_trial import ci
E=Path(__file__).resolve().parents[1]
panel=json.loads((E/'receipts/s4q_sixway_N50_v1/results.json').read_text())
audit=json.loads((E/'receipts/s4q_pool_audit_N50_v1/summary.json').read_text())
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
refs={(r['variant'],r['opponent'],r['seed'],r['seat']):r for r in panel['rows']}
old_mapping=json.loads((E/'profiles/s4m1/freeze.json').read_text())['prior_mapping']
hashes={};cache={};pairs=[]
categories={'move':[1,2,3,4],'logistics':[5,6],'water':[9],'harvest':[10],'fertilize':[11],'feed':[15],'collect_fert':[16],'care':[17]}
def load(label,opp):
    key=(label,opp)
    if key in cache:return cache[key]
    if label=='all_intraday':src=E/f'receipts/s4l_pool_audit_N50_v1/{old_mapping[label]}_{opp}.json.gz'
    elif label=='all_intraday_insert':src=E/f'receipts/s4m1_pool_audit_N50_v1/{label}_{opp}.json.gz'
    else:src=E/f'receipts/s4q_pool_audit_N50_v1/{label}_{opp}.json.gz'
    hashes[str(src.relative_to(E))]=hashlib.sha256(src.read_bytes()).hexdigest()
    with gzip.open(src,'rt',encoding='utf8') as f:raw=json.load(f)
    rows={}
    for r in raw['rows']:
        seat=r['seat'];ref=refs[label,opp,r['seed'],seat]
        assert r['money'][seat]==ref['cash'] and r['money'][1-seat]==ref['opponent_cash'] and r['overflow']==ref['overflow']
        days=[]
        for d in range(30):
            p=r['production'][d][seat];c=r['cash_ledger'][d][seat]
            v={k:sum(p['attempts'][i]-p['no_effect'][i] for i in ids) for k,ids in categories.items()}
            v.update(cash=c['end'],sales=sum(c['sales']),supplies=sum(c['products']),wages=c['hired'],seed_cost=sum(c['seeds']),animal_cost=sum(c['animals']),land_cost=c['land'],no_effect=sum(p['no_effect']),storage_loss=sum(p['drop_loss'])+sum(p['eod_loss']),acquired=sum(p['acquired']),sold=sum(p['sold']))
            for i in range(9):v['sales_'+str(i)]=c['sales'][i];v['sold_'+str(i)]=p['sold'][i]
            days.append(v)
        totals={k:days[-1][k] if k=='cash' else sum(x[k] for x in days) for k in days[0]}
        for k in ('shared_insertion_applied','intraday_activated','intraday_purchase_orders'):
            totals[k]=r['planning'][-1].get(k,0)
        rows[r['seed'],seat]=dict(totals=totals,days=days)
    cache[key]=rows;return rows
for base in ('all_intraday','all_intraday_insert'):
    for opp in panel['identities']:
        for before,after in ((base,base+'_value'),(base+'_value',base+'_value_public'),(base,base+'_value_public')):
            aa,bb=load(before,opp),load(after,opp);assert set(aa)==set(bb)
            estimates={}
            for k in next(iter(aa.values()))['totals']:
                estimates[k]=ci([st.fmean(bb[seed,s]['totals'][k]-aa[seed,s]['totals'][k] for s in (0,1)) for seed in range(20262701,20262751)])
            pairs.append(dict(before=before,after=after,opponent=opp,estimates=estimates,changed_cash_games=sum(aa[k]['totals']['cash']!=bb[k]['totals']['cash'] for k in aa),days=[{k:st.fmean(bb[x]['days'][d][k]-aa[x]['days'][d][k] for x in aa) for k in aa[next(iter(aa))]['days'][d]} for d in range(30)]))
out=E/'receipts/s4q_cash_labour_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_LABOUR_CASH_CHAIN',pairs=pairs,input_hashes=hashes,caveat='Live full-policy paired changes. Each seed clusters both seats. These channels are not separate causal experiments. Historical control ledgers are reused only after both cash and overflow match the current repeated control.'),indent=2))
lines=['# S4Q 多人调度—生产—现金的完整变化','','|之前|之后|对手|现金差|销售差|采购差|工资差|移动差|收获差|新增项目差|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for p in pairs:
    lines.append('|'+p['before']+'|'+p['after']+'|'+p['opponent']+'|'+'|'.join(f"{p['estimates'][k]['mean']:.1f}" for k in ('cash','sales','supplies','wages','move','harvest','intraday_activated'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(status='COMPLETE',pairs=len(pairs))),flush=True)
