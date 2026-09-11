from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    f=E/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
old=read('receipts/s4m1_pool_audit_N50_v1/summary.json');legacy=read('receipts/s4l_pool_audit_N50_v1/summary.json');new=read('receipts/s4s_pool_audit_N50_v1/summary.json');p=read('receipts/s4s_fourway_N50_v1/results.json');paired=read('receipts/s4s_interaction_N50_v1/summary.json')
assert old['status']==legacy['status']==new['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and paired['unchanged_controls']==1800
groups={(r['label'],r['opponent']):r for d in (old,new) for r in d['summary']};rows=[]
for r in legacy['summary']:
    if r['label']=='all_intraday_causal_pickup':groups['all_intraday',r['opponent']]=r
def metrics(r):
    c=r['own_cash'];s=r['own_production'];x=dict(cash=c['cash'],sales=sum(c['sales']),supplies=sum(c['products']),wages=c['hired'],seed_cost=sum(c['seeds']),animal_cost=sum(c['animals']),land_cost=c['land'],projects=r['planner_mean_intraday_activated'])
    for k,ops in dict(move=[1,2,3,4],harvest=[10],water=[9],feed=[15],care=[17],collect=[16],logistics=[5,6]).items():x[k]=sum(s['attempts'][i]-s['no_effect'][i] for i in ops)
    for i in range(9):x['sold_'+str(i)]=c['sold'][i];x['sales_'+str(i)]=c['sales'][i]
    return x
for base in ('all_intraday','all_intraday_insert'):
    for opp in p['identities']:
        before=groups[base,opp];after=groups[base+'_handoff',opp]
        for label,r in ((base,before),(base+'_handoff',after)):
            assert abs(r['own_cash']['cash']-p['summary'][label][opp]['mean_cash'])<1e-6 and r['wins']==p['summary'][label][opp]['wins']
        a,b=metrics(before),metrics(after);rows.append(dict(before=base,after=base+'_handoff',opponent=opp,delta={k:b[k]-a[k] for k in a},before_metrics=a,after_metrics=b))
out=E/'receipts/s4s_channels_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_CHANNEL_COMPARISON',rows=rows,input_hashes=hashes,caveat='Live reconciled policy channel means, not separate causal effects. Old baselines match all current control outcomes. Seed-cluster uncertainty is in interaction report.'),indent=2))
lines=['# S4S 人员—生产—现金','','|背景|对手|现金差|销售差|采购差|工资差|移动差|收获差|收肥差|项目差|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('|'+r['before']+'|'+r['opponent']+'|'+'|'.join(f"{r['delta'][k]:.1f}" for k in ('cash','sales','supplies','wages','move','harvest','collect','projects'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(status='COMPLETE',comparisons=len(rows))))
