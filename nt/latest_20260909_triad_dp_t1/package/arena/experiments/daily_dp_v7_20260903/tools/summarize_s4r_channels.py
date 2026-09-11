"""Joined actual production/cash channels; never a new planning objective."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
old=read('receipts/s4q_pool_audit_N50_v1/summary.json');new=read('receipts/s4r_pool_audit_N50_v1/summary.json')
panel=read('receipts/s4r_eightway_N50_v1/results.json');paired=read('receipts/s4r_interaction_N50_v1/summary.json')
assert old['status']==new['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and paired['unchanged_controls']==3600
groups={(r['label'],r['opponent']):r for meta in (old,new) for r in meta['summary']};rows=[]
def metrics(r):
    c=r['own_cash'];p=r['own_production'];x=dict(cash=c['cash'],sales=sum(c['sales']),supplies=sum(c['products']),wages=c['hired'],seed_cost=sum(c['seeds']),animal_cost=sum(c['animals']),land_cost=c['land'],projects=r['planner_mean_intraday_activated'])
    for k,ops in dict(move=[1,2,3,4],harvest=[10],water=[9],feed=[15],care=[17],logistics=[5,6]).items():x[k]=sum(p['attempts'][i]-p['no_effect'][i] for i in ops)
    for i in range(9):x['sold_'+str(i)]=c['sold'][i];x['sales_'+str(i)]=c['sales'][i]
    return x
for base in ('all_intraday','all_intraday_insert'):
    for before,after in ((base+'_value',base+'_value_next'),(base+'_value_next',base+'_value_next_public')):
        for opp in panel['identities']:
            a,b=groups[before,opp],groups[after,opp]
            for label,r in ((before,a),(after,b)):
                assert abs(r['own_cash']['cash']-panel['summary'][label][opp]['mean_cash'])<1e-6 and r['wins']==panel['summary'][label][opp]['wins']
            ma,mb=metrics(a),metrics(b)
            rows.append(dict(before=before,after=after,opponent=opp,delta={k:mb[k]-ma[k] for k in ma},before_metrics=ma,after_metrics=mb))
out=E/'receipts/s4r_channels_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_CHANNEL_COMPARISON',rows=rows,input_hashes=hashes,caveat='Full-policy means from reconciled live ledgers; not separate causal effects. Win/margin uncertainty is in paired seed-cluster report. Historical S4Q controls were reproduced on the new build before reuse.'),indent=2))
lines=['# S4R 人员—生产—现金','','|之前|之后|对手|现金差|销售差|采购差|工资差|移动差|收获差|项目差|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('|'+r['before']+'|'+r['after']+'|'+r['opponent']+'|'+'|'.join(f"{r['delta'][k]:.1f}" for k in ('cash','sales','supplies','wages','move','harvest','projects'))+'|')
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(dict(status='COMPLETE',comparisons=len(rows))))
