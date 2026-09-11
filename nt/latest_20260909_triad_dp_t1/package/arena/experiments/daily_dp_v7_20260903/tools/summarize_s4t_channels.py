"""Joint economic/workforce attribution from reconciled live-game ledgers."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st

E=Path(__file__).resolve().parents[1];hashes={}
cli=argparse.ArgumentParser();cli.add_argument('--round',choices=['s4t','s4u'],default='s4t');round_name=cli.parse_args().round
def read(rel):
    path=E/rel;hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text())
old=read('receipts/s4m1_pool_audit_N50_v1/summary.json')
legacy=read('receipts/s4l_pool_audit_N50_v1/summary.json')
new=read(f'receipts/{round_name}_pool_audit_N50_v1/summary.json')
panel=read(f'receipts/{round_name}_fiveway_N50_v1/results.json')
paired=read(f'receipts/{round_name}_comparison_N50_v1/summary.json')
assert old['status']==legacy['status']==new['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
assert paired['unchanged_controls']==1800 and new['unchanged_result_games']==2700
groups={(r['label'],r['opponent']):r for d in (old,new) for r in d['summary']}
for r in legacy['summary']:
    if r['label']=='all_intraday_causal_pickup':groups['all_intraday',r['opponent']]=r

def metrics(r):
    c=r['own_cash'];s=r['own_production']
    out=dict(cash=c['cash'],sales=sum(c['sales']),supplies=sum(c['products']),wages=c['hired'],
        seed_cost=sum(c['seeds']),animal_cost=sum(c['animals']),land_cost=c['land'],
        intraday_projects=r['planner_mean_intraday_activated'],
        no_effect=sum(s['no_effect']),escaped=sum(s['escaped']),planted=sum(s['planted']),placed=sum(s['placed']),
        output_generated=sum(s['generated']),output_acquired=sum(s['acquired']),
        terminal_products=sum(s['end_private'][:9]),terminal_unplaced_animals=sum(s['end_private'][9:]),
        terminal_field_products=sum(s['end_field'][:9]),capacity_loss=sum(s['drop_loss'])+sum(s['eod_loss']))
    for k,ops in dict(move=[1,2,3,4],harvest=[10],water=[9],feed=[15],care=[17],collect=[16],logistics=[5,6]).items():
        out[k]=sum(s['attempts'][i]-s['no_effect'][i] for i in ops)
    for i in range(9):out['sold_'+str(i)]=c['sold'][i];out['sales_'+str(i)]=c['sales'][i]
    return out

all_metrics=[]
for label in panel['configurations']:
    for opp in panel['identities']:
        group=groups[label,opp];ref=panel['summary'][label][opp]
        assert abs(group['own_cash']['cash']-ref['mean_cash'])<1e-6 and group['wins']==ref['wins']
        all_metrics.append(dict(label=label,opponent=opp,metrics=metrics(group)))
pairs=[(base,after) for base in ('all_intraday','all_intraday_insert') for after in ('full_workers25','full_chain','full_chain_autonomous')]
pairs += [('full_workers25','full_chain'),('full_chain','full_chain_autonomous')]
comparisons=[]
for before,after in pairs:
    for opp in panel['identities']:
        a,b=metrics(groups[before,opp]),metrics(groups[after,opp])
        comparisons.append(dict(before=before,after=after,opponent=opp,delta={k:b[k]-a[k] for k in a}))

# Read counters already captured during the same 2,700-game ledger repeat;
# this does not simulate again and these diagnostics never enter the policy.
activity=[];no_effect_examples=[]
counter_names=['shared_service_plots','regret_trials','regret_improvements',
    'preparation_pipeline_checks','preparation_pipeline_actions','preparation_actions',
    'step_recoord_checks','step_recoord_rebuilds','step_recoord_reassigned_groups',
    'resource_exchange_checks','resource_exchange_applied','resource_exchange_local_steps_saved',
    'intraday_proposals','intraday_activated','intraday_unfilled','intraday_cancelled',
    'intraday_purchase_orders','service_checks','service_recovered_feed','service_recovered_fertilize',
    'service_buys','service_financed','service_deposits','terminal_schedule_evaluations',
    'terminal_schedule_switches','overflow_dispatch_units']
for group in new['summary']:
    label,opp=group['label'],group['opponent'];path=E/f'receipts/{round_name}_pool_audit_N50_v1/{label}_{opp}.json.gz'
    hashes[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path,'rt',encoding='utf8') as f:data=json.load(f)
    rows=data['rows'];assert len(rows)==100
    final=[r['planning'][-1] for r in rows]
    counters={k:dict(mean=st.fmean(x[k] for x in final),active_games=sum(x[k]>0 for x in final))
              for k in counter_names if all(k in x for x in final)}
    activity.append(dict(label=label,opponent=opp,counters=counters,
        missing_counters=[k for k in counter_names if k not in counters]))
    for r in rows:
        for example in r['no_effect_examples']:
            if example[1]==r['seat']:
                no_effect_examples.append(dict(label=label,opponent=opp,seed=r['seed'],seat=r['seat'],example=example))
out=E/f'receipts/{round_name}_channels_N50_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_ACTUAL_JOINT_CHANNEL_COMPARISON',
    metrics=all_metrics,comparisons=comparisons,activity=activity,no_effect_examples=no_effect_examples,
    input_hashes=hashes,holdout_used=False,
    caveat='Reconciled realized income, effort and output, not independent causal contributions. Missing audit counters are unknown, never zero. No hidden state used by policy.'),indent=2))
lines=[f'# {round_name.upper()} 经营—人员—生产—变现一起看','','|配置|对手|现金|销售|采购|工资|移动|收获|新种植|新放养|日内项目|无效动作|逃跑|',
       '|---|---|'+'---:|'*11]
for r in all_metrics:
    lines.append('|'+r['label']+'|'+r['opponent']+'|'+'|'.join(f"{r['metrics'][k]:.1f}" for k in
        ('cash','sales','supplies','wages','move','harvest','planted','placed','intraday_projects','no_effect','escaped'))+'|')
lines+=['','数字为每场均值；非同价值商品的数量不能直接相加等同利润。资源残余不作为全局硬错误；无效动作需逐项查原因。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(dict(status='COMPLETE',groups=len(all_metrics),comparisons=len(comparisons),activity_groups=len(activity),no_effect_examples=len(no_effect_examples))),flush=True)
