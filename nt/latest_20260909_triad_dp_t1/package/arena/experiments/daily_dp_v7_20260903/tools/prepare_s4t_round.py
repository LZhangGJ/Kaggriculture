from pathlib import Path
import hashlib,json,shutil,re
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4t';out.mkdir(exist_ok=False)
groups={
 'scheduling':['preparation_work','split_service_jobs','regret_schedule','regret_compile','shared_task_atoms_v2','stepwise_recoordination','preparation_pipeline_v2','schedule_value_compare','resource_aware_exchange','shared_service_insertions','idle_task_handoff'],
 'workforce':['insertion_hire_estimate','regret_hire_estimate','intraday_future_workforce'],
 'logistics':['capacity_hauling','terminal_deposit_schedule','overflow_idle_dispatch','preparation_spawn_guard','midroute_delivery','recover_service_inputs','procure_service_inputs','finance_service_inputs','incremental_pickup_repair'],
 'consequences':['day_consequence_compare','day_value_replan_next_day']}
workers=sum(groups.values(),[]);assert len(set(workers))==25
links=['intraday_admission','intraday_procurement','intraday_declared_value','sell_deposits','fix_liquidity','day_value_public_supply'];macro=['autonomous_start','joint_investment_portfolio','portfolio_crop_calendar']
old=json.loads((E/'profiles/s4s/configs.json').read_text());cfg={k:old[k] for k in ('all_intraday','all_intraday_insert')}
cfg['full_workers25']=dict(cfg['all_intraday'],**{k:True for k in workers})
cfg['full_chain']=dict(cfg['full_workers25'],**{k:True for k in links})
cfg['full_chain_autonomous']=dict(cfg['full_chain'],**{k:True for k in macro})
source=(E/'native/policy.hpp').read_text()
for k in workers+links+macro:assert re.search(r'\b'+re.escape(k)+r'\s*=',source),k
b=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in b['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
p=out/'configs.json';p.write_text(json.dumps(cfg,indent=2));doc=E/'reports/S4T_FULL_CHAIN_PRE_REGISTER_ZH.md';shutil.copy2(doc,out/doc.name)
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=b,groups=groups,link_switches=links,macro_switches=macro,configs_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),development=[20262701,20262750],holdout_used=False,runtime_source_changed=False),indent=2));print(json.dumps(dict(status='FROZEN',configs=list(cfg),specialized_switch_count=len(workers))),flush=True)
