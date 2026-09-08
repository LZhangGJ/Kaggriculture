from pathlib import Path
import argparse,hashlib,json,statistics as st
p=argparse.ArgumentParser();p.add_argument('--decision',required=True,choices=['no_promotion','await_holdout']);a=p.parse_args()
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
b=read('native/build/build_receipt.json');freeze=read('profiles/s4r/freeze.json');cfg=read('profiles/s4r/configs.json')
assert b['binary_sha256']==freeze['build']['binary_sha256'] and hashes['profiles/s4r/configs.json']==freeze['configs_sha256']
for rel,h in b['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
assert read('receipts/s4r_prerequisites_v1/acceptance.json')['status']=='PASS_PREREQUISITES_NOT_STRENGTH'
assert read('receipts/s4r_build_validation_v1/acceptance.json')['status']=='PASS'
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot','g001','g003'):
    r=read(f'receipts/{name}_s4r_official_v1/acceptance.json');assert r['status']=='PASS' and r['build']['binary_sha256']==b['binary_sha256']
    if name not in ('g001','g003'):assert read(f'receipts/{name}_s4r_isolation_v1/acceptance.json')['status']=='PASS'
checks={}
for name in ('next_day','day_value','day_scenario','workforce','shared','old','recovery','pickup'):
    r=read(f'receipts/s4r_{name}_mechanisms_v1/acceptance.json');assert r['status']=='PASS'
    checks[name]={k:r[k] for k in ('checks','native_mechanism_checks','continuation_cases','windows','transitions') if k in r}
gap=read('receipts/s4r_cross_day_gap_v1/acceptance.json');assert gap['status']=='PASS_REPRODUCED_OMITTED_OPTION'
nextday=read('receipts/s4r_next_day_audit_v1/acceptance.json');assert nextday['status']=='PASS_ACTUAL_NEXT_DAY_AUDIT_NOT_CAUSAL_ABLATION'
panel=read('receipts/s4r_eightway_N50_v1/results.json');paired=read('receipts/s4r_interaction_N50_v1/summary.json')
audit=read('receipts/s4r_pool_audit_N50_v1/summary.json');channels=read('receipts/s4r_channels_N50_v1/summary.json');runtime=read('receipts/s4r_runtime_probe_v1/acceptance.json')
assert read('receipts/s4r_execution_v1/acceptance.json')['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==7200 and paired['unchanged_controls']==3600
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==3600 and len(channels['rows'])==36
opps=[x for x in panel['identities'] if x!='pass'];assert len(opps)==8
result=dict(status='STAGE_COMPLETE_NOT_GOAL_ACCEPTANCE',decision=a.decision,binary_sha256=b['binary_sha256'],live_games=7200,unchanged_controls=3600,repeated_audit_games=3600,official_games=32,isolation_checks=6,mechanisms=checks,prior_live_audit_games=64,audited_game_days=len(nextday['rows']),changed_next_targets=sum(any(r['target_delta']) for r in nextday['rows']),mean_eight_opponent_development_win_rates={label:st.fmean(s[k]['win_rate'] for k in opps) for label,s in panel['summary'].items()},all_eight_above_90=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in opps)],anomalies={k:sum(sum(x['own_production'][k])*x['games'] for x in audit['summary']) for k in ('no_effect','escaped')},native_latency_games=runtime['games'],native_max_act_seconds=runtime['hard_max_action_seconds'],holdout_used=False,P_used=False,promoted=False,final_goal_complete=False,input_hashes=hashes,report='reports/S4R_CROSS_DAY_CONTINUATION_REVIEW_ZH.md',caveats=['Only public-state conditional next day, not exact unknown future.','Original same-crop residual approximation remains after next-day execution.','Mechanisms/parity/native latency are not final Kaggle CPU deployment acceptance.'])
out=E/'receipts/s4r_stage_acceptance_v1';out.mkdir(exist_ok=False);(out/'acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','mechanisms','caveats')},indent=2),flush=True)
