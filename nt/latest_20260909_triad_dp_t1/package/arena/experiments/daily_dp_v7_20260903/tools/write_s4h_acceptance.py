"""Close a failed interaction round without closing the user's active goal."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads((EXP/p).read_text())
def sha(p):return hashlib.sha256((EXP/p).read_bytes()).hexdigest()
build=read('native/build/build_receipt.json')
for p,h in build['source_hashes'].items():assert sha(p)==h
execution='receipts/s4h1_execution_v1/acceptance.json';official='receipts/g003_s4h1_official_v1/acceptance.json'
panel='receipts/s4h1_eightway_N50_v1/results.json';audit='receipts/s4h1_pool_audit_N50_v1/summary.json'
interaction='receipts/s4h1_interaction_N50_v1/summary.json';previous='receipts/s4g1_stage_acceptance_v1/acceptance.json'
assert read(execution)['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
assert read(official)['status']=='PASS' and read(official)['build']['binary_sha256']==build['binary_sha256']
assert read(previous)['binary_sha256']==build['binary_sha256']
r=read(panel);a=read(audit)
assert r['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(r['rows'])==7200
assert a['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and a['unchanged_result_games']==7200
assert read(interaction)['unchanged_controls']==1800
assert all(sum(x['own_production']['no_effect'])==0 and sum(x['own_production']['escaped'])==0 for x in a['summary'])
winners=[label for label,s in r['summary'].items() if all(s[k]['win_rate']>.9 for k in r['identities'] if k!='pass')]
assert not winners
out=EXP/'receipts/s4h1_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NO_STRENGTH_PROMOTION',goal_acceptance=False,goal_should_remain_active=True,
    binary_sha256=build['binary_sha256'],configuration_match_records=7200,unchanged_controls=1800,
    additional_audit_games_not_independent=7200,holdout_used=False,same_agent_over90_all8=winners,
    input_hashes={p:sha(p) for p in (execution,official,panel,audit,interaction,previous)},
    review='reports/S4H_STARTUP_FEED_INTERACTION_REVIEW_ZH.md',
    next='Compare executable investment portfolios under joint cash, land and staffing resources. No more feed coefficient sweep; no hindsight Oracle.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','configuration_match_records','goal_acceptance')}))
