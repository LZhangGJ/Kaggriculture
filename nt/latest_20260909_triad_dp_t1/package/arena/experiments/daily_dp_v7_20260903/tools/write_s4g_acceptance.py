"""S4G engineering and failed strength result, independently checked receipts."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads((EXP/p).read_text())
def sha(p):return hashlib.sha256((EXP/p).read_bytes()).hexdigest()
build=read('native/build/build_receipt.json')
for rel,h in build['source_hashes'].items():assert sha(rel)==h
paths=['receipts/s4g1_build_validation_v1/acceptance.json','receipts/g001_s4g1_official_v1/acceptance.json','receipts/g003_s4g1_official_v1/acceptance.json','receipts/s4g1_mechanisms_v1/acceptance.json']
for prefix in ('boatlee','kaito','lynn','ecobot','fieldbook','three_day'):
    paths += [f'receipts/{prefix}_s4g1_{kind}_v1/acceptance.json' for kind in ('official','isolation')]
for p in paths:
    r=read(p);assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
panel='receipts/s4g1_eightway_N50_v1/results.json';audit='receipts/s4g1_pool_audit_N50_v1/summary.json'
paired='receipts/s4g1_paired_summary_v1/summary.json';early='receipts/s4g1_cash_chain_N50_v1/summary.json';initial='receipts/s4g1_initial_plan_v1/summary.json'
r=read(panel);a=read(audit)
assert r['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(r['rows'])==7200
assert a['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and a['unchanged_result_games']==5400
assert read(paired)['unchanged_controls']==1800 and read(early)['status']=='COMPLETE_ACTUAL_CASH_CHAIN_AUDIT'
assert all(sum(x['own_production']['no_effect'])==0 and sum(x['own_production']['escaped'])==0 for x in a['summary'])
winners=[label for label,s in r['summary'].items() if all(s[k]['win_rate']>.9 for k in r['identities'] if k!='pass')]
assert not winners
out=EXP/'receipts/s4g1_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NO_STRENGTH_PROMOTION',engineering_checks='PASS_FINITE_TESTS',goal_acceptance=False,goal_should_remain_active=True,
    binary_sha256=build['binary_sha256'],configuration_match_records=7200,unchanged_controls=1800,
    additional_audit_games_not_independent=5400,holdout_used=False,source_oracles_used=False,
    input_hashes={p:sha(p) for p in paths+[panel,audit,paired,early,initial]},same_agent_over90_all8=winners,
    review='reports/S4G_AUTONOMOUS_START_REVIEW_ZH.md',next='Pre-register autonomous-start x existing-animal-feed valuation interaction, no coefficient sweep.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','goal_acceptance','configuration_match_records','unchanged_controls')}))
