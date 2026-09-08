"""Freeze S4F evidence, without promoting a weak agent or completing the goal."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads((EXP/p).read_text())
def sha(p):return hashlib.sha256((EXP/p).read_bytes()).hexdigest()
build=read('native/build/build_receipt.json')
paths=['receipts/s4f1_build_validation_v1/acceptance.json','receipts/g001_s4f1_official_v1/acceptance.json','receipts/g003_s4f1_official_v1/acceptance.json']
for prefix in ('boatlee','kaito','lynn','ecobot','fieldbook','three_day'):
    paths += [f'receipts/{prefix}_s4f1_{kind}_v1/acceptance.json' for kind in ('official','isolation')]
for p in paths:
    r=read(p);assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
panel='receipts/s4f1_sixway_N50_v1/results.json';audit='receipts/s4f1_pool_audit_N50_v1/summary.json'
paired='receipts/s4f1_paired_summary_v1/summary.json';observation='receipts/s4f0_declared_N50_v1/summary.json'
early='receipts/s4f_early_cash_N50_v1/summary.json'
r=read(panel);a=read(audit)
assert r['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(r['rows'])==5400
assert a['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and a['unchanged_result_games']==2700
assert a['all_ledgers_reconciled'] and read(paired)['unchanged_controls']==2700
assert all(sum(x['own_production']['no_effect'])==0 and sum(x['own_production']['escaped'])==0 for x in a['summary'])
winners=[label for label,s in r['summary'].items() if all(s[k]['win_rate']>.9 for k in r['identities'] if k!='pass')]
assert not winners
out=EXP/'receipts/s4f1_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NO_STRENGTH_PROMOTION',goal_acceptance=False,goal_should_remain_active=True,
    engineering_checks='PASS_FINITE_TESTS',binary_sha256=build['binary_sha256'],configuration_match_records=5400,
    unchanged_controls=2700,additional_audit_games_not_independent=2700,holdout_used=False,
    input_hashes={p:sha(p) for p in paths+[panel,audit,paired,observation,early]},
    review='reports/S4F_DECLARED_VALUE_REVIEW_ZH.md',same_agent_over90_all8=winners,
    next='Test autonomous startup using current planner; no true future or hindsight opening selection.')
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
