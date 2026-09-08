"""Freeze stage receipts; engineering completion is not win-rate acceptance."""
from pathlib import Path
import hashlib
import json

EXP=Path(__file__).resolve().parents[1]


def read(rel): return json.loads((EXP/rel).read_text())
def sha(rel): return hashlib.sha256((EXP/rel).read_bytes()).hexdigest()


def main():
    build=read('native/build/build_receipt.json')
    for rel,h in build['source_hashes'].items(): assert sha(rel)==h,rel
    paths=['receipts/s4e1_build_validation_v1/acceptance.json',
           'receipts/g001_s4e1_official_v1/acceptance.json',
           'receipts/g003_s4e1_official_v1/acceptance.json']
    for prefix in ('boatlee','kaito','lynn','ecobot','fieldbook','three_day'):
        for kind in ('official','isolation'):
            paths.append(f'receipts/{prefix}_s4e1_{kind}_v1/acceptance.json')
    for rel in paths:
        r=read(rel);assert r['status']=='PASS'
        assert r['build']['binary_sha256']==build['binary_sha256']
    panel_path='receipts/s4e1_nineway_N50_v1/results.json'
    audit_path='receipts/s4e1_pool_audit_N50_v1/summary.json'
    effect_path='receipts/s4e1_execution_effect_v1/summary.json'
    daily_path='receipts/s4e1_daily_cash_N50_v1/summary.json'
    panel=read(panel_path);audit=read(audit_path);effect=read(effect_path);daily=read(daily_path)
    assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==8100
    assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==8100
    assert effect['unchanged_control_games']==2700 and daily['status']=='PASS_PAIRED_DAILY_ACCOUNTING'
    assert all(sum(r['own_production']['no_effect'])==0 and sum(r['own_production']['escaped'])==0 for r in audit['summary'])
    required=[x for x in panel['identities'] if x!='pass'];assert len(required)==8
    winners=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in required)]
    assert not winners
    result=dict(status='STAGE_COMPLETE_NO_STRENGTH_PROMOTION',engineering_checks='PASS_FINITE_TESTS',
                goal_acceptance=False,goal_should_remain_active=True,holdout_used=False,
                same_agent_over90_all8=winners,unique_configuration_match_records=8100,
                additional_audit_matches_not_independent=8100,unchanged_controls=2700,
                binary_sha256=build['binary_sha256'],
                input_hashes={rel:sha(rel) for rel in paths+[panel_path,audit_path,effect_path,daily_path]},
                review='reports/S4E_INTRADAY_PRODUCTION_REVIEW_ZH.md',
                next_pre_register='reports/S4F_DECLARED_COMMITMENTS_PRE_REGISTER_ZH.md',
                next_action='Read-only actual-state pending-project audit before changing policy; no suffix Oracle or seed-specific rules.')
    out=EXP/'receipts/s4e1_stage_acceptance_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('status','goal_acceptance','goal_should_remain_active','unique_configuration_match_records','unchanged_controls')}))


if __name__=='__main__': main()
