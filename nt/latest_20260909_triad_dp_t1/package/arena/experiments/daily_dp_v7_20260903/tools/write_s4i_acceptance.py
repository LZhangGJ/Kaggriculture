from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
paths=[f'receipts/{s}' for s in ('s4i2_execution_v1/acceptance.json','s4i2_mechanisms_v1/acceptance.json',
 's4i2_build_validation_v1/acceptance.json','g001_s4i2_official_v1/acceptance.json','g003_s4i2_official_v1/acceptance.json',
 's4i2_eightway_N50_v1/results.json','s4i2_pool_audit_N50_v1/summary.json','s4i2_interaction_N50_v1/summary.json','s4i2_cash_chain_N50_v1/summary.json')]
data={p:json.loads((EXP/p).read_text()) for p in paths}
panel=data[paths[5]];audit=data[paths[6]];interaction=data[paths[7]]
assert data[paths[0]]['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
assert all(data[p]['status']=='PASS' for p in paths[1:5])
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==7200
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==3600
assert interaction['unchanged_controls']==3600
for x in audit['summary']:assert sum(x['own_production']['no_effect'])==sum(x['own_production']['escaped'])==0
champions=[k for k,v in panel['summary'].items() if all(s['win_rate']>.9 for opp,s in v.items() if opp!='pass')]
assert not champions
out=EXP/'receipts/s4i2_stage_acceptance_v1';out.mkdir(exist_ok=False)
(out/'acceptance.json').write_text(json.dumps(dict(status='STAGE_COMPLETE_NO_STRENGTH_PROMOTION',goal_acceptance=False,
    goal_should_remain_active=True,binary_sha256=panel['build']['binary_sha256'],configuration_match_records=7200,
    unchanged_controls=3600,additional_repeat_audits=3600,holdout_used=False,same_agent_over90_all8=champions,
    input_hashes={p:hashlib.sha256((EXP/p).read_bytes()).hexdigest() for p in paths},
    review='reports/S4I_JOINT_PORTFOLIO_REVIEW_ZH.md',
    next='S4J: restore startup/portfolio switch independence, audit and repair new-versus-existing project cash calendars. No hindsight Oracle, no feed-weight sweep.'),indent=2))
print('STAGE_COMPLETE_NO_STRENGTH_PROMOTION')
