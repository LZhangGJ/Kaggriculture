"""Close a completed development stage honestly, including failed gates."""
from pathlib import Path
import hashlib,json
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
 p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
run=read('receipts/s4u_execution_v1/acceptance.json');assert run['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE'
audit=read('receipts/s4u_pool_audit_N50_v1/summary.json');assert audit['all_ledgers_reconciled'] and audit['unchanged_result_games']==2700
same=read('receipts/s4u_comparison_N50_v1/summary.json');assert same['unchanged_old_completed_games']==3600
exact=read('receipts/s4u_production_equivalence_v1/acceptance.json');assert exact['action_checks']==80528 and exact['reset_checks']==2800
fail=read('receipts/s4u_escape_trace_v1/acceptance.json');assert len(fail['rows'])==4
escaped=sum(sum(g['own_production']['escaped'])*g['games'] for g in audit['summary'])
ineffective=sum(sum(g['own_production']['no_effect'])*g['games'] for g in audit['summary'])
assert escaped==4 and ineffective==0
build=run['build']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/'profiles/s4u/source'/rel).read_bytes()).hexdigest()==h
official=[]
for prefix in ('g001','g003','boatlee','kaito','lynn','fieldbook','three_day','ecobot'):
 r=read(f'receipts/{prefix}_s4u_official_v1/acceptance.json');assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256'];official.append(prefix)
out=E/'receipts/s4u_stage_acceptance_v1';out.mkdir(exist_ok=False)
(out/'acceptance.json').write_text(json.dumps(dict(status='COMPLETE_STAGE_REJECTED_FOR_PROMOTION',input_hashes=hashes,build=build,
 scope='full_worker_chain_and_autonomous_economy',full_panel_games=4500,ledger_repeat_games=2700,
 no_effect_unit_actions=ineffective,unplanned_escaped_animals=escaped,official_opponents=official,
 pure_speed_equivalence_passed=True,strength_gate=False,anomaly_gate=False,online_python_latency_gate='NOT_TESTED',
 promoted=False,holdout_used=False,final_goal_acceptance=False,next='S4V live market delivery and missing feed candidate coverage; no hindsight Oracle'),indent=2))
print('S4U CLOSED: PERFORMANCE PASS; STRENGTH/ANOMALY FAIL; GOAL ACTIVE')
