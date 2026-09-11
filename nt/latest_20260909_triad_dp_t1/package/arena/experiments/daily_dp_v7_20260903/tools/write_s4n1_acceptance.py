from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    f=EXP/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
build=read('native/build/build_receipt.json');freeze=read('profiles/s4n1/freeze.json')
assert build['binary_sha256']==freeze['build']['binary_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
configs=read('profiles/s4n1/configs.json');assert hashes['profiles/s4n1/configs.json']==freeze['configs_sha256']
checks={}
for k in ('old','shared','recovery','pickup','workforce'):
    x=read(f'receipts/s4n1_{k}_mechanisms_v1/acceptance.json');assert x['status']=='PASS' and x['build']['binary_sha256']==build['binary_sha256']
    checks[k]=x.get('checks',x.get('native_mechanism_checks'))
prereq=read('receipts/s4n1_prerequisites_v1/acceptance.json');assert prereq['status']=='PASS_PREREQUISITES_NOT_STRENGTH'
official=read('receipts/s4n1_build_validation_v1/acceptance.json');assert official['status']=='PASS'
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot','g001','g003'):
    x=read(f'receipts/{name}_s4n1_official_v1/acceptance.json');assert x['status']=='PASS' and x['build']['binary_sha256']==build['binary_sha256']
panel=read('receipts/s4n1_eightway_N50_v1/results.json');audit=read('receipts/s4n1_pool_audit_N50_v1/summary.json')
interaction=read('receipts/s4n1_interaction_N50_v1/summary.json');execution=read('receipts/s4n1_execution_v1/acceptance.json')
cash=read('receipts/s4n1_cash_labour_N50_v1/summary.json');runtime=read('receipts/s4n1_runtime_probe_v1/acceptance.json')
o=read('receipts/s4n1_O50_confirmation_v1/summary.json');oa=read('receipts/s4n1_pool_audit_O50_v1/summary.json')
assert o['status']=='COMPLETE_UNSEEN_SEED_FIXED_POLICY_COMPARISON' and o['games']==1800
assert oa['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and oa['unchanged_result_games']==1800
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==3600
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==1800
assert interaction['unchanged_controls']==1800 and execution['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
opps=[k for k in panel['identities'] if k!='pass'];assert len(opps)==8
winning=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in opps)]
anomalies={k:sum(sum(e['own_production'][k])*e['games'] for e in audit['summary']) for k in ('no_effect','escaped')}
o_anomalies={k:sum(sum(e['own_production'][k])*e['games'] for e in oa['summary']) for k in ('no_effect','escaped')}
out=EXP/'receipts/s4n1_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NOT_FINAL_ACCEPTANCE',binary_sha256=build['binary_sha256'],live_games=3600,
    unchanged_prior_controls=1800,repeated_audit_games=1800,mechanism_checks=checks,official_sample_games=32,
    native_runtime_games=runtime['games'],native_max_act_seconds=runtime['hard_max_action_seconds'],real_effect_anomalies=anomalies,
    development_all_eight_above_90=winning,holdout_used=True,holdout_O_games=1800,repeated_O_audit_games=1800,
    holdout_O_real_effect_anomalies=o_anomalies,P_used=False,promoted=False,final_goal_complete=False,input_hashes=hashes,
    caveats=['Paired 50 seed clusters, each two seats. No per-seed or opponent identity policy selection.',
             'Full-game completed outcomes do not make every bookkeeping delta an isolated causal effect.',
             'Primitive effects and native latency are not exhaustive rule or final submitted Python timing proofs.'])
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'}))
