"""Consolidate live evidence; never mark the persistent goal complete."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--round',default='s4k');a=p.parse_args();r=a.round;assert r in ('s4k','s4k2','s4l')
EXP=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    rel=rel.replace('s4k',r)
    f=EXP/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
build=read('native/build/build_receipt.json');freeze=read('profiles/s4k/freeze.json')
assert build['binary_sha256']==freeze['build']['binary_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
configs=read('profiles/s4k/configs.json');assert hashes[f'profiles/{r}/configs.json']==freeze['configs_sha256']
mechanism=read('receipts/s4k_native_mechanisms_v1/acceptance.json');new=read('receipts/s4k_service_mechanism_v1/acceptance.json')
for x in (mechanism,new):assert x['status']=='PASS' and x['build']['binary_sha256']==build['binary_sha256']
pickup_checks=0
if r=='s4l':
    pickup=read('receipts/s4l_incremental_pickup_v1/acceptance.json')
    assert pickup['status']=='PASS' and pickup['build']['binary_sha256']==build['binary_sha256']
    pickup_checks=pickup['checks']
runtime=read('receipts/s4k_runtime_probe_v1/acceptance.json');check=read('receipts/s4k_build_validation_v1/acceptance.json');assert check['status']=='PASS'
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot','g001','g003'):
    x=read(f'receipts/{name}_s4k_official_v1/acceptance.json');assert x['status']=='PASS' and x['build']['binary_sha256']==build['binary_sha256']
panel=read('receipts/s4k_eightway_N50_v1/results.json');audit=read('receipts/s4k_pool_audit_N50_v1/summary.json')
interaction=read('receipts/s4k_interaction_N50_v1/summary.json');execution=read('receipts/s4k_execution_v1/acceptance.json');cash=read('receipts/s4k_cash_chain_N50_v1/summary.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==7200
audit_games,control_games=(3600,3600) if r=='s4l' else (5400,1800)
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==audit_games
assert interaction['unchanged_controls']==control_games and execution['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
opps=[k for k in panel['identities'] if k!='pass'];assert len(opps)==8
winning=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in opps)]
anomalies={k:sum(sum(e['own_production'][k])*e['games'] for e in audit['summary']) for k in ('no_effect','escaped')}
out=EXP/f'receipts/{r}_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NOT_FINAL_ACCEPTANCE',binary_sha256=build['binary_sha256'],live_games=7200,unchanged_prior_controls=control_games,
    repeated_audit_games=audit_games,pickup_mechanism_checks=pickup_checks,service_mechanism_checks=new['checks'],legacy_mechanism_checks=mechanism['native_mechanism_checks'],
    official_sample_games=32,native_runtime_games=runtime['games'],native_max_act_seconds=runtime['hard_max_action_seconds'],
    real_effect_anomalies=anomalies,development_all_eight_above_90=winning,holdout_used=False,final_goal_complete=False,input_hashes=hashes,
    caveats=['50 development seed clusters per cell, each with both seats; no per-seed policy selection.',
             'Current own market projection is conditional, no actual opponent orders or future random events.',
             'Policy-effect checks and native latency are not exhaustive rule or submitted Python latency proofs.'])
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'}))
