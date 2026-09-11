"""Consolidate completed live evidence, never mark the persistent goal complete."""
from pathlib import Path
import argparse,hashlib,json
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='s4j2');a=p.parse_args();r=a.round
hashes={}
def read(rel):
    f=EXP/rel;hashes[rel]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
build=read('native/build/build_receipt.json');freeze=read(f'profiles/{r}/freeze.json')
assert build['binary_sha256']==freeze['build']['binary_sha256']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((EXP/rel).read_bytes()).hexdigest()==h
configs=read(f'profiles/{r}/configs.json');assert hashes[f'profiles/{r}/configs.json']==freeze['configs_sha256']
mechanism=read(f'receipts/{r}_mechanisms_v1/acceptance.json')
crop=read(f'receipts/{r}_crop_mechanisms_v2/acceptance.json')
assert mechanism['status']=='PASS' and crop['status']=='PASS_CONDITIONAL_CROP_CALENDAR'
for item in (mechanism,crop):assert item['build']['binary_sha256']==build['binary_sha256']
runtime=read(f'receipts/{r}_runtime_probe_v1/acceptance.json')
checks=read(f'receipts/{r}_build_validation_v1/acceptance.json');assert checks['status']=='PASS'
official=[]
for name in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot','g001','g003'):
    x=read(f'receipts/{name}_{r}_official_v1/acceptance.json');assert x['status']=='PASS' and x['build']['binary_sha256']==build['binary_sha256'];official.append(name)
panel=read(f'receipts/{r}_eightway_N50_v1/results.json');audit=read(f'receipts/{r}_pool_audit_N50_v1/summary.json')
interaction=read(f'receipts/{r}_interaction_N50_v1/summary.json');execution=read(f'receipts/{r}_execution_v1/acceptance.json')
assert panel['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' and len(panel['rows'])==10800
assert audit['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE' and audit['unchanged_result_games']==3600
assert interaction['unchanged_controls']==3600 and execution['status']=='COMPLETED_EXECUTION_NOT_GOAL_ACCEPTANCE'
opps=[k for k in panel['identities'] if k!='pass'];assert len(opps)==8
fully_winning=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.9 for k in opps)]
anomalies={key:sum(sum(e['own_production'][key])*e['games'] for e in audit['summary']) for key in ('no_effect','escaped')}
out=EXP/f'receipts/{r}_stage_acceptance_v1';out.mkdir(exist_ok=False)
result=dict(status='STAGE_COMPLETE_NOT_FINAL_ACCEPTANCE',binary_sha256=build['binary_sha256'],
    live_games=10800,unchanged_prior_controls=3600,repeated_audit_games=3600,
    crop_checks=crop['checks'],controlled_crop_games=crop['controlled_engine_games'],legacy_mechanism_checks=mechanism['native_mechanism_checks'],
    official_sample_games=32,official_opponents=official,native_runtime_games=runtime['games'],native_max_act_seconds=runtime['hard_max_action_seconds'],
    real_effect_anomalies=anomalies,development_all_eight_above_90=fully_winning,
    holdout_used=False,final_goal_complete=False,promotion_requires='Fresh seeds, same frozen policy >90% each opponent, official and online latency acceptance',
    input_hashes=hashes,caveats=['Conditional crop forecast is not guaranteed future cash or exact multiplayer logistics.',
    'The failed v1 crop check had an uninitialized test-controller day; preserved rather than overwritten.',
    'The same seed is shared by both seats; 100 games per cell means 50 independent seed clusters.',
    'This is not Candidate hindsight Oracle. Opponents respond throughout each full live game.'])
(out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'}))
