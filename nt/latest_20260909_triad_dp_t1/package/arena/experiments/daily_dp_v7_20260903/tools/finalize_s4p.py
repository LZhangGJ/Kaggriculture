from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((E/'native/build/build_receipt.json').read_text())
prior=json.loads((E/'profiles/s4n1/freeze.json').read_text())['build']
assert build['binary_sha256']==prior['binary_sha256']==sha(Path(build['binary']))
assert build['source_hashes']['native/vendor/simulator.cpp']==prior['source_hashes']['native/vendor/simulator.cpp']
for rel,h in build['source_hashes'].items():assert sha(E/rel)==h
receipts={
    'delivery':'receipts/s4o_delivery_audit_v1/summary.json',
    'mechanisms':'receipts/s4p_day_scenario_mechanisms_v3/acceptance.json',
    'probe':'receipts/s4p_day_probe_build_v2/acceptance.json',
    'live_audit':'receipts/s4p_day_schedule_audit_v1/acceptance.json',
    'prior_official_and_mechanisms':'receipts/s4n1_prerequisites_v1/acceptance.json'}
data={k:json.loads((E/v).read_text()) for k,v in receipts.items()}
assert data['mechanisms']['status']=='PASS' and data['mechanisms']['public_windows']==1920
assert data['live_audit']['status']=='PASS_READONLY_DAY_SCENARIOS_NOT_STRENGTH' and len(data['live_audit']['games'])==64
assert data['prior_official_and_mechanisms']['build']['binary_sha256']==build['binary_sha256']
assert data['probe']['status']=='PASS_BUILD' and sha(Path(data['probe']['binary']))==data['probe']['binary_sha256']
for section in ('mechanisms','probe'):
    for rel,h in data[section]['source_hashes'].items():assert sha(E/rel)==h
out=E/'profiles/s4p';out.mkdir(exist_ok=False)
files=set(data['mechanisms']['source_hashes'])|set(data['probe']['source_hashes'])
files.update('tools/'+n for n in ('test_observed_day_scenario.py','build_day_schedule_probe.py','audit_day_schedule_probe.py','finalize_s4p.py','build_native.py'))
files.update('reports/'+n for n in ('S4O_DELIVERY_FINANCING_REVIEW_ZH.md','S4P_OBSERVABLE_DAY_SCENARIO_PRE_REGISTER_ZH.md','S4P_DAY_CONSEQUENCE_REVIEW_ZH.md'))
for rel in sorted(files):
    dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,dest)
shutil.copy2(E/'profiles/s4n1/configs.json',out/'unchanged_configs.json')
manifest=dict(status='FROZEN_PUBLIC_DAY_SCENARIO_FOUNDATION_NO_PROMOTION',build=build,source_hashes={rel:sha(E/rel) for rel in sorted(files)},receipts={k:dict(path=v,sha256=sha(E/v)) for k,v in receipts.items()},policy_changed=False,online_selection_switch_implemented=False,official_binary_unchanged=True,goal_complete=False)
(out/'freeze.json').write_text(json.dumps(manifest,indent=2))
dest=E/'receipts/s4p_stage_acceptance_v1';dest.mkdir(exist_ok=False)
(dest/'acceptance.json').write_text(json.dumps(dict(**manifest,window_count=data['mechanisms']['public_windows'],scenario_transitions=data['mechanisms']['scenario_transitions'],unchanged_live_games=64,groups=data['live_audit']['groups']),indent=2))
print(json.dumps(dict(status=manifest['status'],policy_changed=False,goal_complete=False,source_files=len(files))),flush=True)
