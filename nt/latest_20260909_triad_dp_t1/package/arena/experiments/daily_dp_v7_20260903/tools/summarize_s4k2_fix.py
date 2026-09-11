"""Verify the corrected run changed no config and only the guard source."""
from pathlib import Path
import hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1]
old=json.loads((EXP/'profiles/s4k/freeze.json').read_text());new=json.loads((EXP/'profiles/s4k2/freeze.json').read_text())
assert old['configs_sha256']==new['configs_sha256']
changes=[k for k,h in old['build']['source_hashes'].items() if new['build']['source_hashes'][k]!=h]
assert changes==['native/service_recovery.hpp'],changes
test=json.loads((EXP/'receipts/s4k2_regression_before_fix_v1/acceptance.json').read_text());assert test['status']=='FAIL' and 'unreserved terminal DROP SELL was deleted' in test['error']
fixed=json.loads((EXP/'receipts/s4k2_service_mechanism_v1/acceptance.json').read_text());assert fixed['status']=='PASS' and fixed['source_sha256']==test['source_sha256']
panels=[json.loads((EXP/f'receipts/{r}_eightway_N50_v1/results.json').read_text()) for r in ('s4k','s4k2')]
assert all(x['status']=='COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE' for x in panels)
tables=[{(x['variant'],x['opponent'],x['seed'],x['seat']):x for x in p['rows']} for p in panels];assert tables[0].keys()==tables[1].keys()
rows=[]
for label in panels[0]['summary']:
    for opp in panels[0]['identities']:
        before=panels[0]['summary'][label][opp];after=panels[1]['summary'][label][opp]
        keys=[k for k in tables[0] if k[0]==label and k[1]==opp]
        delta={metric:st.fmean(float(tables[1][k][metric])-float(tables[0][k][metric]) for k in keys) for metric in ('cash','opponent_cash','margin','win')}
        rows.append(dict(label=label,opponent=opp,games=len(keys),before_cash=before['mean_cash'],after_cash=after['mean_cash'],before_wins=before['wins'],after_wins=after['wins'],delta=delta))
out=EXP/'receipts/s4k2_guard_regression_comparison_v1';out.mkdir(exist_ok=False)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_SAME_CONFIG_GUARD_COMPARISON',changed_runtime_sources=changes,same_config=True,
    regression_test_failed_before_and_passed_after=True,rows=rows,goal_achieved=False),indent=2))
print(json.dumps(dict(status='COMPLETE',changed_runtime_sources=changes,compared_games=len(tables[0]))))
