"""Close a diagnostic stage without promoting conditional predictions."""
from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1]
def read(p):return json.loads((E/p).read_text())
trace=read('receipts/s5c_obligations_v1/acceptance.json');loc=read('receipts/s5c_obligation_summary_v1/acceptance.json');cap=read('receipts/s5c_capacity_v1/acceptance.json')
assert trace['status']=='PASS_UNCHANGED_FAILURE_TRACE' and loc['status']=='PASS_TRACE_LOCALIZATION' and cap['status']=='PASS_OBSERVATION_ONLY_ALTERNATIVES'
assert all(r['control_equal'] for r in cap['games']) and len(trace['rows'])==len(cap['games'])==6
build=read('native/build/build_receipt.json');assert trace['build']['production']==cap['build']['production']==build
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
out=E/'receipts/s5c_stage_acceptance_v1';out.mkdir(exist_ok=False);frozen=E/'profiles/s5c';frozen.mkdir(exist_ok=False)
files=['native/execution_obligation_probe.cpp','native/compile_capacity_probe.cpp','tools/audit_s5c_obligations.py','tools/summarize_s5c_obligations.py','tools/probe_s5c_capacity.py','tools/close_s5c.py']
hashes={}
for rel in files:
 dst=frozen/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,dst);hashes[rel]=hashlib.sha256(dst.read_bytes()).hexdigest()
rows=[]
for r in cap['rows']:
 keep=r['trials'][0];known=[t for t in r['trials'] if t['conditional']['residual_known']]
 best=max(known,key=lambda t:t['conditional']['residual_score']) if known and keep['conditional']['residual_known'] else keep
 has_zero=any(t['conditional']['missed_feed_today']==0 and t['conditional']['escaped_today']==0 for t in r['trials'])
 rows.append(dict(case=r['case'],step=r['step'],trial_count=len(r['trials']),keep_dropped=keep['dropped'],zero_feed_loss_available=has_zero,
  keep=keep['conditional'],estimate_selected=best['name'],selected=best['conditional'],seconds=r['seconds']))
receipt=dict(status='COMPLETE_DIAGNOSTIC_STAGE_NOT_PROMOTED',production_unchanged=True,build=build,source_hashes=hashes,
 unchanged_trace_games=6,unchanged_capacity_games=6,all_complete_against_original_panel=True,lost_animals=loc['lost_animals'],
 observed_states=len(rows),conditional_alternatives=sum(r['trial_count'] for r in rows),zero_feed_loss_available_states=sum(r['zero_feed_loss_available'] for r in rows),
 states_with_fewer_feed_failures_selected=sum(r['selected']['missed_feed_today']+r['selected']['escaped_today']<r['keep']['missed_feed_today']+r['keep']['escaped_today'] for r in rows),
 rows=rows,holdout_used=False,promoted=False,goal_complete=False,
 limits='Failure-enriched old development seeds, correlated seats. Conditional end-of-day + approximate residual; no actual alternative full-game wins asserted.')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','rows','source_hashes')},indent=2))
