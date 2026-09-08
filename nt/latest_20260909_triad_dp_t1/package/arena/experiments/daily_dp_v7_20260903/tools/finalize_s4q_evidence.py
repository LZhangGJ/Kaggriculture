from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4q/evidence';out.mkdir(exist_ok=False)
build=json.loads((E/'native/build/build_receipt.json').read_text())
freeze=json.loads((E/'profiles/s4q/freeze.json').read_text());assert build['binary_sha256']==freeze['build']['binary_sha256']
for rel,h in build['source_hashes'].items():
    assert hashlib.sha256((E/rel).read_bytes()).hexdigest()==h
    assert hashlib.sha256((E/'profiles/s4q/source'/rel).read_bytes()).hexdigest()==h
stage=E/'receipts/s4q_stage_acceptance_v1/acceptance.json';r=json.loads(stage.read_text())
assert r['status']=='STAGE_COMPLETE_NO_PROMOTION_NOT_GOAL_ACCEPTANCE' and not r['promoted'] and not r['final_goal_complete']
files=['reports/S4Q_DAY_RESIDUAL_VALUE_REVIEW_ZH.md','CURRENT_WORK_SCOPE_ZH.md','native/value_schedule_probe.cpp','receipts/s4q_stage_acceptance_v1/acceptance.json']
files += ['tools/'+n for n in ('resume_s4q_prerequisites.py','run_s4q_panel.py','summarize_s4q.py','summarize_s4q_cash_labour.py','build_value_schedule_probe.py','audit_s4q_value_choices.py','write_s4q_acceptance.py','finalize_s4q_evidence.py')]
hashes={}
for rel in files:
    src=E/rel;dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    hashes[rel]=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashes[rel]
(out/'manifest.json').write_text(json.dumps(dict(status='FROZEN_ROUND_EVIDENCE_NO_PROMOTION',binary_sha256=build['binary_sha256'],files=hashes),indent=2))
print(json.dumps(dict(status='FROZEN',files=len(files),goal_complete=False)),flush=True)
