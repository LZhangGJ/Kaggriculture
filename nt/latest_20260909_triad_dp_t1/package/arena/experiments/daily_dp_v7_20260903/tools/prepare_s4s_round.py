from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4s';out.mkdir(exist_ok=False)
old=json.loads((E/'profiles/s4r/configs.json').read_text());cfg={}
for label in ('all_intraday','all_intraday_insert'):
    cfg[label]=dict(old[label],idle_task_handoff=False)
    cfg[label+'_handoff']=dict(cfg[label],idle_task_handoff=True)
p=out/'configs.json';p.write_text(json.dumps(cfg,indent=2))
b=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in b['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
extra={}
for rel in ('native/test_idle_handoff.cpp','reports/S4S_IDLE_TASK_HANDOFF_PRE_REGISTER_ZH.md','receipts/s4s_cooperation_live_v1/acceptance.json','receipts/s4s_official_cooperation_v2/acceptance.json'):
    src=E/rel;dst=out/'evidence'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);extra[rel]=hashlib.sha256(src.read_bytes()).hexdigest()
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=b,configs_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),evidence_hashes=extra,development=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2));print('FROZEN',flush=True)
