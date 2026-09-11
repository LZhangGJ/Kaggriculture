"""Freeze PICKUP semantic repair independently on four existing backgrounds."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
out=EXP/'profiles/s4l';out.mkdir(exist_ok=False)
prior=json.loads((EXP/'profiles/s4k2/configs.json').read_text());cfg={}
for base in ('all_intraday','all_intraday_procure','all_intraday_auto_portfolio_calendar','all_intraday_auto_portfolio_calendar_procure'):
    cfg[base]=dict(prior[base],incremental_pickup_repair=False)
    cfg[base+'_causal_pickup']=dict(prior[base],incremental_pickup_repair=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ('test_policy.cpp','test_service_recovery.cpp','test_incremental_pickup.cpp'):
    shutil.copy2(EXP/'native'/name,out/'source'/name)
doc='S4L_INCREMENTAL_PICKUP_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,
    configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register=doc,
    pre_register_sha256=hashlib.sha256((EXP/'reports'/doc).read_bytes()).hexdigest(),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
