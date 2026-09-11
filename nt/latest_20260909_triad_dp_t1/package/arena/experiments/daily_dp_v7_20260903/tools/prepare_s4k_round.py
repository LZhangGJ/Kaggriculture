"""Freeze eight service-recovery policies before any strength result."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--round',default='s4k');a=p.parse_args();r=a.round;assert r in ('s4k','s4k2')
EXP=Path(__file__).resolve().parents[1];out=EXP/'profiles'/r;out.mkdir(exist_ok=False)
prior=json.loads((EXP/'profiles/s4j2/configs.json').read_text());cfg={}
for name in ('all_intraday','all_intraday_auto_portfolio_calendar'):
    base=dict(prior[name],recover_service_inputs=False,procure_service_inputs=False,finance_service_inputs=False)
    cfg[name]=base
    cfg[name+'_recover']=dict(base,recover_service_inputs=True)
    cfg[name+'_procure']=dict(base,recover_service_inputs=True,procure_service_inputs=True)
    cfg[name+'_finance']=dict(base,recover_service_inputs=True,procure_service_inputs=True,finance_service_inputs=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ('test_policy.cpp','test_service_recovery.cpp'):shutil.copy2(EXP/'native'/name,out/'source'/name)
doc='S4K_SERVICE_REENTRY_PRE_REGISTER_ZH.md'
if r=='s4k2':doc='S4K2_TRANSACTION_PROTECTION_FIX_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,
    configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register=doc,
    pre_register_sha256=hashlib.sha256((EXP/'reports'/doc).read_bytes()).hexdigest(),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
