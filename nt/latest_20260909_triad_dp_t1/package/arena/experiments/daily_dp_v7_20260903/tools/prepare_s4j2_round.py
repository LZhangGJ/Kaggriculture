"""Freeze the twelve pre-registered live policies, before panel outcomes."""
from pathlib import Path
import argparse,hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--round',default='s4j2');a=p.parse_args()
assert a.round.replace('_','').isalnum()
out=EXP/'profiles'/a.round;out.mkdir(exist_ok=False)
prior=json.loads((EXP/'profiles/s4g1/configs.json').read_text());cfg={}
for context in ('intraday_funded','all_intraday'):
    base=dict(prior[context],joint_investment_portfolio=False,portfolio_crop_calendar=False)
    cfg[context]=base
    cfg[context+'_auto']=dict(base,autonomous_start=True)
    cfg[context+'_portfolio']=dict(base,joint_investment_portfolio=True)
    cfg[context+'_auto_portfolio']=dict(base,autonomous_start=True,joint_investment_portfolio=True)
    cfg[context+'_portfolio_calendar']=dict(base,joint_investment_portfolio=True,portfolio_crop_calendar=True)
    cfg[context+'_auto_portfolio_calendar']=dict(base,autonomous_start=True,joint_investment_portfolio=True,portfolio_crop_calendar=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ('test_policy.cpp','test_crop_delivery.cpp','test_portfolio_representation.cpp'):
    shutil.copy2(EXP/'native'/name,out/'source'/name)
doc='S4J2_UNIFIED_CROP_CALENDAR_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,
    configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register=doc,
    pre_register_sha256=hashlib.sha256((EXP/'reports'/doc).read_bytes()).hexdigest(),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
