from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1];out=EXP/'profiles/s4m1';out.mkdir(exist_ok=False)
prior=json.loads((EXP/'profiles/s4l/configs.json').read_text())
mapping={'all_intraday':'all_intraday_causal_pickup','auto_portfolio':'all_intraday_auto_portfolio_calendar_causal_pickup',
         'auto_portfolio_procure':'all_intraday_auto_portfolio_calendar_procure_causal_pickup'}
bases={'no_intraday':dict(prior['all_intraday_causal_pickup'],intraday_admission=False,intraday_procurement=False)}
bases.update({k:prior[v] for k,v in mapping.items()});cfg={}
for label,pars in bases.items():
    cfg[label]=dict(pars,shared_service_insertions=False)
    cfg[label+'_insert']=dict(pars,shared_service_insertions=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ('test_policy.cpp','test_service_recovery.cpp','test_incremental_pickup.cpp','test_shared_insertions.cpp'):
    shutil.copy2(EXP/'native'/name,out/'source'/name)
doc=EXP/'reports/S4M1_SHARED_INSERTION_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,prior_mapping=mapping,
    configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
