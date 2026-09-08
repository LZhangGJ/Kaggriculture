from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4n1';out.mkdir(exist_ok=False)
prior=json.loads((E/'profiles/s4m1/configs.json').read_text());cfg={}
for label in ('all_intraday','all_intraday_insert'):
    cfg[label]=dict(prior[label],intraday_future_workforce=False)
    cfg[label+'_workforce']=dict(prior[label],intraday_future_workforce=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ('test_policy.cpp','test_intraday_workforce.cpp','test_shared_insertions.cpp'):
    shutil.copy2(E/'native'/name,out/'source'/name)
doc=E/'reports/S4N1_MARGINAL_WORKFORCE_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,
    configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),
    development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
