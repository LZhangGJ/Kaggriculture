from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4q';out.mkdir(exist_ok=False)
prior=json.loads((E/'profiles/s4n1/configs.json').read_text());configs={}
for label in ('all_intraday','all_intraday_insert'):
    configs[label]=dict(prior[label],day_consequence_compare=False,day_value_public_supply=False)
    configs[label+'_value']=dict(configs[label],day_consequence_compare=True)
    configs[label+'_value_public']=dict(configs[label+'_value'],day_value_public_supply=True)
path=out/'configs.json';path.write_text(json.dumps(configs,indent=2))
build=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for n in ('test_day_consequence.cpp','test_observed_day_scenario.cpp'):shutil.copy2(E/'native'/n,out/'source'/n)
doc=E/'reports/S4Q_DAY_RESIDUAL_VALUE_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=build,configs_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(configs))),flush=True)
