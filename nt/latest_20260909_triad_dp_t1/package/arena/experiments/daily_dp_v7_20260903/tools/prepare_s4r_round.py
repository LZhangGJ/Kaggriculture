from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4r';out.mkdir(exist_ok=False)
old=json.loads((E/'profiles/s4q/configs.json').read_text());configs={}
for base in ('all_intraday','all_intraday_insert'):
    configs[base]=dict(old[base],day_value_replan_next_day=False)
    configs[base+'_value']=dict(old[base+'_value'],day_value_replan_next_day=False)
    configs[base+'_value_next']=dict(configs[base+'_value'],day_value_replan_next_day=True)
    configs[base+'_value_next_public']=dict(configs[base+'_value_next'],day_value_public_supply=True)
p=out/'configs.json';p.write_text(json.dumps(configs,indent=2))
b=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in b['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for n in ('test_next_day_value.cpp','test_cross_day_gap.cpp'):shutil.copy2(E/'native'/n,out/'source'/n)
doc=E/'reports/S4R_CROSS_DAY_CONTINUATION_PRE_REGISTER_ZH.md'
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',build=b,configs_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pre_register_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),development_N=[20262701,20262750],holdout_used=False,oracle_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(configs))),flush=True)
