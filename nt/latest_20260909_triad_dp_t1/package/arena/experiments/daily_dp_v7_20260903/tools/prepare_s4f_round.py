"""Freeze one declared-value ablation in three existing deployment contexts."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
out=EXP/'profiles/s4f1';out.mkdir(exist_ok=False)
prior=json.loads((EXP/'profiles/s4e1/configs.json').read_text())
cfg={}
for label in ('intraday_stock','intraday_funded','all_intraday'):
    cfg[label]=dict(prior[label],intraday_declared_value=False)
    cfg[label+'_declared']=dict(prior[label],intraday_declared_value=True)
path=out/'configs.json';path.write_text(json.dumps(cfg,indent=2))
build=json.loads((EXP/'native/build/build_receipt.json').read_text());hashes={}
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);hashes[rel]=h
shutil.copy2(EXP/'native/test_policy.cpp',out/'source/test_policy.cpp')
(out/'freeze.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_STRENGTH',source_hashes=hashes,
    config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),binary_sha256=build['binary_sha256'],
    development_N=[20262701,20262750],confirmation_O=[20262801,20262850],unseen_P=[20262901,20262950],
    oracle_used=False,final_holdout_used=False),indent=2))
print(json.dumps(dict(status='FROZEN',configs=list(cfg))))
