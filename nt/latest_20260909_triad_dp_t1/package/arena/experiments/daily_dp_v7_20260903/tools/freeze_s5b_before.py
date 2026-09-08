from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s5b';out.mkdir(exist_ok=False)
receipt=json.loads((E/'native/build/build_receipt.json').read_text())
for rel,h in receipt['source_hashes'].items():
 p=E/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 target=out/'before'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for rel in ['native/rotation_portfolio_prototype.hpp','native/test_rotation_portfolio.cpp','tools/build_native.py']:
 target=out/'before'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(E/rel,target)
(out/'before_build.json').write_text(json.dumps(receipt,indent=2));print('FROZEN_BEFORE_S5B')
