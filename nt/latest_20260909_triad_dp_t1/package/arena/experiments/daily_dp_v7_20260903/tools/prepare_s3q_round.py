"""Freeze a proposal/funding/schedule comparison, not an opponent-specific plan."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'profiles/s3p/configs.json';old=json.loads(src.read_text())['old']
out=EXP/'profiles/s3q';out.mkdir(exist_ok=False)
configs={'old':dict(old,funded_bundle_mode=0),'funded_repair':dict(old,funded_bundle_mode=1),
         'funded_all':dict(old,funded_bundle_mode=2)}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    A=[20261401,20261450],B=[20261501,20261550],conditional_new_G=[20262001,20262050],
    G_condition='Run only after broad development benefit; otherwise leave unused.',
    final_holdout_used=False,fitted_parameters=0),indent=2))
snapshot=out/'source';snapshot.mkdir()
for rel in ('native/policy.hpp','native/module.cpp','native/test_policy.cpp'):
    (snapshot/Path(rel).name).write_bytes((EXP/rel).read_bytes())
print(json.dumps(dict(configs=str(out/'configs.json'),labels=list(configs))))
