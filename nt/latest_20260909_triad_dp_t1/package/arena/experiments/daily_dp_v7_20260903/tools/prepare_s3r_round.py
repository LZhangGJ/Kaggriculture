"""Freeze a 2x2 ablation before observing the procurement-overlap results."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'profiles/s3q/configs.json';previous=json.loads(src.read_text())
out=EXP/'profiles/s3r';out.mkdir(exist_ok=False)
configs={
    'old':dict(previous['old'],preparation_work=False),
    'prep_only':dict(previous['old'],preparation_work=True),
    'bundle_only':dict(previous['funded_all'],preparation_work=False),
    'bundle_prep':dict(previous['funded_all'],preparation_work=True),
}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    A=[20261401,20261450],B=[20261501,20261550],conditional_new_H=[20262101,20262150],
    H_condition='Run unchanged configurations only after reviewing A/B evidence; no final-holdout use.',
    final_holdout_used=False,fitted_parameters=0),indent=2))
snapshot=out/'source';snapshot.mkdir()
for rel in ('native/policy.hpp','native/module.cpp','native/test_policy.cpp'):
    (snapshot/Path(rel).name).write_bytes((EXP/rel).read_bytes())
print(json.dumps(dict(configs=str(out/'configs.json'),labels=list(configs))))
