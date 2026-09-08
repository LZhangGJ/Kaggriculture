"""Freeze non-trained, paired valuation modes before any outcome is inspected."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'profiles/s3n/configs.json'
base=json.loads(src.read_text())['expiry_plan']
out=EXP/'profiles/s3o';out.mkdir(exist_ok=False)
configs={key:dict(base,cashflow_value_mode=mode) for key,mode in
         [('old',0),('local_cash',1),('portfolio_cash',2)]}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,
    baseline_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    A=[20261401,20261450],B=[20261501,20261550],fresh_E=[20261801,20261850],
    final_holdout_used=False,opponent_identity_features=False),indent=2))
snapshot=out/'source';snapshot.mkdir()
for rel in ('native/policy.hpp','native/module.cpp','native/test_policy.cpp'):
    (snapshot/Path(rel).name).write_bytes((EXP/rel).read_bytes())
print(json.dumps(dict(configs=str(out/'configs.json'),labels=list(configs))))
