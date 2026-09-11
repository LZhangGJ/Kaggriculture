"""Freeze payoff-margin comparison without fitting any new weight or opening."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'profiles/s3o/configs.json';previous=json.loads(src.read_text())
out=EXP/'profiles/s3p';out.mkdir(exist_ok=False)
configs={k:previous[k] for k in ('old','portfolio_cash')}
configs['margin_cash']=dict(configs['portfolio_cash'],cashflow_value_mode=3)
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,
    source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    A=[20261401,20261450],B=[20261501,20261550],fresh_F=[20261901,20261950],
    final_holdout_used=False,new_fitted_parameters=0),indent=2))
snapshot=out/'source';snapshot.mkdir()
for rel in ('native/policy.hpp','native/module.cpp','native/test_policy.cpp'):
    (snapshot/Path(rel).name).write_bytes((EXP/rel).read_bytes())
print(json.dumps(dict(configs=str(out/'configs.json'),labels=list(configs))))
