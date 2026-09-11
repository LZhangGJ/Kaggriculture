"""Pre-register complementary service decomposition and joint assignment."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
src=EXP/'profiles/s3r/configs.json';old=json.loads(src.read_text())['old']
out=EXP/'profiles/s3s';out.mkdir(exist_ok=False)
configs={
    'old':dict(old,split_service_jobs=False,regret_schedule=False),
    'split_only':dict(old,split_service_jobs=True,regret_schedule=False),
    'regret_only':dict(old,split_service_jobs=False,regret_schedule=True),
    'split_regret':dict(old,split_service_jobs=True,regret_schedule=True),
}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    A=[20261401,20261450],B=[20261501,20261550],conditional_new_I=[20262201,20262250],
    I_condition='After reviewing fixed A/B interaction and broad outcomes; no configuration refit.',
    final_holdout_used=False,fitted_parameters=0),indent=2))
snapshot=out/'source';snapshot.mkdir()
for rel in ('native/policy.hpp','native/module.cpp','native/test_policy.cpp'):
    (snapshot/Path(rel).name).write_bytes((EXP/rel).read_bytes())
print(json.dumps(dict(configs=str(out/'configs.json'),labels=list(configs))))
