"""S3L one-mechanism configurations; frozen before strength testing."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1];source=EXP/'profiles/s3k/configs.json'
base=json.loads(source.read_text())['replant_0'];out=EXP/'profiles/s3l';out.mkdir(exist_ok=False)
configs={'hire_base':dict(base,insertion_hire_estimate=False),'hire_insertion':dict(base,insertion_hire_estimate=True)}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(configurations=configs,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    development_ranges=[[20261401,20261450],[20261501,20261550],[20261601,20261650]],
    independent_confirmation_if_promising=[20261701,20261750],final_holdout_used=False,
    no_opponent_identity_routing=True),indent=2))
print(json.dumps(configs))
