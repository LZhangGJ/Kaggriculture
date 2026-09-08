"""Freeze S3K parameters before any match. No opponent-dependent tuning."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
source=EXP/'profiles/s3j/confirmation_configs.json'
base=json.loads(source.read_text())['log1_own0_new0']
out=EXP/'profiles/s3k';out.mkdir(exist_ok=False)
configs={f'replant_{w:g}':dict(base,existing_replant_weight=w) for w in (0,.5,1)}
(out/'configs.json').write_text(json.dumps(configs,indent=2))
(out/'freeze.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    configurations=configs,development_seeds={'A':[20261401,20261450],'B':[20261501,20261550],'C':[20261601,20261650]},
    same_config_all_opponents=True,final_holdout_used=False),indent=2))
print(json.dumps(configs,indent=2))
