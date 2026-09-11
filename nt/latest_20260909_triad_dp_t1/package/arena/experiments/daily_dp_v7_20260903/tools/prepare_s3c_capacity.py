"""One-factor/storage interactions following the preregistered land ablation."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'profiles/S3C03.json').read_text());configs={}
for land in (3,4):
    for label,change in {
        'base':{},
        'return':dict(fix_logistics=True),
        'idle':dict(early_deposit=2,sell_deposits=True),
        'return_idle':dict(fix_logistics=True,early_deposit=2,sell_deposits=True),
        'space70':dict(hold_capacity=70),
        'space40':dict(hold_capacity=40),
    }.items():configs[f'L{land}_{label}']=dict(base,max_land=land,**change)
out=EXP/'profiles/s3c/capacity_configs.json'
assert not out.exists();out.write_text(json.dumps(configs,indent=2),encoding='utf-8')
print(len(configs))
