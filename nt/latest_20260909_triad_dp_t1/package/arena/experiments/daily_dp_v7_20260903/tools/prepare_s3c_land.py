"""Freeze the paired three/four quadrant ablation; original best is untouched."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
out=EXP/'profiles/s3c'
out.mkdir(exist_ok=False)
base=json.loads((EXP/'profiles/S3C03.json').read_text())
configs={'S3C03':base,'allow_four':dict(base,max_land=4)}
(out/'land_configs.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
(out/'allow_four.json').write_text(json.dumps(configs['allow_four'],indent=2),encoding='utf-8')
(out/'land_plan.json').write_text(json.dumps(dict(
    objective='Independent fourth-quadrant capacity ablation, not forced expansion.',
    development_A=dict(seed_start=20261401,count=50,seats=[0,1]),
    development_B=dict(seed_start=20261501,count=50,seats=[0,1]),
    opponents=['PASS','original_full_realtime_G001'],
    promotion='Both PASS cash and G001 wins considered. Both sets are development; no formal holdout used.',
    legacy_land_mode='economic_land=false retains old date gates; four-quadrant experiment uses economic_land=true.',
    config_labels=list(configs)),indent=2),encoding='utf-8')
print(out)
