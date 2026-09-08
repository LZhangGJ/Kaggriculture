import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=json.loads((root/'s3_shortlist_configs.json').read_text());out={}
for name in ('portfolio_182','portfolio_333','portfolio_255','old_opening_control'):
    for rotate in (False,True):out[name+('_rotate' if rotate else '_keep')]={**base[name],'rotate_finite':rotate}
path=root/'s3_rotation_configs.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
