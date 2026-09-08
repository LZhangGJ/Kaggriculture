"""Economic forecast ablation. No opponent identity or replay in parameters."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'s3_shortlist_configs.json').read_text())
out={}
for opening in ('portfolio_182','portfolio_333','old_opening_control'):
    for feed in (0.,.5,1.):
        for temporal in (0.,.25,.5,1.):
            p=dict(base[opening],own_feed_demand_weight=feed,temporal_value_weight=temporal)
            out[f'{opening}_feed{feed}_time{temporal}']=p
(EXP/'s3_temporal_configs.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
