"""Test the feed-cash hypothesis using existing generic policy controls."""
from pathlib import Path
import copy,json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'profiles/S3C03.json').read_text())
configs={'S3C03':base}
for cover in (1,3,7):
    for capacity in (30,60,90):
        for demand in (0.,.5,1.):
            cfg=copy.deepcopy(base)
            cfg.update(feed_cover_days=cover,feed_stock_cap=capacity,own_feed_demand_weight=demand)
            configs[f'cover{cover}_cap{capacity}_demand{demand:g}']=cfg
dest=EXP/'s3b_feed_configs.json';assert not dest.exists()
dest.write_text(json.dumps(configs,indent=2),encoding='utf-8')
print(len(configs))
