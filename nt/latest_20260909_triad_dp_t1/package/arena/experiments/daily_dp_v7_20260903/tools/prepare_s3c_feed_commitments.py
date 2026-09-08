"""Independent existing-versus-new animal feed demand ablations."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'profiles/S3C03.json').read_text());configs={}
for logistics in (False,True):
    for own,new in ((0,0),(0,1),(.5,0),(.5,.5),(1,0),(1,1)):
        name=f'log{int(logistics)}_own{own}_new{new}'
        configs[name]=dict(base,fix_logistics=logistics,own_feed_demand_weight=own,committed_feed_weight=new)
out=EXP/'profiles/s3c/feed_commitment_configs.json'
assert not out.exists();out.write_text(json.dumps(configs,indent=2),encoding='utf-8')
(EXP/'profiles/s3c/feed_commitment_plan.json').write_text(json.dumps(dict(
    hypothesis='A new animal selected in the current planning pass generates feed obligations before appearing on the farm.',
    effect='Add those obligations to wheat demand used to rank subsequent projects; do not modify output totals or official rules.',
    control='Independent existing herd demand weights and logistics toggle; all default-zero branches retain old behavior.',
    evaluation='All 12 configs on A and B (each 50 seeds, both seats, PASS and full G001). Used only as development.',
    final_holdout='20261201..50 PASS and 20261301..50 G001 remain unused.'),indent=2),encoding='utf-8')
print(len(configs))
