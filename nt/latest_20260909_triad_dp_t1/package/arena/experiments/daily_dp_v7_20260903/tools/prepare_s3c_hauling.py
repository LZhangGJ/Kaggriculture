"""Freeze staffing/hauling ablation, including the existing-return control."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'profiles/S3C03.json').read_text());configs={}
for land in (3,4):
    for mode in ('base','return','staffed_hauling'):
        cfg=dict(base,max_land=land)
        if mode!='base':cfg['fix_logistics']=True
        if mode=='staffed_hauling':cfg['capacity_hauling']=True
        configs[f'L{land}_{mode}']=cfg
out=EXP/'profiles/s3c/hauling_configs.json'
assert not out.exists();out.write_text(json.dumps(configs,indent=2),encoding='utf-8')
(EXP/'profiles/s3c/hauling_plan.json').write_text(json.dumps(dict(
    hypothesis='On days with predicted output over shed capacity, include pre-EOD hauling in staffing and scheduling instead of using only leftover time.',
    no_future_access='Only own current jobs and their intended output. No environment seed, future RNG or opponent private state.',
    constraints='Retain baseline and all failures; extra wages and displaced tasks may outweigh recovered products.',
    sets=['20261401..20261450 both seats (A development)','20261501..20261550 both seats (B development)'],
    staged='A first, B all 6 to avoid selective reporting; no formal holdout'),indent=2),encoding='utf-8')
print(len(configs))
