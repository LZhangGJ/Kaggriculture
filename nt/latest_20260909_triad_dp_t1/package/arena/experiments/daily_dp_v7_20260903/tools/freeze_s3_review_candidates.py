"""Preregister a small Pareto shortlist before expanding development seeds."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
configs=json.loads((EXP/'s3_maintenance_configs.json').read_text());out={}
labels=['portfolio_333_land1_r450_w3_c2_f0_control_water1_care1',
        'portfolio_182_early2_sell1_water0_care1',
        'portfolio_182_land1_r100_w4_c3_f0_control_water1_care1',
        'portfolio_333_land0_r450_w4_c2_f0_control_water1_care1',
        'portfolio_333_early2_sell1_water0_care0',
        'old_opening_control_control_water0_care0']
for i,label in enumerate(labels):out[f'S3C{i+1:02d}']=configs[label]
out['S1_native_control']={}
(EXP/'s3_review_candidates.json').write_text(json.dumps(out,indent=2),encoding='utf8')
(EXP/'s3_review_provenance.json').write_text(json.dumps(dict(labels=labels,new_development_seeds=[20261401,20261450],never_used_before_shortlist=True,formal_holdouts_untouched=True),indent=2),encoding='utf8')
external={}
for label in ('S3C01','S3C02','S3C03','S3C04'):
    for weight in (0.,.1,.25,.5,1.,2.):
        external[f'{label}_external{weight}']=dict(out[label],competition_objective_weight=weight)
(EXP/'s3_externality_configs.json').write_text(json.dumps(external,indent=2),encoding='utf8')
print(json.dumps(dict(review=len(out),externality=len(external))))
