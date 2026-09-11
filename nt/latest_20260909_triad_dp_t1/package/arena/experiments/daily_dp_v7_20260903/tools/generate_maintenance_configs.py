from pathlib import Path
import itertools,json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'s3_deposit_configs.json').read_text());out={}
for label in ('portfolio_182_early2_sell1','portfolio_333_early2_sell1','portfolio_333_land0_r450_w4_c2_f0_control','portfolio_182_land1_r100_w4_c3_f0_control','portfolio_333_land1_r450_w3_c2_f0_control','old_opening_control_control'):
    for water,care in itertools.product((False,True),repeat=2):
        out[f'{label}_water{int(water)}_care{int(care)}']=dict(base[label],efficient_water=water,efficient_care=care)
(EXP/'s3_maintenance_configs.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
