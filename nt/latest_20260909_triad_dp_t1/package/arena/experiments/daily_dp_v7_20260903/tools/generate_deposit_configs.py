from pathlib import Path
import itertools,json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'s3_shortlist_configs.json').read_text());capacity=json.loads((EXP/'s3_capacity_configs.json').read_text())
bases={k:base[k] for k in ('portfolio_182','portfolio_333','old_opening_control')}
for k in ('portfolio_333_land0_r450_w4_c2_f0','portfolio_182_land1_r100_w4_c3_f0','portfolio_333_land1_r450_w3_c2_f0'):
    bases[k]=capacity[k]
out={}
for label,p in bases.items():
    out[label+'_control']=p
    for mode,sell in itertools.product((1,2),(False,True)):
        out[f'{label}_early{mode}_sell{int(sell)}']=dict(p,early_deposit=mode,sell_deposits=sell)
(EXP/'s3_deposit_configs.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
