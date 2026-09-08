from pathlib import Path
import itertools,json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'s3_shortlist_configs.json').read_text());out={}
for opening in ('portfolio_182','portfolio_333','old_opening_control'):
    out[opening+'_control']=base[opening]
    for land,reserve,wheat,carrot,fert in itertools.product((False,True),(100.,450.),(2,3,4),(2,3),(False,True)):
        p=dict(base[opening],economic_land=land,operating_reserve=reserve,crop_harvest_age=[wheat,carrot,8,10,10],finite_fertilizer=fert,fix_finite_projection=True)
        label=f'{opening}_land{int(land)}_r{int(reserve)}_w{wheat}_c{carrot}_f{int(fert)}'
        out[label]=p
(EXP/'s3_capacity_configs.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
