"""Bounded, source-independent calibration of project values, not route tapes."""
from pathlib import Path
import itertools,json
root=Path(__file__).resolve().parents[1]
base={'opponent_supply_weight':.75,'competitive_sell_weight':.5,'sell_horizon_days':1}
configs={'control':base}
for sb,melon,animals in itertools.product((1.,1.5,2.,3.),(.25,.5,1.),(.5,1.)):
    configs[f's{sb}_m{melon}_a{animals}']={**base,'project_bias':{'STRAWBERRY':sb,'MELON':melon,'COW':animals,'SHEEP':animals,'GOOSE':animals*.9}}
for cover,cap,shadow in itertools.product((1,3,7),(.65,.85,.94),(2.,8.)):
    configs[f'feed{cover}_capital{cap}_shadow{shadow}']={**base,'feed_cover_days':cover,'capital_fraction':cap,'action_shadow':shadow}
path=root/'s3_economic_configs.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(configs,indent=2),encoding='utf8')
print(len(configs),'universal configs; original opening unchanged')
