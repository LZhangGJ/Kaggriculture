"""Generic opening portfolio search. No opponent source/replay/date information."""
from pathlib import Path
import itertools,json,random
root=Path(__file__).resolve().parents[1]
base={'opponent_supply_weight':.75,'competitive_sell_weight':.5,'sell_horizon_days':1}
candidates=[]
for cow,sheep,goose in itertools.product(range(5),range(6),range(3)):
    animals=cow+sheep+goose
    if not 2<=animals<=6:continue
    for wheat,carrot,tomato,strawberry,melon in itertools.product((0,2,4),(0,2),(0,2),(0,2,4,6,8),(0,2,4,6,8)):
        total=cow*400+sheep*500+goose*300+wheat*10+carrot*20+tomato*50+strawberry*100+melon*80
        # Reserve one initial feed per animal and at least modest cash/worker buffer.
        if total+animals*25>2850 or total<2050:continue
        if animals+wheat+carrot+tomato+strawberry+melon>25:continue
        if strawberry+tomato+melon+wheat+carrot==0:continue
        candidates.append(dict(opening_animals=['COW']*cow+['SHEEP']*sheep+['GOOSE']*goose,opening_crops=[wheat,carrot,tomato,strawberry,melon]))
random.Random(37051).shuffle(candidates)
configs={'control':base}
for i,c in enumerate(candidates[:384]):configs[f'portfolio_{i:03}']={**base,**c}
path=root/'s3_opening_configs.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(configs,indent=2),encoding='utf8')
print(dict(feasible_grid=len(candidates),sampled=len(configs)-1,search_rng_seed=37051))
