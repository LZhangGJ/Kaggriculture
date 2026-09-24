"""Official-rule one-turn cash/resource audit, not a full-game win claim."""
from pathlib import Path
import contextlib,copy,io,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification'))
from policy_host import Policy,LocalGame,load_engine


def main():
    pool=json.loads((HERE/'SOURCE_POOL.json').read_text())['opponents']
    seed=json.loads((HERE/'SEEDS.json').read_text())['development'][0]
    rows=[]
    for rival in pool:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            env=LocalGame(seed,load_engine())
            own=Policy(str(HERE/'candidates/cf_control/main.py'),'own')
            enemy=Policy(rival['entry'],'enemy')
            base=own(env.observation(0),copy.deepcopy(env.configuration))
            other=enemy(env.observation(1),copy.deepcopy(env.configuration))
            for quantity in (0,5,10,15,20,30,40):
                game=LocalGame(seed,load_engine());act=copy.deepcopy(base)
                if quantity:act['market']=[['BUY_PRODUCT','WHEAT',quantity],['SELL','WHEAT',quantity]]+act['market']
                game.advance([act,copy.deepcopy(other)])
                obs=game.observation(0)
                rows.append(dict(opponent=rival['id'],quantity=quantity,cash=obs['farms'][0]['money'],rival_cash=obs['farms'][1]['money'],
                    own_shed=obs['private']['shed'],own_seeds=obs['private']['seeds'],own_hands=len(obs['farms'][0]['hands'])))
    (HERE/'OPENING_AUDIT.json').write_text(json.dumps(dict(seed=seed,seat=0,scope='one official step; no terminal strength claim',rows=rows),indent=2)+'\n')
    for rival in pool:
        selected=[r for r in rows if r['opponent']==rival['id']];reference=selected[0]
        same=all(all(r[k]==reference[k] for k in ('own_shed','own_seeds','own_hands')) for r in selected)
        print(rival['id'],'resources_identical',same,[(r['quantity'],r['cash']-reference['cash'],r['rival_cash']-reference['rival_cash']) for r in selected])


if __name__=='__main__':main()
