"""Daily economic diagnostics for a development seed; no policy access to seed."""
from pathlib import Path
import argparse
import collections
import contextlib
import copy
import io
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification'))
from policy_host import Policy, LocalGame, load_engine


def farm_summary(obs):
    farm=obs['farms'][int(obs['player'])]
    crop=collections.Counter();animal=collections.Counter();yield_units=collections.Counter()
    for row in farm['tiles']:
        for t in row:
            if not isinstance(t,dict):continue
            if t.get('crop'):crop[t['crop']]+=1;yield_units[t['crop']]+=t.get('yield_units',0)
            if t.get('animal'):animal[t['animal']]+=1;yield_units[t['animal']]+=t.get('yield_units',0)
    return dict(cash=farm['money'],crops=dict(crop),animals=dict(animal),yield_units=dict(yield_units),
                hands=len(farm['hands']),land=farm['unlocked_quadrants'],shed=obs['private']['shed'],seeds=obs['private']['seeds'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--opponent',required=True)
    p.add_argument('--seed-index',type=int,default=0);p.add_argument('--seed',type=int,help='Explicit seed for post-evaluation diagnostics only')
    p.add_argument('--seat',type=int,default=0);p.add_argument('--out',required=True)
    a=p.parse_args();seeds=json.loads((HERE/'SEEDS.json').read_text())['development'];seed=a.seed if a.seed is not None else seeds[a.seed_index]
    pool=json.loads((HERE/'SOURCE_POOL.json').read_text())['opponents'];rival=next(r for r in pool if r['id']==a.opponent)
    daily=[];opening=[];tallies=[collections.Counter(),collections.Counter()]
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        own=Policy(str(HERE/'candidates'/a.candidate/'main.py'),'trace_own');enemy=Policy(rival['entry'],'trace_rival')
        env=LocalGame(seed,load_engine())
        while not env.done:
            observations=[env.observation(s) for s in (0,1)]
            if env.t%24==0:
                daily.append(dict(step=env.t,own=farm_summary(observations[a.seat]),rival=farm_summary(observations[1-a.seat]),market=copy.deepcopy(observations[0]['market'])))
            actions=[None,None]
            for seat,policy in ((a.seat,own),(1-a.seat,enemy)):
                actions[seat]=policy(observations[seat],copy.deepcopy(env.configuration))
                for order in actions[seat].get('market',[]):
                    tallies[seat][' '.join(str(x) for x in order[:2])]+=order[2] if len(order)>2 else 1
            if env.t<48:opening.append(dict(step=env.t,own=actions[a.seat],rival=actions[1-a.seat]))
            env.advance(actions)
        final=[f['money'] for f in env.state[0].observation.farms]
    result=dict(candidate=a.candidate,opponent=a.opponent,seed=seed,seat=a.seat,steps=env.t,own_cash=final[a.seat],rival_cash=final[1-a.seat],
                daily=daily,opening=opening,market_intents_own=dict(tallies[a.seat]),market_intents_rival=dict(tallies[1-a.seat]))
    out=HERE/'diagnostics'/a.out;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('candidate','opponent','seed','seat','steps','own_cash','rival_cash')}))


if __name__=='__main__':main()
