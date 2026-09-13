"""Focused engine/actor checks and full-game differential evidence."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from . import native
from .plans import simple_plan, repair
from .runtime import LocalGame, ENGINE, clean_obs, buffers, dump, sha, HERE


def equal(batch, game):
    for seat in (0,1):
        a,b=clean_obs(game.observation(seat)),batch.observation(0,seat)
        if a!=b:
            for k in a:
                if a[k]!=b[k]: raise AssertionError((game.t,seat,k,a[k],b[k]))


def focused():
    cases=[]
    specifications=[
      ('shared_seed',0,{'market':[['BUY_SEED','WHEAT',1],['HIRE']]},
       {'farmer':['PLANT','WHEAT'],'hands':[['PLANT','WHEAT']]}),
      ('cash_and_partial_trade',0,{'market':[['BUY_PRODUCT','WHEAT',100],['BUY_ANIMAL','COW',99],['HIRE']]},{'market':[['SELL','WHEAT',99],['BUY_LAND'],['BUY_PRODUCT','FERTILIZER',99]]}),
      ('capacity_and_overnight',22,{'market':[['BUY_PRODUCT','WHEAT',99],['BUY_ANIMAL','GOOSE',2]]}, {'farmer':['PICKUP','WHEAT',99],'market':[['BUY_PRODUCT','WHEAT',99]]}),
      ('final_action',717,{'market':[['BUY_PRODUCT','WHEAT',10]]},{'market':[['SELL','WHEAT',999],['BUY_ANIMAL','GOOSE',999]]}),
    ]
    for name,start,a,b in specifications:
        g=LocalGame(174043811,ENGINE);n=native.Batch([174043811],1)
        for t in range(719):
            pair=[copy.deepcopy(a if t==start else b if t==start+1 else {}),{}]
            if name=='cash_and_partial_trade': pair[1]={'market':[['BUY_PRODUCT','WHEAT',80],['SELL','WHEAT',80]]}
            g.advance(pair);n.primitive_step(0,pair);equal(n,g)
        cases.append({'name':name,'steps':g.t,'cash':[s.reward for s in g.state]})
    return cases


def direct_game(seed=174043812):
    g=LocalGame(seed,ENGINE);n=native.Batch([seed],1);n.action_modes([1,1]);b=buffers(1)
    rng=np.random.default_rng(201); checked=0;verbs=set()
    for t in range(719):
        n.begin(b['tiles'],b['glob'])
        actors=[native.Actor(clean_obs(g.observation(s))) for s in (0,1)]
        for s,a in enumerate(actors):
            x,y=a.encode();np.testing.assert_allclose(x,b['tiles'][s],atol=1e-6);np.testing.assert_allclose(y,b['glob'][s],atol=1e-6)
        for depth in range(29):
            count,rows=n.next(b['features'],b['owners'],b['bounds'],b['offsets'],b['ids'])
            if not count:break
            picks=[];qs=[]
            for i,seat in enumerate(b['ids'][:count]):
                lo,hi=b['offsets'][i:i+2];f,bounds=actors[seat].menu()
                np.testing.assert_allclose(f,b['features'][lo:hi],atol=1e-6)
                np.testing.assert_array_equal(bounds,b['bounds'][lo:hi]);checked+=1
                pick=int(rng.integers(hi-lo));q=int(rng.integers(1,bounds[pick]+1))
                verbs.add(int(f[pick,:27].argmax()));picks.append(pick);qs.append(q);actors[seat].apply(pick,q)
            n.apply(np.asarray(picks,np.int64),np.asarray(qs,np.int64))
        else:raise AssertionError('decoder depth')
        actions=[n.action(0,s) for s in (0,1)]
        assert actions==[a.action() for a in actors]
        n.step();g.advance(actions);equal(n,g)
    return {'steps':719,'menu_comparisons':checked,'verbs':sorted(verbs)}


def planner_games():
    out=[]
    for kind in (0,4,6):
        p=repair(simple_plan(kind));seed=174043820+kind
        g=LocalGame(seed,ENGINE);n=native.Batch([seed],1);lat=[];actions=[]
        for t in range(719):
            start=time.perf_counter()
            pair=[native.plan_action(clean_obs(g.observation(s)),p) for s in (0,1)]
            lat.append((time.perf_counter()-start)/2)
            actions.append(pair);g.advance(pair);n.primitive_step(0,pair);equal(n,g)
        fast=native.plan_games(p,[seed],[p])[0]
        assert fast[0]==g.state[0].reward and fast[1]==g.state[1].reward
        out.append({'kind':kind,'steps':g.t,'cash':fast[:2].tolist(),'latency_max':max(lat),'latency_p99':float(np.quantile(lat,.99))})
    return out


def boundary():
    g=LocalGame(174043830,ENGINE);obs=clean_obs(g.observation(0))
    a=native.Actor(obs);before=a.encode()
    hidden=copy.deepcopy(obs);hidden['seed']=123
    try:native.Actor(hidden)
    except RuntimeError:pass
    else:raise AssertionError('hidden seed accepted')
    # Changing the evaluator's rival private inventory must not alter actor features.
    n=native.Batch([174043830],1);both=[clean_obs(g.observation(s)) for s in (0,1)]
    both[1]['private']['shed']['MILK']=91;both[1]['private']['seeds']['MELON']=72
    n.restore(0,999123,both);n.action_modes([1,1]);b=buffers(1);n.begin(b['tiles'],b['glob'])
    np.testing.assert_array_equal(before[0],b['tiles'][0]);np.testing.assert_array_equal(before[1],b['glob'][0])
    return {'seed_rejected':True,'opponent_private_and_rng_invariance':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    start=time.perf_counter()
    report={'focused':focused(),'direct':direct_game(),'planner':planner_games(),'boundary':boundary()}
    report.update(passed=True,seconds=time.perf_counter()-start,
        sources={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.cpp','.hpp','.so')})
    dump(args.out,report);print(json.dumps(report))


if __name__=='__main__':main()
