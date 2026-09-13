"""Run the actual collector and optimizer before allocating training compute."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from . import native
from .policy import Policy,PolicyAgent,save,load
from .ppo import collect,replay,update
from .plans import simple_plan,repair,solver_repair
from .runtime import LocalGame,ENGINE,clean_obs,dump
from .preflight import equal


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(192);policy=Policy()
    r=collect(policy,[174043841,174043842,174043844,174043845],
              [repair(simple_plan(k)) for k in (0,4,6)])
    assert r['player_turns']==719*6 and r['simulated_player_turns']==719*8
    with torch.no_grad():
        h,v=policy.sequence(r['x']);lp,_=replay(policy,h,r['stages'])
        err=float((lp-r['logp'].flatten()).abs().max())
        assert err<.0001,err
    before={k:v.clone() for k,v in policy.state_dict().items()}
    result=update(policy,torch.optim.Adam(policy.parameters(),lr=.0003),r)
    changed=sum(not torch.equal(before[k],v) for k,v in policy.state_dict().items())
    assert changed>0
    # A complete actor game checks observation-only deployment and engine parity.
    agent=PolicyAgent(policy);g=LocalGame(174043843,ENGINE);n=native.Batch([174043843],1);lat=[]
    for t in range(719):
        start=time.perf_counter();action=agent(clean_obs(g.observation(0)));lat.append(time.perf_counter()-start)
        pair=[action,native.plan_action(clean_obs(g.observation(1)),repair(simple_plan(4)))]
        g.advance(pair);n.primitive_step(0,pair);equal(n,g)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'p.pt';save(policy,path,{});other,_=load(path)
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in policy.state_dict().items())
    p0=repair(simple_plan());start=time.perf_counter();sg=native.plan_games(p0,[174043850+i for i in range(4)],[repair(simple_plan(4)),repair(simple_plan(6))])
    seconds=time.perf_counter()-start;assert len(sg)==16 and (sg[:,2]==719).all()
    assert solver_repair(p0)==p0
    out={'passed':True,'collector_games':r['games'],'player_turns':r['player_turns'],'collector_seconds':r['seconds'],
         'player_turns_per_second':r['player_turns']/r['seconds'],'optimizer':result,'changed_tensors':changed,
         'replay_logp_max_error':err,'full_game_deployed_actor_parity':True,'latency_p99':float(np.quantile(lat,.99)),
         'latency_max':max(lat),'search_games_per_second':16/seconds,'search_game_seconds':seconds/16,
         'deployment_cash':[s.reward for s in g.state],'parameters':sum(p.numel() for p in policy.parameters())}
    dump(a.out,out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()
