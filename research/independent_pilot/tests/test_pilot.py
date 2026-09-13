import copy
import unittest
import numpy as np
from research.independent_pilot import native
from research.independent_pilot.plans import repair,solver_repair,simple_plan,random_plan
from research.independent_pilot.runtime import LocalGame,ENGINE,clean_obs


class PilotTests(unittest.TestCase):
    def test_solver_capacity_and_maturation(self):
        rng=np.random.default_rng(123)
        for _ in range(4):
            p=random_plan(rng);p[4][:8]=[70]*8
            fixed=solver_repair(p)
            for stage,row in enumerate(fixed):
                self.assertLessEqual(sum(row[:8]),25*row[9])
                for i,delay in enumerate([2,2,8,10,10,4,8,6]):
                    if stage*6+delay>=30:self.assertEqual(row[i],0)

    def test_all_official_commands_and_stop_are_representable(self):
        obs=clean_obs(LocalGame(174043861,ENGINE).observation(0));f=obs['farms'][0]
        f['money']=100000;obs['private']['seeds']={k:5 for k in obs['private']['seeds']}
        obs['private']['shed']={k:2 for k in obs['private']['shed']}
        obs['private']['inventories'][0]={k:2 for k in obs['private']['shed']}
        fixtures=[None,{'kind':'WEED'},{'kind':'COOP'}, {'kind':'PASTURE'},
            {'kind':'PLANT','crop':'WHEAT','planted_day':0,'yield_units':2,'watered_today':False},
            {'kind':'COOP','animal':'GOOSE','placed_day':0,'yield_units':2,'fed_today':False,'fertilizer_available':True}]
        verbs=set()
        for tile in fixtures:
            x=copy.deepcopy(obs);x['step']=120;x['day']=5;x['farms'][0]['tiles'][4][4]=tile
            a=native.Actor(x);features,b=a.menu();verbs.update(features[:,:27].argmax(-1).tolist())
            a.apply(0,1);features,b=a.menu();verbs.update(features[:,:27].argmax(-1).tolist())
        self.assertEqual(verbs,{0,*range(3,27)})

    def test_shared_seed_cash_and_hire_cap(self):
        obs=clean_obs(LocalGame(174043862,ENGINE).observation(0))
        obs['farms'][0]['hands']=[[4,4]];obs['private']['inventories'].append({});obs['private']['seeds']['WHEAT']=1
        a=native.Actor(obs);f,b=a.menu();ix=int(np.flatnonzero(f[:,7]>0)[0]);a.apply(ix,1)
        f,b=a.menu();self.assertFalse((f[:,7]>0).any());a.apply(0,1)
        f,b=a.menu();ix=int(np.flatnonzero((f[:,23]>0)&(f[:,27]>0))[0]);a.apply(ix,int(b[ix]))
        f,b=a.menu();self.assertFalse((f[:,21]>0).any())
        rich=copy.deepcopy(obs);rich['farms'][0]['hands']=[[4,4]]*16;rich['private']['inventories']=[{}]*17
        a=native.Actor(rich)
        for _ in range(17):a.menu();a.apply(0,1)
        f,b=a.menu();self.assertFalse((f[:,21]>0).any())

    def test_actor_rejects_hidden_inputs(self):
        obs=clean_obs(LocalGame(174043863,ENGINE).observation(0))
        for key in ('seed','rng','opponent_id'):
            x=copy.deepcopy(obs);x[key]=3
            with self.assertRaises(RuntimeError):native.Actor(x)

    def test_plan_zero_cash_does_not_hang(self):
        obs=clean_obs(LocalGame(174043864,ENGINE).observation(0));obs['farms'][0]['money']=0
        a=native.plan_action(obs,repair(simple_plan()))
        self.assertEqual(a['market'],[])


if __name__=='__main__':unittest.main()
