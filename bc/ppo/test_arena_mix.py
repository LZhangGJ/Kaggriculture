"""arena-mix-v2 checks: allocation, deterministic weighted sampling, weight rule."""
import json,os,random,tempfile,unittest
from pathlib import Path
from ppo.arena_opponents import assign,family_weights_from_margins,update_family_weights


def pool(families):
    return dict(opponents=[dict(id=f'{f}-{i}',family=f,archive='a',manifest={}) for f in families for i in range(2)],config={},arena_repo='.',runtime_files={})


class Checks(unittest.TestCase):
    def jobs(self,n=64):
        rows=[];idx=0
        for fam,count in [('selfplay',n*4//8),('champion_slot',n*1//8),('history_slot',n*3//8)]:
            for j in range(count):
                policies=['learner','learner']
                if fam!='selfplay':policies[j%2]=fam
                rows.append(dict(game=idx,seed=idx,policies=policies,family=fam,learner_seats=[s for s,p in enumerate(policies) if p=='learner']));idx+=1
        return rows

    def test_allocation_and_turns(self):
        jobs=self.jobs(512);fam={};seats=0
        for j in jobs:fam[j['family']]=fam.get(j['family'],0)+1;seats+=len(j['learner_seats'])
        self.assertEqual(fam,{'selfplay':256,'champion_slot':64,'history_slot':192});self.assertEqual(seats,768)

    def test_weighted_sampling_is_deterministic_and_biased(self):
        p=pool(['a','b','c','d']);jobs=self.jobs(512)
        u1=assign(jobs,p,1,7);u2=assign(jobs,p,1,7)
        self.assertEqual([j.get('arena_family') for j in u1],[j.get('arena_family') for j in u2])
        w={'a':3.,'b':.5,'c':.25,'d':.25}
        w1=assign(jobs,p,1,7,w);w2=assign(jobs,p,1,7,dict(w))
        self.assertEqual([j.get('arena_family') for j in w1],[j.get('arena_family') for j in w2])
        counts={}
        for j in w1:
            if j['family']=='arena':counts[j['arena_family']]=counts.get(j['arena_family'],0)+1
        self.assertEqual(sum(counts.values()),192);self.assertGreater(counts['a'],counts['b']*3)
        self.assertEqual(len([j for j in w1 if j['family']=='arena']),192)
        with self.assertRaises(ValueError):assign(jobs,p,1,7,{'a':0.})

    def test_weight_rule(self):
        w=family_weights_from_margins({'weak':5000.,'mid':-10000.,'strong':-30000.})
        self.assertAlmostEqual(sum(w.values())/3,1.,places=5)
        self.assertLess(w['weak'],w['mid']);self.assertLess(w['mid'],w['strong'])
        self.assertEqual(family_weights_from_margins({'x':100.,'y':50.}),{'x':1.,'y':1.})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'w.json'
            games=[dict(family='arena',arena_family='strong',learner_seats=[0],cash=[70000,100000],faults=[False,False]),dict(family='arena',arena_family='weak',learner_seats=[1],cash=[90000,95000],faults=[False,False]),dict(family='selfplay',learner_seats=[0,1],cash=[1,2],faults=[False,False])]
            r=update_family_weights(path,games,10);self.assertEqual(r['ema'],{'strong':-30000.,'weak':5000.});self.assertGreater(r['weights']['strong'],r['weights']['weak'])
            r2=update_family_weights(path,games,11);self.assertEqual(r2['ema']['strong'],-30000.)
            r3=update_family_weights(path,[dict(family='arena',arena_family='strong',learner_seats=[0],cash=[100000,90000],faults=[False,False])],12)
            self.assertAlmostEqual(r3['ema']['strong'],.9*-30000+.1*10000,places=3)


if __name__=='__main__':unittest.main(verbosity=2)
