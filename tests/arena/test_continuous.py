import tempfile
import unittest
import zipfile
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tools.arena import store,intake,schedule,elo,sandbox


class ContinuousTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);store.init(self.root)
        self.ids=[]
        for name in ('a','b','c'):
            p=self.root/(name+'.zip')
            with zipfile.ZipFile(p,'w') as z:z.writestr('main.py',name)
            aid=intake.submit(self.root,p,dict(name=name,author='test',version='1',run=['python','main.py']))['agent']
            a=store.read(self.root/'agents'/f'{aid}.json');a['build_verified']=True
            store.write(self.root/'agents'/f'{aid}.json',a);self.ids.append(aid)
        intake.set_roster(self.root,[dict(agent=a,category='candidate',reason='test') for a in self.ids])

    def tearDown(self):self.temp.cleanup()

    def fill(self,m,draw=False):
        for g in m['games']:
            win=g['agents'].index(min(g['agents']))
            schedule.save_result(self.root,m,g,dict(game=g['id'],agents=g['agents'],resolved=True,terminal=True,
                reason='terminal',outcome='draw' if draw else f'win{win}',cash=[1,1] if draw else [1,0] if win==0 else [0,1]))

    def test_rounds_use_distinct_seeds_and_all_seats(self):
        first=schedule.plan(self.root,'continuous-1','continuous',2)
        second=schedule.plan(self.root,'continuous-2','continuous',2)
        self.assertEqual(len(first['games']),12)
        self.assertFalse({g['seed'] for g in first['games']}&{g['seed'] for g in second['games']})

    def test_elo_is_idempotent_and_ignores_daily(self):
        daily=schedule.plan(self.root,'daily-test','daily',1);self.fill(daily)
        self.assertEqual(elo.update(self.root)['ratings'],[])
        m=schedule.plan(self.root,'continuous-1','continuous',2);self.fill(m)
        a=elo.update(self.root)['ratings'];b=elo.update(self.root)['ratings']
        self.assertEqual(a,b);self.assertEqual(sum(r['games'] for r in a),24)
        self.assertAlmostEqual(sum(r['elo'] for r in a),4500)

    def test_draws_preserve_equal_ratings(self):
        m=schedule.plan(self.root,'continuous-1','continuous',2);self.fill(m,draw=True)
        self.assertTrue(all(r['elo']==1500 for r in elo.update(self.root)['ratings']))

    def test_transported_image_requires_matching_content(self):
        info={'RootFS':{'Layers':['layer-a']},'Architecture':'amd64','Os':'linux','Config':{'User':'65534'}}
        cfg={'image':'sha256:original','image_runtime':'sha256:transport','image_fingerprint':sandbox.image_fingerprint(info)}
        with patch.object(sandbox.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([info]))):
            sandbox.preflight(cfg)
        changed={**info,'RootFS':{'Layers':['different-layer']}}
        with patch.object(sandbox.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([changed]))):
            with self.assertRaises(RuntimeError):sandbox.preflight(cfg)

    def test_workers_do_not_claim_each_others_rounds(self):
        from tools.arena.continuous import owns_round
        self.assertTrue(owns_round({}, 'mini'))
        self.assertFalse(owns_round({}, 'vast'))
        self.assertTrue(owns_round({'executor':'vast'}, 'vast'))
        self.assertFalse(owns_round({'executor':'vast'}, 'mini'))
