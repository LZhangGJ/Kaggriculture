import unittest
from unittest.mock import patch
from tests.arena import test_arena
from tools.arena import controller, daily_dispatch, schedule, store


class DailyDispatchTest(unittest.TestCase):
    def setUp(self):
        self.f = test_arena.ArenaTest(); self.f.setUp(); self.root = self.f.root
        self.f.roster(3)

    def tearDown(self):
        self.f.tearDown()

    def test_partition_preserves_games_pairs_and_resume(self):
        m = schedule.plan(self.root, 'daily-one', 'daily', 32)
        hosts = [('mini', dict(enabled=True, tournament_weight=1800)),
                 ('vast', dict(enabled=True, tournament_weight=7000))]
        daily_dispatch.prepare(self.root, hosts)
        owners = daily_dispatch.assignments(self.root, m)
        self.assertEqual(set(owners), {g['id'] for g in m['games']})
        groups = {}
        for g in m['games']:
            key = (g['seed'], tuple(sorted(g['agents'])))
            groups.setdefault(key, set()).add(owners[g['id']])
        self.assertTrue(all(len(x)==1 for x in groups.values()))
        self.assertGreater(list(owners.values()).count('vast'), len(owners)*.6)
        daily_dispatch.prepare(self.root, hosts)
        self.assertEqual(owners, daily_dispatch.assignments(self.root, m))
        mini = daily_dispatch.shard(self.root, m, 'mini')
        vast = daily_dispatch.shard(self.root, m, 'vast')
        self.assertFalse({g['id'] for g in mini['games']} & {g['id'] for g in vast['games']})
        self.assertEqual(mini['contract_hash'], m['contract_hash'])

    def test_remote_daily_preempts_and_then_resumes_continuous(self):
        daily = schedule.plan(self.root, 'daily-worker', 'daily', 1)
        daily['parent_daily'] = 'daily-parent'
        store.write(self.root/'runs'/daily['id']/'manifest.json', daily)
        schedule.plan(self.root, 'continuous-one', 'continuous', 1)
        cfg = store.read(self.root/'config.json'); cfg['max_games_per_tick']=2
        seen = []
        with patch.object(controller, 'run_one', side_effect=lambda root,m,g,cfg: seen.append(m['kind'])):
            controller.work(self.root,cfg,remote_only=True)
        self.assertEqual(seen, ['daily','daily'])
        self.f.fill(daily); seen.clear()
        with patch.object(controller, 'run_one', side_effect=lambda root,m,g,cfg: seen.append(m['kind'])):
            controller.work(self.root,cfg,remote_only=True)
        self.assertEqual(seen, ['continuous','continuous'])

    def test_coordinator_skips_remote_owned_games(self):
        m = schedule.plan(self.root, 'daily-one', 'daily', 2)
        daily_dispatch.prepare(self.root, [('vast',dict(enabled=True,tournament_weight=7000))])
        owners=daily_dispatch.assignments(self.root,m); seen=[]
        cfg=store.read(self.root/'config.json')
        with patch.object(controller,'run_one',side_effect=lambda root,m,g,cfg:seen.append(g['id'])):
            controller.work(self.root,cfg)
        self.assertTrue(seen)
        self.assertTrue(all(owners[g]=='coordinator' for g in seen))
