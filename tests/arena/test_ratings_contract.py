import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from tools.arena.ratings import fit, tally
from tests.arena import test_arena
from tools.arena import cumulative, schedule, store


class RatingMathTest(unittest.TestCase):
    def rows(self, outcomes):
        return [dict(agents=['a', 'b'], outcome=o, seed=i, cash=[0, 0]) for i, o in enumerate(outcomes)]

    def test_draw_is_half_win_and_half_loss(self):
        # Equal sample sizes: replacing two draws by a win and loss has identical likelihood.
        a, _ = fit(self.rows(['win0'] * 8 + ['draw'] * 2), ['a', 'b'])
        b, _ = fit(self.rows(['win0'] * 9 + ['win1']), ['a', 'b'])
        self.assertAlmostEqual(a['a'], b['a'], places=8)

    def test_draws_pull_unequal_strengths_together(self):
        a, _ = fit(self.rows(['win0'] * 10), ['a', 'b'])
        b, _ = fit(self.rows(['win0'] * 10 + ['draw'] * 20), ['a', 'b'])
        self.assertLess(abs(b['a'] - b['b']), abs(a['a'] - a['b']))

    def test_labels_and_seats_do_not_change_strength(self):
        rows = self.rows(['win0'] * 8 + ['win1'] * 2 + ['draw'])
        a, _ = fit(rows, ['a', 'b'])
        b, _ = fit(rows, ['b', 'a'])
        self.assertAlmostEqual(a['a'], b['a'], places=8)
        self.assertAlmostEqual(sum(a.values()), 0)

    def test_all_draws_have_zero_strength_and_half_score(self):
        rows = self.rows(['draw'] * 20)
        a, _ = fit(rows, ['a', 'b'])
        self.assertEqual(a, dict(a=0., b=0.))
        self.assertEqual(tally(rows, 'a')['score'], .5)
        self.assertEqual(tally(rows, 'a')['win_rate'], 0)

    def test_line_search_failure_requires_stationarity(self):
        result = SimpleNamespace(success=False, x=np.zeros(2), message='precision')
        with patch('tools.arena.ratings.minimize', return_value=result):
            self.assertEqual(fit(self.rows(['draw']), ['a', 'b'])[0], dict(a=0., b=0.))
            with self.assertRaises(RuntimeError):
                fit(self.rows(['win0']), ['a', 'b'])


class CumulativeTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_arena.ArenaTest()
        self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def test_active_history_retirement_and_idempotence(self):
        a, b, c = self.fixture.roster(3)
        daily = schedule.plan(self.root, 'daily-history', 'daily', 1)
        self.fixture.fill(daily)
        cont = schedule.plan(self.root, 'continuous-history', 'continuous', 1)
        self.fixture.fill(cont)
        first = cumulative.reports(self.root)
        self.assertEqual(first[0]['completed'], 12)
        self.assertEqual(first, cumulative.reports(self.root))
        store.write(self.root / 'roster.json', [dict(agent=a), dict(agent=b)])
        current = cumulative.reports(self.root)[0]
        self.assertEqual(current['completed'], 4)
        self.assertNotIn(c, current['ratings'])
