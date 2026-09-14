import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.arena import intake, schedule, store, reporting, gates
from tools.arena.check_export import check
from tools.arena.check_issue import parse
from tools.arena.ratings import summary


class ArenaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        store.init(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, name):
        path = self.root / (name + ".zip")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("main.py", "# fixture " + name)
        record = intake.submit(self.root, path, {"name": name, "author": name, "version": "1", "run": ["python", "main.py"]})
        aid = record["agent"]
        a = store.read(self.root / "agents" / (aid + ".json"))
        a.update(build_verified=True, image=None)
        store.write(self.root / "agents" / (aid + ".json"), a)
        return aid

    def roster(self, n=3):
        ids = [self.agent(str(i)) for i in range(n)]
        intake.set_roster(self.root, [{"agent": a, "category": "public", "reason": "fixture"} for a in ids])
        return ids

    def fill(self, m):
        for g in m["games"]:
            schedule.save_result(self.root, m, g, {"game": g["id"], "agents": g["agents"], "resolved": True,
                "reason": "terminal", "terminal": True, "outcome": "draw", "cash": [3000, 3000]})

    def test_zip_traversal(self):
        p = self.root / "bad.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("../outside", "x")
        with self.assertRaises(ValueError):
            intake.inspect_zip(p)

    def test_zip_symlink(self):
        p = self.root / "bad.zip"
        i = zipfile.ZipInfo("link")
        i.create_system = 3
        i.external_attr = (0o120777 << 16)
        with zipfile.ZipFile(p, "w") as z:
            z.writestr(i, "/etc/passwd")
        with self.assertRaises(ValueError):
            intake.inspect_zip(p)

    def test_manifest_no_shell(self):
        with self.assertRaises(ValueError):
            intake.manifest_check({"name": "a", "author": "b", "version": "1", "run": "echo bad"})

    def test_duplicate_submission(self):
        aid = self.agent("a")
        a = store.read(self.root / "agents" / (aid + ".json"))
        record = intake.submit(self.root, self.root / "a.zip", a["manifest"])
        self.assertEqual(aid, record["agent"])
        self.assertEqual(1, len(store.records(self.root, "agents")))

    def test_hash_rejection(self):
        self.agent("a")
        with self.assertRaises(ValueError):
            intake.submit(self.root, self.root / "a.zip", {"name":"a","author":"a","version":"2","run":["python"],"sha256":"bad"})

    def test_round_robin_seats(self):
        ids = self.roster()
        m = schedule.plan(self.root, "daily-one", count=4)
        self.assertEqual(24, len(m["games"]))
        self.assertEqual(24, len({g["id"] for g in m["games"]}))
        self.assertEqual(4, m["seed_count"])
        for a in ids:
            for seat in (0, 1):
                self.assertEqual(8, sum(g["agents"][seat] == a for g in m["games"]))

    def test_plan_resume_no_new_seeds(self):
        self.roster()
        m = schedule.plan(self.root, "one", count=4)
        self.assertEqual(m, schedule.plan(self.root, "one", count=4))
        self.assertEqual(4, len(store.read(self.root / "private/seeds.json")))

    def test_no_daily_overlap(self):
        self.roster()
        m = schedule.plan(self.root, "one", count=2)
        with self.assertRaises(ValueError):
            schedule.plan(self.root, "two", count=2)
        self.fill(m)
        schedule.plan(self.root, "two", count=2)

    def test_no_seed_reuse(self):
        schedule.reserve(self.root, "a", 2, [1, 2])
        with self.assertRaises(ValueError):
            schedule.reserve(self.root, "b", 2, [2, 3])

    def test_result_identity_and_terminal(self):
        self.roster(2)
        m = schedule.plan(self.root, "one", count=1)
        g = m["games"][0]
        with self.assertRaises(ValueError):
            schedule.save_result(self.root, m, g, {"game": "other"})
        with self.assertRaises(ValueError):
            schedule.save_result(self.root, m, g, {"game":g["id"],"agents":g["agents"],"resolved":True,"reason":"terminal","outcome":"draw"})

    def test_report_incomplete_and_no_private_fields(self):
        self.roster(2)
        schedule.plan(self.root, "one", count=2)
        data = reporting.build(self.root, bootstrap=0)
        self.assertFalse(data["runs"][0]["complete"])
        check(self.root / "site")
        self.assertNotIn('"seed":', (self.root / "site/data.json").read_text())

    def test_draws_are_not_wins(self):
        ids = self.roster(2)
        m = schedule.plan(self.root, "one", count=2)
        self.fill(m)
        data = reporting.build(self.root, bootstrap=0)
        st = data["runs"][0]["stats"][ids[0]]
        self.assertEqual(st["win_rate"], 0)
        self.assertEqual(st["score"], .5)

    def test_placement_outside_cap(self):
        refs = self.roster(6)
        newcomer = self.agent("new")
        m = schedule.plan(self.root, "placement-new", "placement", candidate=newcomer, references=refs, count=32)
        self.assertEqual(384, len(m["games"]))
        self.assertEqual(6, len(store.read(self.root / "roster.json")))

    def test_issue_form(self):
        m = parse('### Agent manifest\n```json\n{"name":"x","author":"y","version":"1","run":["x"]}\n```')
        self.assertEqual(m["name"], "x")

    def test_no_effect_gate_rejects(self):
        candidate, incumbent, *refs = [self.agent(str(i)) for i in range(5)]
        panel = [{"agent": aid, "family":str(i), "panel": p, "protected":True} for i, (aid,p) in enumerate(zip(refs,("representative","original","stress")))]
        m = gates.plan_comparison(self.root, "gate-one", candidate, incumbent, panel, 32)
        self.fill(m)
        result = gates.compare(self.root, "gate-one")
        self.assertFalse(result["pass"])

    def test_disconnected_ratings(self):
        result = summary([], ["a", "b"], bootstrap=0)
        self.assertEqual(len(result["components"]), 2)

    def test_frozen_sample_budget(self):
        self.roster(2)
        schedule.plan(self.root, "one", count=2)
        with self.assertRaises(ValueError):
            schedule.plan(self.root, "one", count=3)

    def test_outcome_cash_mismatch(self):
        self.roster(2)
        m = schedule.plan(self.root, "one", count=1)
        g = m["games"][0]
        with self.assertRaises(ValueError):
            schedule.save_result(self.root, m, g, {"game":g["id"],"agents":g["agents"],"resolved":True,"reason":"terminal","terminal":True,"outcome":"win0","cash":[0,3000]})

    def test_topup_adds_only_missing_seeds(self):
        a, b = self.roster(2)
        m = schedule.plan(self.root, "one", count=2)
        self.fill(m)
        followup = schedule.topup(self.root, "extra", "one", a, b, 5)
        self.assertEqual(6, len(followup["games"]))
        self.assertFalse({g["seed"] for g in m["games"]} & {g["seed"] for g in followup["games"]})

    def test_report_escapes_names(self):
        a = self.agent("safe")
        path = self.root / "agents" / (a + ".json")
        record = store.read(path)
        record["manifest"]["name"] = '<script>alert(1)</script>'
        store.write(path, record)
        reporting.build(self.root, bootstrap=0)
        html = (self.root / "site/index.html").read_text()
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_coordinator_lock(self):
        with store.locked(self.root):
            with self.assertRaises(RuntimeError):
                with store.locked(self.root):
                    pass

    def test_rating_recovers_stronger_agent(self):
        rows = [{"agents":["a","b"],"outcome":"win0","seed":i,"cash":[2,1]} for i in range(30)]
        rows += [{"agents":["b","a"],"outcome":"win1","seed":i,"cash":[1,2]} for i in range(30)]
        result = summary(rows, ["a","b"], bootstrap=10)
        self.assertGreater(result["ratings"]["a"], result["ratings"]["b"])

    def test_terminal_result_is_not_overwritten(self):
        self.roster(2)
        m = schedule.plan(self.root, "one", count=1)
        self.fill(m)
        g = m["games"][0]
        saved = store.read(self.root / "runs/one/games" / (g["id"]+".json"))
        self.assertFalse(schedule.save_result(self.root,m,g,saved))


if __name__ == "__main__":
    unittest.main()
