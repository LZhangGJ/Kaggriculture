import tempfile
import unittest
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch
from tools.arena import discovery, intake, store


class DiscoveryTest(unittest.TestCase):
    def test_recent_update_window(self):
        at=datetime(2026,9,14,12,tzinfo=timezone.utc)
        self.assertTrue(discovery.recently_updated(at-timedelta(hours=24),at))
        self.assertTrue(discovery.recently_updated(datetime(2026,9,14,11),at))
        self.assertFalse(discovery.recently_updated(at-timedelta(hours=24,seconds=1),at))
        self.assertFalse(discovery.recently_updated(at+timedelta(seconds=1),at))
        self.assertFalse(discovery.recently_updated(None,at))
    def test_paired_gate_rejects_tie_and_accepts_gain(self):
        games=[]
        for seed in range(128):
            for aid in ('new','old'):
                for seat in (0,1):
                    agents=[aid,'opponent'] if seat==0 else ['opponent',aid]
                    games.append(dict(id=str(len(games)),agents=agents,seed=seed))
        m=dict(id='trial',games=games,public_challenge=dict(candidate='new',incumbent='old',panel=['opponent']))
        def result(path):
            g=games[int(Path(path).stem)]
            winner=g['agents'].index('new') if 'new' in g['agents'] else g['agents'].index('opponent')
            return dict(game=g['id'],agents=g['agents'],resolved=True,reason='terminal',terminal=True,
                        outcome=f'win{winner}',cash=[1,0] if winner==0 else [0,1])
        with patch.object(discovery,'read',side_effect=result):
            self.assertTrue(discovery.evaluate(Path('unused'),m)['passed'])
        def tie(path):
            r=result(path);r.update(outcome='draw',cash=[1,1]);return r
        with patch.object(discovery,'read',side_effect=tie):
            self.assertFalse(discovery.evaluate(Path('unused'),m)['passed'])

    def test_replacement_preserves_frozen_run_and_roster_size(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);store.init(root)
            cfg=store.read(root/'config.json');cfg['public_discovery_enabled']=True;store.write(root/'config.json',cfg)
            ids=[]
            for name in ('old','new','team'):
                archive=root/(name+'.zip')
                with zipfile.ZipFile(archive,'w') as z:z.writestr('main.py',name)
                aid=intake.submit(root,archive,dict(name=name,author='test',version='1',run=['python','main.py']))['agent']
                a=store.read(root/'agents'/f'{aid}.json');a['build_verified']=True;store.write(root/'agents'/f'{aid}.json',a);ids.append(aid)
            old,new,team=ids
            roster=[dict(agent=old,category='public',reason='test'),dict(agent=team,category='established',reason='test')]
            intake.set_roster(root,roster)
            m=dict(id='public-test',public_challenge=dict(candidate=new,incumbent=old,roster_hash=store.digest(roster)))
            path=root/'runs/public-test/manifest.json';store.write(path,m);before=path.read_bytes()
            decision=dict(candidate=new,incumbent=old,passed=True)
            with patch.object(discovery.schedule,'complete',return_value=True),patch.object(discovery,'evaluate',return_value=decision):
                discovery.advance(root)
            self.assertEqual({new,team},{e['agent'] for e in store.read(root/'roster.json')})
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(store.read(root/'agents'/f'{old}.json')['status'],'archived')
