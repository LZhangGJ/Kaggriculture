import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dispatcher as d
import priority_hook


class DispatchTests(unittest.TestCase):
    def test_dispatch_preserves_destination_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);model=r/'model.pt';model.write_bytes(b'trusted-checkpoint')
            candidate=dict(path=str(model),sha256=d.sha(model));opp=dict(name='PASS',kind='pass')
            job=(candidate,opp,9,dict(image='pinned'),'unused');key=d.digest(['PASS',9])
            rows=[dict(opponent='PASS',seed=9,seat=s,valid=True,terminal=True,reason='terminal',
                statuses=['DONE','DONE'],cash=[3,3],score=.5,checkpoint_sha256=candidate['sha256'],
                opponent_sha256=None) for s in (0,1)]
            def remote(host,code):
                self.assertEqual(host,'arena-host')
                if 'config.json' in code:return dict(image='pinned')
                if '.exists()' in code:return True
                return {key:rows,'status':dict(state='complete')}
            host=dict(host='arena-host',root='/remote',arena_root='/arena',coordinator_arena=tmp)
            with patch.object(d,'remote_json',side_effect=remote),patch.object(d,'ssh'),patch.object(d,'copy_file') as cp:
                d.dispatch(host,[(key,job)],r,'request',None)
                self.assertTrue(cp.called)
                self.assertTrue(all(call.args[0]=='arena-host' for call in cp.call_args_list))

    def test_terminal_receipt_identity_and_score(self):
        job=({'sha256':'candidate'},{'name':'opponent','sha256':'opponent'},7,{},'unused')
        rows=[dict(seat=s,seed=7,opponent='opponent',valid=True,terminal=True,
            reason='terminal',statuses=['DONE','DONE'],cash=[10,5],score=1-s,
            checkpoint_sha256='candidate',opponent_sha256='opponent') for s in (0,1)]
        d.check_rows(rows,job)
        for field,value in [('score',.5),('terminal',False),('checkpoint_sha256','wrong'),('cash',[float('nan'),5])]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):d.check_rows(bad,job)
        with self.assertRaises(ValueError):d.check_rows([rows[0],rows[0]],job)

    def test_queue_drains_then_returns_to_arena(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);q=r/'queue';q.mkdir()
            (r/'priority-eval.json').write_text(json.dumps(dict(queue=str(q),command=['worker'])))
            with patch.object(priority_hook.subprocess,'run') as run:
                self.assertFalse(priority_hook.work(r))
                (q/'request.json').write_text('{}')
                self.assertTrue(priority_hook.work(r));run.assert_called_once()
                (q/'request.done').write_text('{}')
                self.assertFalse(priority_hook.work(r))

    def test_resume_retains_assignment_and_avoids_duplicate_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);hosts=r/'hosts.json';hosts.write_text(json.dumps([dict(host='one',workers=1)]))
            candidate=dict(sha256='a');panel=dict(config={},opponents=[dict(name='PASS',kind='pass',seeds=[9])])
            calls=[]
            def fake(host,jobs,out,rid,stop):
                calls.append(jobs)
                for key,job in jobs:
                    d.atomic(out/(key+'.json'),[dict(opponent='PASS',seed=9,seat=s,valid=True,
                        terminal=True,reason='terminal',statuses=['DONE','DONE'],cash=[3,3],score=.5,
                        checkpoint_sha256='a',opponent_sha256=None) for s in (0,1)])
            with patch.object(d,'CONFIG',hosts),patch.object(d.core,'runtime_identity',return_value='runtime'),patch.object(d,'dispatch',side_effect=fake):
                d.evaluate(candidate,panel,r/'out','arena')
                plan=(r/'out/dispatch-plan.json').read_text()
                hosts.write_text(json.dumps([dict(host='different',workers=20)]))
                d.evaluate(candidate,panel,r/'out','arena')
                self.assertEqual(len(calls),1)
                self.assertEqual((r/'out/dispatch-plan.json').read_text(),plan)

if __name__=='__main__':unittest.main()
