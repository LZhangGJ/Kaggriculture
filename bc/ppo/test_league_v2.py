import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ppo.league_runtime import PROTOCOL,LiveLeague,copy_weights,storage,admit
from ppo.league_eval import summarize,paired_stats,regressions
from ppo.league_controller import Controller

def spec(i):return dict(path='/tmp/'+str(i),sha256=str(i).zfill(64))

class LeagueTests(unittest.TestCase):
    def test_controller_nomination_recovery_and_ingest_identity(self):
        from ppo.league import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);training=root/'training';training.mkdir();(training/'snapshots').mkdir()
            manifest=root/'manifest.json';panel=root/'panel.json'
            m=dict(protocol=PROTOCOL,version=0,champion=spec(1),bc=spec(1),recent=[],roster=[spec(1),spec(2)])
            manifest.write_text(json.dumps(m));panel.write_text(json.dumps(dict(config={},seeds=[1],opponents=[dict(name='PASS')])))
            (training/'latest.json').write_text(json.dumps(dict(update=55)))
            c=Controller(root,training,manifest,panel,d)
            c.put('best_candidate',dict(checkpoint=spec(2),score=1))
            c.nominate();job=c.get('active')
            self.assertEqual(len(set(job['seeds'])),256)
            self.assertEqual(json.loads((root/'latest-event.json').read_text())['body']['id'],job['id'])
            c.nominate();self.assertEqual(c.get('active'),job)
            c.db.close();c=Controller(root,training,manifest,panel,d)
            self.assertEqual(c.get('active'),job)
            c.put('last_ingested_update',49)
            for n in (59,69):
                (training/'snapshots'/f'update-{n:06d}.pt').write_bytes(str(n).encode())
            latest=training/'snapshots/update-000069.pt'
            (training/'snapshot-request.json').write_text(json.dumps(dict(path=str(latest),sha256=sha(latest),update=69)))
            c.ingest();self.assertEqual(c.get('pending_dev')['update'],59)
            c.ingest();self.assertEqual(c.get('pending_dev')['update'],69)
            self.assertEqual(c.get('pending_dev')['sha256'],sha(latest))
            version=json.loads(manifest.read_text())['version'];c.ingest()
            self.assertEqual(json.loads(manifest.read_text())['version'],version)
            # Prepared decisions apply once, preserve the newer mailbox, and recover idempotently.
            c.put('best_candidate',dict(checkpoint=job['candidate'],score=1))
            result=dict(candidate=job['candidate'],incumbent=job['incumbent'],comparison_accepted=True,promoted=False)
            c.finalize(job,result);self.assertTrue(result['promoted'])
            promoted_version=json.loads(manifest.read_text())['version']
            c.finalize(job,result);self.assertEqual(json.loads(manifest.read_text())['version'],promoted_version)
            self.assertEqual(c.get('pending_dev')['update'],69);self.assertIsNone(c.get('best_candidate'))
            # An already-tested candidate must not block later candidates.
            m['champion']=spec(1);manifest.write_text(json.dumps(m))
            c.put('best_candidate',dict(checkpoint=job['candidate'],score=1));c.put('last_nomination',-50)
            c.nominate();self.assertIsNone(c.get('best_candidate'))
            # A stale accepted comparison cannot claim a promotion.
            m['champion']=spec(7);manifest.write_text(json.dumps(m))
            c.finalize(job,dict(result,promoted=False))
            self.assertTrue(json.loads((root/'confirmations'/job['id']/'decision.json').read_text())['promoted'])
            c.failed_attempt('confirmation',job,OSError('post-commit reporting failure'))
            self.assertTrue(json.loads((root/'confirmations'/job['id']/'decision.json').read_text())['promoted'])
            stale=dict(job,id='new-stale-job');stale_result=dict(result,promoted=False)
            c.finalize(stale,stale_result);self.assertFalse(stale_result['promoted'])
            c.db.close()

    def test_checkpoint_publication_crash_recovery(self):
        from ppo.league_runtime import finalize_checkpoint
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'update-000059.pt'
            torch.save(dict(architecture='fixture',schema='fixture',cache_identity={},worker_quantities=[],market_quantities=[],model={'x':torch.ones(2)},update=59),source)
            finalize_checkpoint(source,root,59,spec(0))
            expected=json.loads((root/'snapshot-request.json').read_text())
            (root/'snapshot-request.json').unlink()
            finalize_checkpoint(source,root,59,spec(0))
            self.assertEqual(json.loads((root/'snapshot-request.json').read_text()),expected)
            self.assertEqual(torch.load(expected['path'],weights_only=True)['update'],59)
            altered=torch.load(source,weights_only=True);altered['model']['x'].fill_(2);torch.save(altered,source)
            with self.assertRaises(ValueError):finalize_checkpoint(source,root,59,spec(0))

    def test_committed_development_survives_report_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);panel=root/'panel.json';panel.write_text(json.dumps(dict(opponents=[])))
            c=Controller(root,root,root/'manifest.json',panel,d)
            item=dict(spec(2),update=2);newer=dict(spec(3),update=3);c.put('pending_dev',item)
            def evaluated(*args):
                c.put('pending_dev',newer)
                return dict(failures=0,opponents={'other':dict(score=.5)}),[]
            with patch('ppo.league_controller.evaluate',side_effect=evaluated),patch('ppo.league_controller.atomic',side_effect=OSError('report unavailable')):
                c.develop(item)
            c.db.close();c=Controller(root,root,root/'manifest.json',panel,d)
            self.assertEqual(c.get('pending_dev'),newer)
            self.assertEqual(c.get('best_candidate')['checkpoint'],item)
            self.assertIsNotNone(c.db.execute('select 1 from events where id=?',('dev-'+item['sha256'],)).fetchone())
            c.db.close()

    def test_panel_fails_closed(self):
        from ppo.league_eval import validate_panel
        with self.assertRaises(ValueError):validate_panel(dict(opponents=[dict(name='public',seeds=[1])]))
        with self.assertRaises(ValueError):validate_panel(dict(opponents=[dict(name='PASS',kind='pass',seeds=[1,1])]))
        a=dict(failures=0,opponents={'PASS':dict(score=0),'other':dict(score=0)})
        b=dict(failures=0,opponents={'PASS':dict(score=1),'other':dict(score=.5)})
        self.assertEqual(regressions(a,b),['PASS','other'])

    def test_evaluator_timeout_cleanup_and_retry_budget(self):
        import multiprocessing as mp
        import sys,types
        from unittest.mock import MagicMock
        from ppo.league_eval import evaluate
        with tempfile.TemporaryDirectory() as d:
            pool=MagicMock();pool.apply_async.return_value.get.side_effect=mp.TimeoutError('fixture timeout')
            sandbox=types.ModuleType('tools.arena.sandbox');sandbox.cleanup=MagicMock()
            with patch('ppo.league_eval.runtime_identity',return_value='fixture'),patch('ppo.league_eval.mp.get_context') as context,patch.dict(sys.modules,{'tools.arena.sandbox':sandbox}):
                context.return_value.Pool.return_value=pool
                with self.assertRaises(mp.TimeoutError):evaluate(spec(1),dict(config={},opponents=[dict(name='PASS',kind='pass',seeds=[123])]),d,d)
            pool.terminate.assert_called_once();pool.join.assert_called_once();self.assertEqual(sandbox.cleanup.call_count,2)
            root=Path(d);panel=root/'panel.json';panel.write_text(json.dumps(dict(opponents=[])))
            c=Controller(root,root,root/'manifest.json',panel,d);item=spec(2);c.put('pending_dev',item)
            c.failed_attempt('development',item,mp.TimeoutError());self.assertEqual(c.get('pending_dev'),item)
            c.failed_attempt('development',item,mp.TimeoutError());self.assertIsNone(c.get('pending_dev'))
            c.db.close()

    def test_assignments_rotation_and_bounded_admission(self):
        m=dict(protocol=PROTOCOL,version=0,champion=spec(0),bc=spec(0),recent=[],roster=[spec(0)])
        for i in range(1,1001):m=admit(m,spec(i))
        self.assertLessEqual(len(m['roster']),8)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text(json.dumps(m))
            live=LiveLeague(p,{},None,None,spec(9));live.loaded=dict(reference='r',champion_slot='c',history_slot='h')
            for update in (0,17,100000):
                alljobs=[]
                for rank in range(2):
                    jobs=live.jobs(190000000,update,512,2,rank);alljobs+=jobs
                    self.assertEqual(len(jobs),256)
                    self.assertEqual(sum(len(x['learner_seats']) for x in jobs),384)
                    for family in ('champion_slot','history_slot'):
                        rows=[x for x in jobs if x['family']==family]
                        self.assertEqual(len(rows),64)
                        self.assertEqual(sum(x['learner_seats']==[0] for x in rows),32)
                self.assertEqual(len({x['seed'] for x in alljobs}),512)

    def test_copy_preserves_storage_and_does_not_alias(self):
        a=torch.nn.Linear(3,2);b=copy.deepcopy(a);old=storage(b)
        with torch.no_grad():a.weight.fill_(3)
        copy_weights(b,a.state_dict());self.assertEqual(storage(b),old)
        self.assertTrue(torch.equal(a.weight,b.weight));self.assertNotEqual(a.weight.data_ptr(),b.weight.data_ptr())

    def test_exact_assignment_and_paired_confirmation(self):
        p=dict(opponents=[dict(name='opp',seeds=list(range(256)))])
        rows=[dict(opponent='opp',seed=s,seat=t,valid=True,score=1 if s<180 else 0,cash=[10,5]) for s in range(256) for t in (0,1)]
        self.assertEqual(summarize(rows,p)['failures'],0)
        self.assertTrue(paired_stats(rows,'opp')['accepted'])
        with self.assertRaises(ValueError):summarize(rows[:-1]+[rows[0]],p)
        rows[0]['valid']=False
        with self.assertRaises(ValueError):paired_stats(rows,'opp')

    def test_boundary_refresh_resume_and_rejected_binding(self):
        m=dict(protocol=PROTOCOL,version=0,champion=spec(1),bc=spec(1),recent=[],roster=[spec(2),spec(3)])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';p.write_text(json.dumps(m))
            models={k:torch.nn.Linear(2,2) for k in ('champion_slot','history_slot')};ref=torch.nn.Linear(2,2)
            def staged(model,specification,identity):
                return {k:torch.full_like(v,float(int(specification['sha256']))) for k,v in model.state_dict().items()}
            live=LiveLeague(p,models,ref,{},spec(4))
            with patch('ppo.league_runtime.stage',side_effect=staged):
                live.boundary(0);live.complete();live.boundary(1);live.complete()
                state=copy.deepcopy(live.state)
                resumed=LiveLeague(p,models,ref,{},spec(99),state)
                live.boundary(2);resumed.boundary(2)
                self.assertEqual(live.state,resumed.state)
                self.assertEqual(live.loaded['history_slot'],spec(3)['sha256'])
                live.state['reference_age']=10;live.boundary(10,spec(5))
                self.assertEqual(live.loaded['reference'],spec(5)['sha256'])
                self.assertEqual(live.state['reference_refreshes'],1)
            previous=copy.deepcopy(live.state)
            changed=copy.deepcopy(m);changed['version']=1;changed['champion']=spec(77);p.write_text(json.dumps(changed))
            with patch('ppo.league_runtime.stage',side_effect=ValueError('bad hash')):
                self.assertFalse(live.boundary(11))
            self.assertEqual(live.state,previous)
            live.state['reference_age']=10
            def rejected_opponent(model,specification,identity):
                if specification==spec(77):raise ValueError('missing desired opponent')
                return staged(model,specification,identity)
            with patch('ppo.league_runtime.stage',side_effect=rejected_opponent):
                self.assertFalse(live.boundary(12,spec(6)))
            self.assertEqual(live.loaded['reference'],spec(6)['sha256'])
            self.assertEqual(live.state['reference_age'],0)

if __name__=='__main__':unittest.main()
