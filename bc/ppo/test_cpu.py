"""Small CPU semantic checks. No complete games, GPU use, or production writes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest
import copy
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from exact_model import ExactWorkerMarketPolicyV1
from exact_decoder import prepare_turn
from exact_training import pack_turns, merge_chunks
from features_v2 import PublicHistory
from ppo.replay import ExactPPOChunk, gae, ppo_terms, categorical_kl
from ppo.sampler import BatchedExactSampler
from ppo.evaluate import LocalBackend
from ppo.actors import ActorPool
from ppo.transport import ArrayArena
from ppo.league import assignments


def equal_arrays(expected,actual):
    if isinstance(expected,dict):
        if expected.keys()!=actual.keys(): raise AssertionError('Feature keys differ')
        for key in expected: equal_arrays(expected[key],actual[key])
    else: np.testing.assert_array_equal(expected,actual)


class Checks(unittest.TestCase):
    def test_evaluation_terminal_accounting(self):
        from ppo.evaluate import run_panel
        terminal=dict(done=True,cash=[100,20],faults=[False,False],statuses=['DONE','DONE'],turns=719)
        class Backend:
            def __init__(self,*args): pass
            def call(self,command,payload): return {0:terminal} if command=='step' else {}
        with tempfile.TemporaryDirectory() as temp, \
             patch('ppo.train.load_model',return_value=(self.model,dict(worker_quantities=self.wq,market_quantities=self.mq))), \
             patch('ppo.evaluate.LocalBackend',Backend), \
             patch('ppo.evaluate.BatchedExactSampler'),patch('ppo.evaluate.sha',return_value='test'):
            panel=dict(opponents=[dict(id='starter',kind='script',path='starter',seeds=[1])])
            result=run_panel('unused',panel,Path(temp)/'valid')
            self.assertEqual(result['failures'],0)
            self.assertEqual(result['opponents']['starter']['wins'],1)
            self.assertEqual(result['opponents']['starter']['losses'],1)
            self.assertEqual(result['opponents']['starter']['mean_cash'],60)
            terminal['faults']=[False,True];terminal['statuses']=['DONE','ERROR']
            result=run_panel('unused',panel,Path(temp)/'failed')
            self.assertEqual(result['failures'],2)
            self.assertEqual(result['opponents']['starter']['wins'],0)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(77)
        from ppo.shaped_reward import attach_shaped_head
        cls.model=attach_shaped_head(ExactWorkerMarketPolicyV1()).eval()
        cls.wq=[-1,0,1,2,100];cls.mq=[1,2,10,100]
        cls.jobs=[dict(game=i,seed=9387000+i,policies=['learner','learner'],learner_seats=[0,1],family='self') for i in range(2)]

    def test_terminal_bootstrap(self):
        torch.testing.assert_close(gae([0,1],[.2,.4],[False,True]),torch.tensor([.77,.6]))
        torch.testing.assert_close(gae([0],[.2],[False],bootstrap=.7),torch.tensor([.5]))
        torch.testing.assert_close(gae([-1],[.2],[True],bootstrap=.7),torch.tensor([-1.2]))

    def test_shared_memory_roundtrip(self):
        arena=ArrayArena(size=4096)
        try:
            value={'a':np.arange(12).reshape(3,4),'b':[np.ones(4,dtype=bool),5]}
            spec,_=arena.write(value);got=arena.read(spec)
            np.testing.assert_array_equal(got['a'],value['a']);np.testing.assert_array_equal(got['b'][0],value['b'][0])
        finally:arena.close()

    def test_batched_density_recurrent_and_gradients(self):
        backend=LocalBackend(self.jobs,self.wq,self.mq)
        sampler=BatchedExactSampler({'learner':self.model},self.model,self.wq,self.mq)
        keys=['0:0','0:1','1:0','1:1'];turns={k:[] for k in keys}
        histories={k:PublicHistory() for k in keys}
        for step in range(3):
            starts=backend.call('start',keys)
            sampled=sampler.act('learner',keys,starts,backend)
            for k in keys:
                row=sampled[k];turns[k].append(row)
                # Independently reconstruct features from the sampled wire action using BC's canonical path.
                seat=backend.games.seats[k]
                wire=dict(farmer=seat['slots'][0],hands=seat['slots'][1:])
                prepared=prepare_turn(dict(observation=seat['obs'],worker_action=wire,market_requests=seat['ledger'].orders),histories[k],self.wq,self.mq)
                self.assertEqual(len(prepared['events']),len(row['events']))
                for field in ('x','post_ledger','post_farm'): equal_arrays(prepared[field],row[field])
                for expected,actual in zip(prepared['events'],row['events']):
                    self.assertEqual(expected['index'],actual['index'])
                    np.testing.assert_allclose(expected['ledger'],actual['ledger'])
                    self.assertEqual(expected.get('qindex'),actual.get('qindex'))
                    for field in ('compact','candidates','delta','advance','qfeatures','qmask','quantity'):
                        self.assertEqual(field in expected,field in actual)
                        if field in expected: equal_arrays(expected[field],actual[field])
                from exact_actions import worker_wire
                from bc_runtime import MARKET
                worker=[worker_wire(e['index'],e['quantity'] if 'qindex' in e else None) for e in row['events'] if e['phase']==0]
                market=[]
                for e in row['events']:
                    if e['phase']!=1 or e['index']==0: continue
                    op,item=MARKET[e['index']]
                    market.append([] if op=='NOOP' else [op] if item is None else [op,item,e['quantity']])
                self.assertEqual(worker,seat['slots']);self.assertEqual(market,seat['ledger'].orders)
            backend.call('step',[0,1])
        batch=merge_chunks([pack_turns(turns[k],0,2) for k in keys])
        state=(torch.zeros(4,256),torch.zeros(4,256))
        score=ExactPPOChunk(self.model)
        result=score(batch,state,reference=score,reference_state=state)
        with torch.no_grad():
            ref_only=score(batch,state,reference_only=True)
        torch.testing.assert_close(ref_only['logp'],result['logp'],atol=2e-6,rtol=2e-6)
        for left,right in zip(ref_only['factors'],result['factors']):
            for a,b in zip(left,right):
                if a is not None:torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        old=torch.tensor(np.stack([[r['old_logp'] for r in turns[k]] for k in keys],1).reshape(-1),dtype=torch.float32)
        torch.testing.assert_close(result['logp'],old,atol=2e-5,rtol=2e-6)
        torch.testing.assert_close(result['kl'],torch.zeros_like(old),atol=1e-6,rtol=0)
        expected=torch.tensor(np.stack([[r['factor_count'] for r in turns[k]] for k in keys],1).reshape(-1))
        torch.testing.assert_close(result['factor_count'],expected.float())
        loss,stats=ppo_terms(result,old,torch.ones_like(old),torch.full_like(old,2,dtype=torch.long),returns=torch.ones_like(old))
        torch.testing.assert_close(stats['approx_kl'],torch.zeros_like(old),atol=1e-6,rtol=0)
        self.model.zero_grad(set_to_none=True);loss.mean().backward()
        grads=[p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        saved_grads={n:p.grad.clone() for n,p in self.model.named_parameters() if p.grad is not None}
        from unittest.mock import patch
        with patch('ppo.replay.checkpoint',lambda fn,*a,**kw:fn(*a)):
            direct=score(batch,state,reference=score,reference_state=state)
            direct_loss,_=ppo_terms(direct,old,torch.ones_like(old),torch.full_like(old,2,dtype=torch.long),returns=torch.ones_like(old))
            self.model.zero_grad(set_to_none=True);direct_loss.mean().backward()
        torch.testing.assert_close(direct['logp'],result['logp'],rtol=0,atol=0)
        for n,p in self.model.named_parameters():
            if n in saved_grads: torch.testing.assert_close(p.grad,saved_grads[n],rtol=1e-5,atol=1e-7)
        # Replay from a stored interior recurrent anchor without an optimizer step.
        chunk=merge_chunks([pack_turns(turns[k][1:],1,2) for k in keys])
        anchors=torch.tensor(np.stack([turns[k][1]['state'] for k in keys]))
        interior=score(chunk,(anchors[:,0],anchors[:,1]),burn=1)
        torch.testing.assert_close(interior['logp'],old[4:],atol=2e-5,rtol=2e-6)
        # Exercise the actual update loop on synthetic targets for these three-turn traces.
        # This is not a complete rollout or a strength test.
        from types import SimpleNamespace
        from ppo.train import LossModule, update, save_checkpoint, make_optimizer
        clone=copy.deepcopy(self.model)
        optimizer=make_optimizer(clone,0.,0.)
        episodes=[dict(turns=turns[k],outcome=2,shaped_return=1.5,advantages=np.array([-.4,.1,.5],np.float32)) for k in keys]
        args=SimpleNamespace(burn=1,sequence=2,minibatch=2,seed=17,update_number=0,bf16=False,clip=.1,anchor=.01,max_kl=.03)
        before={k:v.clone() for k,v in clone.state_dict().items()}
        report=update(LossModule(clone),ExactPPOChunk(self.model).requires_grad_(False),copy.deepcopy(episodes),optimizer,'cpu',args,1)
        self.assertGreater(report['optimizer_steps'],0)
        for k,v in clone.state_dict().items(): torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        for group in optimizer.param_groups: group['lr']=1e-5
        update(LossModule(clone),ExactPPOChunk(self.model),copy.deepcopy(episodes),optimizer,'cpu',args,1)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in clone.state_dict().items()))
        with tempfile.TemporaryDirectory() as directory:
            base=dict(architecture=clone.architecture,schema=clone.schema,cache_identity={},worker_quantities=self.wq,market_quantities=self.mq)
            path=Path(directory)/'checkpoint.pt'
            save_checkpoint(path,clone,optimizer,base,{'test':True},3,[{'cpu':torch.get_rng_state(),'cuda':None}])
            saved=torch.load(path,weights_only=True)
            from ppo.shaped_reward import load_exact_with_head
            restored=attach_shaped_head(ExactWorkerMarketPolicyV1());load_exact_with_head(restored,saved)
            restored_optimizer=make_optimizer(restored,2e-5,5e-5);restored_optimizer.load_state_dict(saved['optimizer'])
            self.assertEqual(saved['update'],3)
            self.assertEqual(len(optimizer.state),len(restored_optimizer.state))
            self.assertEqual(len(restored_optimizer.param_groups),2)
            for a,b in zip(optimizer.state.values(),restored_optimizer.state.values()):
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        from ppo.rollout import windows,batch_windows
        rows=next(iter(windows(episodes,1,2).values()))
        bad,anchor,refanchor=batch_windows(rows)
        bad['factor_count'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'factor-count'):
            LossModule(clone)(bad,anchor,refanchor,ExactPPOChunk(self.model),0,1.,.1,.01)
        self.model.requires_grad_(True)

    def test_masked_kl_gradients(self):
        logits=torch.tensor([[1.,2.,3.]],requires_grad=True)
        lp=logits.masked_fill(torch.tensor([[False,True,False]]),-torch.inf).log_softmax(-1)
        kl=categorical_kl(lp.detach(),lp);kl.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_request_boundaries(self):
        from exact_actions import WORKER
        from worker_phase import engine
        cases=[([0],[0]),([5],[0]),([WORKER.index(('PICKUP','WHEAT'))],[4,0]),([15,15],[4]*10)]
        for worker,market in cases:
            backend=LocalBackend(self.jobs[:1],self.wq,self.mq)
            obs=backend.games.games[0]['env'].state[0].observation
            obs.farms[0].farmer=list(sorted(engine()._shed_access_tiles(10))[0])
            obs.private.shed['WHEAT']=2
            if len(worker)>1:
                obs.farms[0].hands=[[3,4]];obs.private.inventories.append({});obs.private.seeds['WHEAT']=1
            starts=backend.call('start',['0:0'])
            forced=iter(worker+market)
            def choose(probs,num_samples,generator=None):
                return torch.full((len(probs),1),next(forced) if probs.shape[1] in (44,23) else 0,device=probs.device,dtype=torch.long)
            sampler=BatchedExactSampler({'learner':self.model},self.model,self.wq,self.mq)
            with patch('torch.multinomial',side_effect=choose):
                row=sampler.act('learner',['0:0'],starts,backend)['0:0']
            seat=backend.games.seats['0:0']
            canonical=prepare_turn(dict(observation=seat['obs'],worker_action=dict(farmer=seat['slots'][0],hands=seat['slots'][1:]),market_requests=seat['ledger'].orders),PublicHistory(),self.wq,self.mq)
            self.assertEqual(len(canonical['events']),len(row['events']))
            for a,b in zip(canonical['events'],row['events']):
                for field in ('compact','candidates','ledger','index','quantity','delta','advance','qfeatures','qmask','qindex'):
                    self.assertEqual(field in a,field in b)
                    if field in a: equal_arrays(a[field],b[field])
            for field in ('x','post_ledger','post_farm'): equal_arrays(canonical[field],row[field])
            self.assertEqual([e['index'] for e in row['events'] if e['phase']==1],market)
            if len(market)==10:self.assertFalse(any(e['phase']==1 and e['index']==0 for e in row['events']))
            if worker[0]==WORKER.index(('PICKUP','WHEAT')):self.assertIn('qindex',row['events'][0])

    def test_cpu_process_features_match_inline(self):
        local=LocalBackend(self.jobs,self.wq,self.mq)
        pool=ActorPool(self.jobs,self.wq,self.mq,workers=2)
        try:
            keys=['0:0','1:1'];a=local.call('start',keys);b=pool.call('start',keys)
            for k in keys:
                for field in a[k]['x']: np.testing.assert_array_equal(a[k]['x'][field],b[k]['x'][field])
            a=local.call('worker',keys);b=pool.call('worker',keys)
            for k in keys: np.testing.assert_array_equal(a[k]['qstats'],b[k]['qstats'])
        finally:pool.close()

    def test_assignment_ownership(self):
        league={'opponents':[dict(id='self',kind='self',weight=.5),dict(id='bc',kind='bc',weight=.5)]}
        a=assignments(league,100,2,64,2,0);b=assignments(league,100,2,64,2,1)
        self.assertFalse({x['seed'] for x in a}&{x['seed'] for x in b})
        self.assertEqual(len(a)+len(b),64)

    def test_league_and_starter(self):
        import json
        from ppo.league import load_league,add_snapshot
        league=dict(opponents=[dict(id='current',kind='self',weight=.5),dict(id='frozen',kind='bc',weight=.4),dict(id='starter',kind='script',path='starter',weight=.1)])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'league.json';path.write_text(json.dumps(league))
            checkpoint=Path(temp)/'weights';checkpoint.write_bytes(b'test')
            add_snapshot(path,checkpoint,'one','one');add_snapshot(path,checkpoint,'two','two')
            loaded=load_league(path)
            self.assertAlmostEqual(sum(r['weight'] for r in loaded['opponents'] if r['kind']=='checkpoint'),.25)
            loaded['opponents'][-1]['id']='learner';path.write_text(json.dumps(loaded))
            with self.assertRaisesRegex(ValueError,'Reserved'): load_league(path)
        from ppo.actors import Games
        games=Games([dict(game=0,seed=9388900,policies=['script:starter']*2,learner_seats=[],family='starter')],self.wq,self.mq)
        result=games.run('step',[0])[0]
        self.assertEqual(result['statuses'],['ACTIVE','ACTIVE']);self.assertFalse(any(result['faults']))


if __name__=='__main__':unittest.main(verbosity=2)
