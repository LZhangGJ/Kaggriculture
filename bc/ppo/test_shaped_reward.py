"""shaped-reward-v1 focused CPU checks: reward, loss term, head tolerance, export, migration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from exact_model import ExactWorkerMarketPolicyV1
from ppo.replay import shaped_reward, ppo_terms
from ppo.shaped_reward import migrate_state, SHAPED_HEAD_KEYS, with_shaped_head, without_shaped_head, attach_shaped_head, load_exact_with_head
from ppo.train import make_optimizer, save_checkpoint
from ppo import league_runtime


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5);cls.model=attach_shaped_head(ExactWorkerMarketPolicyV1()).eval()

    def test_shaped_reward_values(self):
        r=shaped_reward(torch.tensor([-60000.,-25000.,-5000.,0.,5000.,25000.,60000.]),1.,25000.)
        torch.testing.assert_close(r,torch.tensor([-2.,-2.,-1.2,0.,1.2,2.,2.]))
        torch.testing.assert_close(shaped_reward(torch.tensor([-30000.,30000.]),0.,25000.),torch.tensor([-1.,1.]))
        self.assertEqual(float(shaped_reward(-12500.,1.,25000.)),-1.5)
        self.assertTrue((shaped_reward(torch.randn(1000)*1e5,1.,25000.).abs()<=2.).all())

    def test_head_zero_init_and_grouping(self):
        for k in SHAPED_HEAD_KEYS:self.assertTrue(torch.equal(self.model.state_dict()[k],torch.zeros_like(self.model.state_dict()[k])))
        opt=make_optimizer(self.model,1e-5,5e-5)
        self.assertEqual(len(opt.param_groups),3);head_params=[id(p) for p in opt.param_groups[2]['params']]
        self.assertEqual(head_params,[id(self.model.value_shaped.weight),id(self.model.value_shaped.bias)]);self.assertAlmostEqual(opt.param_groups[2]['lr'],5e-3)
        self.assertFalse(any(id(p)==id(self.model.value_shaped.weight) for p in opt.param_groups[1]['params']))
        names=[n for n,_ in self.model.named_parameters()]
        self.assertEqual(names[-2:],list(SHAPED_HEAD_KEYS))

    def test_ppo_terms_requires_returns_and_adds_huber(self):
        n=6;result=dict(logp=torch.zeros(n),logits=torch.zeros(n,3),value=torch.tensor([0.,0.,0.,0.,0.,0.]),entropy=torch.zeros(n),kl=torch.zeros(n),factor_count=torch.ones(n))
        old=torch.zeros(n);adv=torch.zeros(n);outcome=torch.zeros(n,dtype=torch.long)
        with self.assertRaisesRegex(ValueError,'shaped returns'):ppo_terms(result,old,adv,outcome)
        returns=torch.tensor([-2.,-1.,-.5,0.,.5,2.])
        loss,stats=ppo_terms(result,old,adv,outcome,returns=returns,vf=0.,ent=0.,anchor=0.,vf_shaped=1.)
        # Huber(delta=1) of prediction 0 against the return: quadratic inside |r|<=1, linear outside.
        torch.testing.assert_close(stats['value_shaped'],torch.tensor([1.5,.5,.125,0.,.125,1.5]))
        torch.testing.assert_close(loss,stats['value_shaped'])
        torch.testing.assert_close(stats['shaped_prediction'],result['value'])
        adv=torch.tensor([1.,-1.,2.,-2.,.5,-.5]);old=torch.zeros(n);res2=dict(result,logp=torch.full((n,),.05))
        l1,_=ppo_terms(res2,old,adv,outcome,returns=returns,vf=0.,ent=0.,anchor=0.,vf_shaped=0.,policy_weight=1.)
        l0,_=ppo_terms(res2,old,adv,outcome,returns=returns,vf=0.,ent=0.,anchor=0.,vf_shaped=0.,policy_weight=0.)
        self.assertGreater(float(l1.abs().sum()),0.);torch.testing.assert_close(l0,torch.zeros(n))

    def test_tolerant_load_and_export_roundtrip(self):
        full=self.model.state_dict()
        stripped=without_shaped_head(full)
        self.assertEqual(set(full)-set(stripped),set(SHAPED_HEAD_KEYS))
        fresh=attach_shaped_head(ExactWorkerMarketPolicyV1())
        ck=dict(architecture=self.model.architecture,schema=self.model.schema,model=stripped)
        load_exact_with_head(fresh,ck)  # old-format checkpoint loads with a zero head
        with self.assertRaisesRegex(ValueError,'exact worker/market'):load_exact_with_head(fresh,dict(ck,schema='other'))
        plain=ExactWorkerMarketPolicyV1();plain.load_exact(ck)  # unchanged architecture module still loads exports strictly
        for k,v in full.items():self.assertTrue(torch.equal(fresh.state_dict()[k],v),k)
        again=with_shaped_head(stripped,fresh);self.assertEqual(set(again),set(full))
        with_head=with_shaped_head(full,fresh);self.assertTrue(torch.equal(with_head['value_shaped.weight'],full['value_shaped.weight']))
        # stage() accepts a stripped roster snapshot and returns the full key set.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snap.pt';torch.save(dict(cache_identity={'id':1},model=stripped),p)
            weights=league_runtime.stage(fresh,dict(path=str(p),sha256=league_runtime.sha(p)),{'id':1})
            self.assertEqual(set(weights),set(full))
            league_runtime.copy_weights(fresh,weights)

    def test_snapshot_export_strips_head(self):
        opt=make_optimizer(self.model,1e-5,5e-5)
        with tempfile.TemporaryDirectory() as d:
            base=dict(architecture=self.model.architecture,schema=self.model.schema,cache_identity={},worker_quantities=[0],market_quantities=[0])
            ck=Path(d)/'update-000009.pt';save_checkpoint(ck,self.model,opt,base,{'t':1},9,[{'cpu':torch.get_rng_state(),'cuda':None}])
            self.assertIn('value_shaped.weight',torch.load(ck,weights_only=True)['model'])
            item=league_runtime.snapshot(ck,Path(d)/'snapshots'/'update-000009.pt')
            exported=torch.load(item['path'],weights_only=True)
            self.assertNotIn('value_shaped.weight',exported['model']);self.assertNotIn('value_shaped.bias',exported['model'])
            self.assertEqual(set(exported['model'])|set(SHAPED_HEAD_KEYS),set(self.model.state_dict()))
            # Idempotent re-snapshot with identical content passes the existing provenance check.
            league_runtime.snapshot(ck,item['path'])

    def test_migration_preserves_state_and_registers_head(self):
        # Build an old-format full checkpoint: model without head, optimizer without the two head params but with moments.
        old_model=copy.deepcopy(self.model);old_opt=make_optimizer(old_model,1e-5,5e-5)
        del old_opt.param_groups[2]  # simulate the pre-migration two-group optimizer
        for p in old_model.parameters():p.grad=torch.randn_like(p)*1e-3
        old_model.value_shaped.weight.grad=None;old_model.value_shaped.bias.grad=None
        old_opt.step()
        saved=dict(model=without_shaped_head({k:v.detach().clone() for k,v in old_model.state_dict().items()}),optimizer=old_opt.state_dict(),update=42)
        n_state=len(saved['optimizer']['state']);old_groups=copy.deepcopy(saved['optimizer']['param_groups']);old_state=copy.deepcopy(saved['optimizer']['state'])
        new_model=attach_shaped_head(ExactWorkerMarketPolicyV1());receipt=migrate_state(saved,new_model)
        self.assertEqual(receipt['added_optimizer_param_ids'],[sum(len(g['params']) for g in old_groups),sum(len(g['params']) for g in old_groups)+1])
        new_model.load_state_dict(saved['model'],strict=True)
        new_opt=make_optimizer(new_model,1e-5,5e-5);new_opt.load_state_dict(saved['optimizer'])
        self.assertEqual(len(new_opt.state),n_state)  # no state for the fresh head
        sd=new_opt.state_dict()
        self.assertEqual(sd['param_groups'][0]['params'],old_groups[0]['params'])
        self.assertEqual(sd['param_groups'][1]['params'],old_groups[1]['params']);self.assertEqual(sd['param_groups'][2]['params'],receipt['added_optimizer_param_ids']);self.assertAlmostEqual(sd['param_groups'][2]['lr'],5e-3)
        for i,st in old_state.items():
            for k,v in st.items():
                if torch.is_tensor(v):self.assertTrue(torch.equal(sd['state'][i][k],v),(i,k))
                else:self.assertEqual(sd['state'][i][k],v)
        for k,v in old_model.state_dict().items():self.assertTrue(torch.equal(new_model.state_dict()[k],v),k)
        for k in SHAPED_HEAD_KEYS:self.assertTrue(torch.equal(new_model.state_dict()[k],torch.zeros_like(new_model.state_dict()[k])))
        with self.assertRaisesRegex(ValueError,'already carries'):migrate_state(saved,new_model)


if __name__=='__main__':unittest.main(verbosity=2)


class Composition(unittest.TestCase):
    def test_value_is_utility_plus_residual_and_detached(self):
        from ppo.replay import utility
        model=attach_shaped_head(ExactWorkerMarketPolicyV1()).eval()
        critic=torch.randn(4,256,requires_grad=True)
        logits=model.value(critic);residual=model.value_shaped(critic.detach()).squeeze(-1)
        value=utility(logits).detach()+residual
        torch.testing.assert_close(value,utility(logits).detach())  # zero head at migration time
        with torch.no_grad():model.value_shaped.weight.fill_(.01)
        value=utility(logits).detach()+model.value_shaped(critic.detach()).squeeze(-1)
        value.sum().backward()
        self.assertIsNone(critic.grad)  # no Huber-path gradient into the trunk
        self.assertIsNotNone(model.value_shaped.weight.grad)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n.startswith('value.')))


class WarmUp(unittest.TestCase):
    def test_freeze_actor_gradients_steps_only_value_heads(self):
        from ppo.shaped_reward import freeze_actor_gradients
        torch.manual_seed(3);model=attach_shaped_head(ExactWorkerMarketPolicyV1())
        opt=make_optimizer(model,1e-3,1e-3)
        before={k:v.clone() for k,v in model.state_dict().items()}
        loss=sum(p.sum() for p in model.parameters());loss.backward()
        dropped=freeze_actor_gradients(model,opt)
        self.assertEqual(dropped,len(opt.param_groups[0]['params']))
        opt.step()
        critic={id(p) for g in opt.param_groups[1:] for p in g['params']}
        for n,p in model.named_parameters():
            if id(p) in critic:self.assertFalse(torch.equal(p,before[n]),n)
            else:self.assertTrue(torch.equal(p,before[n]),n)
        self.assertEqual(len(opt.state),len(opt.param_groups[1]['params'])+len(opt.param_groups[2]['params']))  # no Adam state created for frozen params
