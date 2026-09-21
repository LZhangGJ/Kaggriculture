"""Exact adapter parity on boundary prefixes and sampled official states."""
import copy
import unittest
import numpy as np
import torch
from ppo.fast_features import WorkerPrefix,own_farm,worker_stats,market_stats,QuantityFeatures,expand_worker_cached,worker_features_fast,encode_exact_fast
from exact_actions import WORKER,worker_quantity_features,worker_features,expand_worker
from exact_features import encode_exact
from exact_decoder import market_state
from features_v2 import PublicHistory,quantity_features
from bc_runtime import MARKET
from worker_phase import resolve_worker_phase,engine
from ppo.evaluate import LocalBackend

def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else: np.testing.assert_array_equal(a,b)

class FastChecks(unittest.TestCase):
    def test_prefix_farm_and_quantities(self):
        torch.set_num_threads(1)
        jobs=[dict(game=0,seed=9194321,policies=['script:starter']*2,learner_seats=[],family='starter')]
        backend=LocalBackend(jobs,[-1,0,1,100],[1,2,100])
        wq=[-10,0,1,2,37,100,1000];mq=[1,2,10,100,1000]
        qgen=QuantityFeatures(wq,mq,'cpu');history=PublicHistory()
        # The first plant must physically change a tile before cancellation.
        o=copy.deepcopy(dict(backend.games.games[0]['env'].state[0].observation))
        f=o['farms'][0];f['farmer']=[0,0];f['hands']=[[1,0]]
        f['tiles'][0][0]=None;f['tiles'][0][1]=None
        o['private']['inventories']=[{},{}];o['private']['seeds']['WHEAT']=1
        prefix=WorkerPrefix(o);prefix.append(['PLANT','WHEAT'])
        self.assertEqual(prefix.resolved['farms'][0]['tiles'][0][0]['crop'],'WHEAT')
        prefix.append(['PLANT','WHEAT'])
        equal(prefix.resolved,resolve_worker_phase(o,[['PLANT','WHEAT']]*2)[0])
        self.assertIsNone(prefix.resolved['farms'][0]['tiles'][0][0])
        for t in range(720):
            obs=dict(backend.games.games[0]['env'].state[0].observation)
            history.observe(obs)
            if t%47==0 or t==719:
                equal(own_farm(obs),encode_exact(obs,history)['farms'][0])
                equal(encode_exact_fast(obs,history),encode_exact(obs,history))
                o=copy.deepcopy(obs);f=o['farms'][0]
                f['hands']=[[3,4]]*7;o['private']['inventories']=[{} for _ in range(8)]
                o['private']['seeds']['WHEAT']=1;o['private']['seeds']['CARROT']=1
                f['farmer']=list(sorted(engine()._shed_access_tiles(10))[0]);o['private']['shed']['WHEAT']=90
                prefix=WorkerPrefix(o)
                slots=[['PLANT','WHEAT'],['PICKUP','WHEAT',2],['PLANT','CARROT'],['PLANT','WHEAT'],['PLACE','WHEAT',2],['PLANT','CARROT'],['NORTH'],['HARVEST']]
                for i,a in enumerate(slots):
                    ref,_,_=resolve_worker_phase(o,slots[:i]);equal(prefix.resolved,ref)
                    compact=worker_features(o,ref,slots[:i],i)
                    equal(compact,worker_features_fast(o,prefix.resolved,slots[:i],i,prefix.points()))
                    packed={k:torch.as_tensor(np.asarray(v)[None]) for k,v in compact.items()}
                    for k,v in expand_worker(packed).items(): torch.testing.assert_close(v,expand_worker_cached(packed)[k],rtol=0,atol=0)
                    ix=torch.arange(44);got,mask=qgen(0,ix,torch.tensor(worker_stats(ref,i)))
                    for j in range(44):
                        expected,em=worker_quantity_features(ref,i,j,wq)
                        np.testing.assert_allclose(got[j].numpy(),expected,rtol=1e-7,atol=1e-7);equal(mask[j].numpy(),em)
                    prefix.append(a)
                equal(prefix.resolved,resolve_worker_phase(o,slots)[0])
                equal(own_farm(prefix.resolved),encode_exact(prefix.resolved,history)['farms'][0])
                ledger=market_state(prefix.resolved)
                for request in (None,(4,10),(2,0),(15,100)):
                    if request:ledger.add_order(*request)
                    ix=torch.tensor([i for i,(_,item) in enumerate(MARKET) if i>1 and item is not None])
                    got,mask=qgen(1,ix,torch.tensor(market_stats(ledger))[ix])
                    for j,index in enumerate(ix.tolist()):
                        expected,em=quantity_features(ledger,index,mq)
                        np.testing.assert_allclose(got[j].numpy(),expected,rtol=1e-7,atol=1e-7);equal(mask[j].numpy(),em)
            if t<719: backend.call('step',[0])

if __name__=='__main__':unittest.main(verbosity=2)
