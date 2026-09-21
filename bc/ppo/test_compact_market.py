import copy,os,unittest
import numpy as np
import torch
from kaggle_environments import make
from exact_decoder import market_state,market_candidates,ledger_vector
from features_v2 import quantity_features
from ppo.gpu_market import market_seed
from ppo.fast_features import QuantityFeatures
from ppo.compact_market import materialize_market

class CompactMarketTests(unittest.TestCase):
    def test_regeneration_matches_canonical_ordered_requests(self):
        env=make('kaggriculture',configuration={'seed':9198001,'episodeSteps':720},debug=False)
        env.reset(2);ledgers=[]
        for cash in (0,31,10000):
            obs=copy.deepcopy(dict(env.state[0].observation));obs['farms'][0]['money']=cash
            obs['private']['shed']={'WHEAT':20,'EGG':5};ledgers.append(market_state(obs))
        seeds=[market_seed(l) for l in ledgers];device=os.environ.get('PARITY_DEVICE','cpu')
        batch={'market_seed':{k:torch.as_tensor(np.stack([s[k] for s in seeds]),device=device) for k in seeds[0]},'events':[]}
        expected=[];vocab=[1,10,100]
        for depth in range(10):
            ids=[2,0,1] if depth%2 else [1,2,0]
            indices=[0 if depth==9 else (2+(depth*3+i)%21) for i in ids]
            qr=[j for j,index in enumerate(indices) if index>=4]
            event=dict(phase=1,depth=depth,ids=torch.tensor(ids,device=device),index=torch.tensor(indices,device=device),
                       quantity=torch.tensor([10]*3,device=device),qr=torch.tensor(qr,device=device,dtype=torch.long))
            batch['events'].append(event);gold=[]
            for i,index in zip(ids,indices):
                ledger=ledgers[i];before=ledger_vector(ledger);candidate=market_candidates(ledger)
                q=quantity_features(ledger,index,vocab) if index>=4 else None
                if index:ledger.add_order(index,10)
                gold.append((before,candidate,ledger_vector(ledger)-before,q))
            expected.append(gold)
        materialize_market(batch,QuantityFeatures([1],vocab,device))
        for event,gold in zip(batch['events'],expected):
            for row,(ledger,candidates,delta,quantity) in enumerate(gold):
                np.testing.assert_allclose(event['ledger'][row].cpu(),ledger,rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(event['delta'][row].cpu(),delta,rtol=1e-6,atol=1e-7)
                for k,value in candidates.items():
                    np.testing.assert_allclose(event['candidates'][k][row].cpu(),value,rtol=1e-6,atol=1e-7)
                if quantity is not None:
                    j=event['qr'].tolist().index(row)
                    np.testing.assert_allclose(event['qfeatures'][j].cpu(),quantity[0],rtol=1e-6,atol=1e-7)
                    np.testing.assert_array_equal(event['qmask'][j].cpu(),quantity[1])

if __name__=='__main__':unittest.main()
