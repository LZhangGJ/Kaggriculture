import copy,unittest
import numpy as np
import torch
from ppo.gpu_market import MarketBatch,market_seed
from ppo.evaluate import LocalBackend
from ppo.transport import BatchTransfer
from exact_decoder import market_state,ledger_vector,market_candidates
from features_v2 import quantity_features
from ppo.fast_features import QuantityFeatures

class MarketChecks(unittest.TestCase):
    def test_request_sequences(self):
        torch.set_num_threads(1)
        b=LocalBackend([dict(game=0,seed=42,policies=['script:starter']*2,learner_seats=[],family='test')],[1],[1])
        obs=dict(b.games.games[0]['env'].state[0].observation)
        ledgers=[]
        for n in range(4):
            o=copy.deepcopy(obs);o['farms'][0]['money']=n*901;o['farms'][0]['hires_today']=n;o['private']['shed']['WHEAT']=n*51;o['market']['inventory']['WHEAT']=-100*n
            ledgers.append(market_state(o))
        device=__import__('os').environ.get('PARITY_DEVICE','cpu')
        batch=MarketBatch(BatchTransfer(device).mapping([market_seed(l) for l in ledgers]))
        actions=[[(2,0),(2,0),(3,0),(3,0),(3,0),(3,0),(4,200),(14,200),(9,40),(0,0)],[(4,99),(4,3),(15,1),(9,900),(3,0),(2,0),(10,2),(21,1),(22,99),(9,40)],[(0,0)]*10,[(22,10),(13,2),(2,0),(4,1),(3,0),(3,0),(3,0),(3,0),(3,0),(2,0)]]
        active=np.ones(4,bool)
        for depth in range(10):
            vector=batch.vector();cand=batch.candidates(vector)
            for row,l in enumerate(ledgers):
                np.testing.assert_allclose(vector[row].cpu().numpy(),ledger_vector(l),atol=1e-7,rtol=1e-7)
                ref=market_candidates(l)
                for k in ref: np.testing.assert_allclose(cand[k][row].cpu().numpy(),ref[k],atol=1e-7,rtol=1e-7)
            ix=torch.tensor([a[depth][0] for a in actions],device=device);q=torch.tensor([a[depth][1] for a in actions],device=device)
            batch.apply(ix,q,torch.tensor(active,device=device))
            for row,l in enumerate(ledgers):
                index,qty=actions[row][depth]
                if active[row] and index:l.add_order(index,qty)
                if index==0: active[row]=False
        for row,l in enumerate(ledgers):np.testing.assert_allclose(batch.vector()[row].cpu().numpy(),ledger_vector(l),atol=1e-7,rtol=1e-7)
if __name__=='__main__':unittest.main()
