"""Regenerate market replay features from the exact seed and chosen requests."""
import torch
from ppo.gpu_market import MarketBatch
from ppo.fast_features import QuantityFeatures

@torch.no_grad()
def materialize_market(batch, quantities):
    state=MarketBatch(batch['market_seed'])
    n=state.n;device=state.device
    for event in batch['events']:
        if event['phase']!=1:continue
        ids=event['ids'];index=event['index'];qr=event['qr']
        ledger=state.vector();candidates=state.candidates(ledger)
        event['ledger']=ledger[ids]
        event['candidates']={k:v[ids] for k,v in candidates.items()}
        all_index=torch.zeros(n,device=device,dtype=torch.long)
        all_quantity=torch.zeros_like(all_index);active=torch.zeros(n,device=device,dtype=torch.bool)
        all_index[ids]=index;all_quantity[ids]=event['quantity'];active[ids]=True
        if len(qr):
            stats=state.qstats(all_index)[ids[qr]]
            event['qfeatures'],event['qmask']=quantities(1,index[qr],stats)
        state.apply(all_index,all_quantity,active)
        event['delta']=(state.vector()-ledger)[ids]
    batch['market_materialized']=True
