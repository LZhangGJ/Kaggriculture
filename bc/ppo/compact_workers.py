"""Regenerate exact worker features from the turn seed and recorded requests."""
import torch
from ppo.worker_state import WorkerState
from ppo.worker_features_device import WorkerFeatures

@torch.no_grad()
def materialize_workers(batch, quantities):
    state=WorkerState(batch['worker_seed']);features=WorkerFeatures(state)
    indices=[];amounts=[];n=state.n;device=state.device
    counts=torch.zeros(n,5,device=device,dtype=torch.long)
    for event in batch['events']:
        if event['phase']!=0:continue
        ids=event['ids'];index=event['index'];qr=event['qr'];depth=event['depth']
        state.resolve(indices,amounts)
        compact,need,stats=features(state.workers[depth].expand(n),counts)
        event['compact']={k:v[ids] for k,v in compact.items()}
        event['ledger']=event['compact']['ledger']
        delta=torch.zeros(len(ids),128,device=device)
        delta.scatter_(1,index[:,None],1)
        delta[:,44]=quantities.scaled(event['quantity'].double())
        event['delta']=delta
        if len(qr):
            event['qfeatures'],event['qmask']=quantities(0,index[qr],stats[ids[qr],index[qr]])
        all_index=torch.zeros(n,device=device,dtype=torch.long);all_quantity=torch.zeros_like(all_index)
        all_index[ids]=index;all_quantity[ids]=event['quantity']
        indices.append(all_index);amounts.append(all_quantity)
        counts.scatter_add_(1,(all_index-15).clamp(0,4)[:,None],((all_index>=15)&(all_index<20)).long()[:,None])
    batch['workers_materialized']=True
