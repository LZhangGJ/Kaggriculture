"""Regenerate exact worker features from the turn seed and recorded requests."""
import os
import torch
from ppo.worker_state import WorkerState
from ppo.worker_features_device import WorkerFeatures

INCREMENTAL = os.environ.get('PPO_INCREMENTAL_RESOLVE', '1') == '1'


@torch.no_grad()
def materialize_workers(batch, quantities):
    if INCREMENTAL:
        return materialize_workers_incremental(batch, quantities)
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


@torch.no_grad()
def materialize_workers_incremental(batch, quantities):
    """v40 incremental-resolve: same features as the loop above without re-resolving every prefix.

    The old loop called state.resolve(prefix) before every depth: reset + one WorkerState.apply per earlier worker, so
    W workers cost W(W-1)/2 applies. resolve() cancels a PLANT when the prefix's plant count for that crop exceeds the
    seeds, so an earlier worker's effect only changes when a row's blocked-crop set changes (counts only grow). Apply
    the newest worker on top of the previous prefix state and fall back to the full resolve() exactly at the depths
    where any row's blocked set changes (computed up front with a single host read), as fast_features.WorkerPrefix
    does on CPU. The state is integer-valued, so the features are bitwise identical."""
    state=WorkerState(batch['worker_seed']);features=WorkerFeatures(state,lean=True)
    n=state.n;device=state.device;rows=state.rows
    events=[e for e in batch['events'] if e['phase']==0]
    alls=[]
    for event in events:
        all_index=torch.zeros(n,device=device,dtype=torch.long);all_quantity=torch.zeros_like(all_index)
        all_index[event['ids']]=event['index'];all_quantity[event['ids']]=event['quantity']
        alls.append((all_index,all_quantity))
    blocked=[];changed=[]
    if events:
        index=torch.stack([a for a,_ in alls])  # [D,n]
        plant=((index>=15)&(index<20)).long();crop=(index-15).clamp(0,4)
        prefix_counts=torch.zeros(len(events),n,5,device=device,dtype=torch.long).scatter_add_(2,crop[...,None],plant[...,None]).cumsum(0)
        after=prefix_counts>state.seed['seeds'][None]       # blocked set after including depth d
        before=torch.cat((torch.zeros_like(after[:1]),after[:-1]),0)
        changed=(after!=before).flatten(1).any(1).tolist()   # the one host read
        blocked=after.unbind(0)
    indices=[];amounts=[]
    counts=torch.zeros(n,5,device=device,dtype=torch.long)
    for d,event in enumerate(events):
        ids=event['ids'];index=event['index'];qr=event['qr'];depth=event['depth']
        # state == resolve(indices, amounts) here
        compact,need,stats=features(state.workers[depth].expand(n),counts)
        event['compact']={k:v[ids] for k,v in compact.items()}
        event['ledger']=event['compact']['ledger']
        delta=torch.zeros(len(ids),128,device=device)
        delta.scatter_(1,index[:,None],1)
        delta[:,44]=quantities.scaled(event['quantity'].double())
        event['delta']=delta
        if len(qr):
            event['qfeatures'],event['qmask']=quantities(0,index[qr],stats[ids[qr],index[qr]])
        all_index,all_quantity=alls[d]
        indices.append(all_index);amounts.append(all_quantity)
        counts.scatter_add_(1,(all_index-15).clamp(0,4)[:,None],((all_index>=15)&(all_index<20)).long()[:,None])
        if d+1<len(events):  # advance the state to the prefix including this worker (only needed for a later depth)
            if changed[d]:
                state.resolve(indices,amounts)
            else:
                plant=(all_index>=15)&(all_index<20)
                cancelled=plant&blocked[d][rows,(all_index-15).clamp(0,4)]
                state.apply(state.workers[len(indices)-1].expand(n),torch.where(cancelled,0,all_index),all_quantity)
    batch['workers_materialized']=True
