"""Time-batched recurrent BC with per-turn, per-component loss means."""
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from features_v2 import batch_arrays
from exact_actions import expand_worker
from exact_decoder import score_event


def pack_turns(turns,begin,outcome):
    groups={};stats=Counter()
    for i,turn in enumerate(turns):
        stats.update(turn['stats'])
        for event in turn['events']:groups.setdefault((event['phase'],event['depth']),[]).append((i,event))
    events=[]
    for (phase,depth),pairs in sorted(groups.items()):
        ids,rows=zip(*pairs);e=dict(phase=phase,depth=depth,ids=np.asarray(ids,np.int64))
        for key in ('ledger','index','quantity','supervised','delta','weights'):
            e[key]=np.asarray([r[key] for r in rows])
        branch='compact' if phase==0 else 'candidates'
        e[branch]={k:np.stack([r[branch][k] for r in rows]) for k in rows[0][branch]}
        e['advance_indices']=np.asarray([j for j,r in enumerate(rows) if r['advance']],np.int64)
        e['qr']=np.asarray([j for j,r in enumerate(rows) if 'qindex' in r],np.int64)
        if len(e['qr']):
            for key in ('qfeatures','qmask','qindex'):e[key]=np.asarray([rows[j][key] for j in e['qr']])
        events.append(e)
    return dict(x={k:v.numpy() for k,v in batch_arrays([t['x'] for t in turns],'cpu').items()},
        events=events,post_ledger=np.stack([t['post_ledger'] for t in turns]),
        post_farm=np.stack([t['post_farm'] for t in turns]),stats=dict(stats),begin=begin,
        steps=len(turns),last=begin+len(turns)==719,outcome=np.asarray([outcome],np.int64))


def merge_chunks(chunks):
    b=len(chunks);steps=chunks[0]['steps'];x={};groups={};stats=Counter()
    for key in chunks[0]['x']:
        vals=[c['x'][key] for c in chunks]
        if key in ('workers','worker_valid'):
            w=max(v.shape[1] for v in vals)
            vals=[np.pad(v,[(0,0),(0,w-v.shape[1])]+[(0,0)]*(v.ndim-2)) for v in vals]
        x[key]=torch.from_numpy(np.stack(vals,1).reshape(steps*b,*vals[0].shape[1:]))
    for i,c in enumerate(chunks):
        if (c['begin'],c['steps'])!=(chunks[0]['begin'],steps):raise ValueError('Misaligned recurrent chunks')
        stats.update(c['stats'])
        for e in c['events']:groups.setdefault((e['phase'],e['depth']),[]).append((i,e))
    events=[]
    for (phase,depth),pairs in sorted(groups.items()):
        e=dict(phase=phase,depth=depth,ids=torch.from_numpy(np.concatenate([r['ids']*b+i for i,r in pairs])))
        for key in ('ledger','index','quantity','supervised','delta','weights'):
            e[key]=torch.from_numpy(np.concatenate([r[key] for _,r in pairs]))
        branch='compact' if phase==0 else 'candidates'
        e[branch]={k:torch.from_numpy(np.concatenate([r[branch][k] for _,r in pairs])) for k in pairs[0][1][branch]}
        offset=0;qr=[];adv=[]
        for _,r in pairs:
            qr.extend(r['qr']+offset);adv.extend(r['advance_indices']+offset);offset+=len(r['ids'])
        e['qr']=torch.tensor(qr,dtype=torch.long);e['advance_indices']=torch.tensor(adv,dtype=torch.long)
        if qr:
            for key in ('qfeatures','qmask','qindex'):
                e[key]=torch.from_numpy(np.concatenate([r[key] for _,r in pairs if len(r['qr'])]))
            if not e['qmask'].any(-1).all():raise ValueError('Empty quantity grammar')
        events.append(e)
    result=dict(x=x,events=events,stats=dict(stats),batch=b,steps=steps,begin=chunks[0]['begin'],last=chunks[0]['last'],
                outcome=torch.from_numpy(np.concatenate([c['outcome'] for c in chunks])))
    for key in ('post_ledger','post_farm'):
        vals=[c[key] for c in chunks];result[key]=torch.from_numpy(np.stack(vals,1).reshape(steps*b,-1))
    return result


def to_device(v,device):
    if isinstance(v,torch.Tensor):return v.to(device,non_blocking=True)
    if isinstance(v,dict):return {k:to_device(x,device) for k,x in v.items()}
    if isinstance(v,list):return [to_device(x,device) for x in v]
    return v


class ExactChunk(nn.Module):
    def __init__(self,model):super().__init__();self.model=model

    def forward(self,batch,state):
        m=self.model;b=batch['batch'];steps=batch['steps'];token,context=m.encode_features(batch['x'])
        token=token.reshape(steps,b,-1);ah,ch=state;actors=[];critics=[]
        for t in range(steps):
            ah=m.actor_memory(token[t],ah);ch=m.critic_memory(token[t],ch);actors.append(ah);critics.append(ch)
        actor=torch.stack(actors).flatten(0,1);critic=torch.stack(critics).flatten(0,1)
        vl=F.cross_entropy(m.value(critic).float(),batch['outcome'].repeat(steps),reduction='sum')
        post=m.summarize_post_worker(batch['post_ledger'],batch['post_farm'])
        prefix=torch.zeros_like(actor);components=torch.zeros(6,device=actor.device);correct=torch.zeros(6,device=actor.device)
        density=torch.zeros(len(actor),device=actor.device);market=False
        for e in batch['events']:
            phase=e['phase'];head=m.worker_head if phase==0 else m.market_head
            if phase==1 and not market:prefix=torch.zeros_like(prefix);market=True
            ids=e['ids'];idx=e['index'];r=torch.arange(len(ids),device=ids.device)
            ctx={k:v[ids] for k,v in context.items()}
            c=expand_worker(e['compact']) if phase==0 else e['candidates']
            extra=ctx['workers'][r,e['compact']['worker']] if phase==0 else post[ids]
            lp,gates,commands,emb=score_event(m,phase,actor[ids],prefix[ids],ctx,c,e['ledger'],extra)
            gate_target=(idx!=0).long();gate=gates[r,gate_target]
            command=commands[r,(idx-1).clamp_min(0)]
            weights=e['weights'];off=phase*3
            # where BEFORE multiplication avoids 0 * -inf on unsupervised slots.
            components[off]=components[off]-(torch.where(weights[:,0]>0,gate,0)*weights[:,0]).sum()
            components[off+1]=components[off+1]-(torch.where(weights[:,1]>0,command,0)*weights[:,1]).sum()
            correct[off]+=((gates.argmax(-1)==gate_target)&(weights[:,0]>0)).sum()
            correct[off+1]+=((commands.argmax(-1)+1==idx)&(weights[:,1]>0)).sum()
            selected=torch.where(e['supervised'],lp[r,idx],0)
            chosen=emb[r,idx];qr=e['qr']
            if qr.numel():
                qlp=head.quantities(actor[ids[qr]],prefix[ids[qr]],chosen[qr],e['ledger'][qr],e['qfeatures'],e['qmask'])
                qselected=qlp[torch.arange(len(qr),device=ids.device),e['qindex']]
                qw=weights[qr,2];components[off+2]=components[off+2]-(torch.where(qw>0,qselected,0)*qw).sum()
                correct[off+2]+=((qlp.argmax(-1)==e['qindex'])&(qw>0)).sum()
                selected=selected.index_add(0,qr,torch.where(qw>0,qselected,0))
            density=density.index_add(0,ids,selected)
            adv=e['advance_indices']
            if adv.numel():
                nxt=head.advance(prefix[ids[adv]],chosen[adv],e['quantity'][adv],e['delta'][adv])
                prefix=prefix.index_copy(0,ids[adv],nxt.to(prefix.dtype))
        return components.sum()+.05*vl,(ah.detach(),ch.detach()),dict(components=components.detach(),
            value=vl.detach(),correct=correct.detach(),density=density.detach())
