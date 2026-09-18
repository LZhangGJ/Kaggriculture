"""One causal request builder for replay supervision and live decoding."""
from collections import Counter
import numpy as np
import torch
from bc_runtime import MARKET
from features_v2 import PublicHistory, RequestLedger, candidates, quantity_features, batch_arrays
from exact_features import encode_exact
from exact_actions import (WORKER, worker_target, worker_wire, worker_features,
    worker_quantity_features, needs_quantity, market_target, expand_worker)
from worker_phase import command_slots, resolve_worker_phase


def ledger_vector(ledger):
    value=ledger.vector();value[4]=0
    return value


def market_state(resolved):
    ledger=RequestLedger(resolved,[])
    ledger.worker_projected=True;ledger.projection_exact=True
    return ledger


def market_candidates(ledger):
    c=candidates(ledger,None,1)
    c['admissible'][1]=False  # NOOP is not a sampled market request.
    c['raw'][:,76]=0  # No obsolete pending-program capacity.
    refs=c['refs']
    return dict(raw=c['raw'],source=refs[:,0],target=refs[:,0],items=refs[:,1],
                workers=refs[:,2],legal=c['admissible'])


def worker_delta(index,quantity):
    # Request feedback, not a claim about final physical effects.
    delta=np.zeros(128,np.float32);delta[index]=1
    delta[44]=np.sign(quantity)*np.log1p(abs(quantity))/12
    return delta


def remember_requests(history,requests):
    history.requests=[]
    for request in requests:
        index,q,valid=market_target(request)
        if valid:
            op,item=MARKET[index]
            history.requests.append([op] if item is None else [op,item,int(q)])


def prepare_turn(turn,history,worker_quantities,market_quantities):
    obs=turn['observation'];history.observe(obs);row=encode_exact(obs,history)
    raw=command_slots(turn['worker_action']);n=1+len(obs['farms'][obs['player']]['hands'])
    prefix=[];canonical=[];events=[];stats=Counter();valid_prefix=True
    wi={q:i for i,q in enumerate(worker_quantities)};mi={q:i for i,q in enumerate(market_quantities)}
    for depth in range(n):
        provisional,_,_=resolve_worker_phase(obs,prefix)
        action=raw[depth] if depth<len(raw) else ['PASS']
        index,q,valid,has_q=worker_target(action,provisional,depth)
        if has_q and q not in wi:raise ValueError('Worker quantity missing from audited vocabulary')
        valid_prefix=valid_prefix and valid
        compact=worker_features(obs,provisional,prefix,depth)
        event=dict(phase=0,depth=depth,compact=compact,ledger=compact['ledger'],index=index,
                   quantity=q,supervised=valid_prefix,delta=worker_delta(index,q),advance=True)
        if has_q:
            qf,qm=worker_quantity_features(provisional,depth,index,worker_quantities)
            event.update(qfeatures=qf,qmask=qm,qindex=wi[q])
        events.append(event);prefix.append(action)
        canonical.append(worker_wire(index,q if has_q else None))
        stats['worker_slots']+=1;stats['worker_unsupported']+=not valid
    resolved,_,_=resolve_worker_phase(obs,raw)
    canonical_resolved,_,_=resolve_worker_phase(obs,canonical)
    # Extra raw hands can alter the atomic PLANT prepass. Do not teach a
    # canonical sequence whose execution differs from the recorded sequence.
    equivalent=(resolved['private']==canonical_resolved['private'] and
                resolved['farms'][obs['player']]==canonical_resolved['farms'][obs['player']])
    if not equivalent:
        stats['worker_transition_mismatch_turns']+=1
        for event in events:event['supervised']=False
    ledger=market_state(resolved);post_ledger=ledger_vector(ledger)
    # Encoding here runs on CPU only; the neural board encoder still runs once.
    post_farm=encode_exact(resolved,history)['farms'][0]
    valid_prefix=True;requests=turn['market_requests']
    for depth in range(min(10,len(requests)+1)):
        end=depth==len(requests)
        index,q,valid=(0,0,True) if end else market_target(requests[depth])
        if valid and MARKET[index][1] is not None and q not in mi:
            raise ValueError('Market quantity missing from audited vocabulary')
        valid_prefix=valid_prefix and valid
        before=ledger_vector(ledger)
        event=dict(phase=1,depth=depth,candidates=market_candidates(ledger),ledger=before,
                   index=index,quantity=q,supervised=valid_prefix,advance=not end)
        if valid and MARKET[index][1] is not None:
            qf,qm=quantity_features(ledger,index,market_quantities)
            event.update(qfeatures=qf,qmask=qm,qindex=mi[q])
        if not end:
            if valid:ledger.add_order(index,q)
            else:ledger.orders.append([])
        event['delta']=ledger_vector(ledger)-before;events.append(event)
        stats['market_slots']+=not end;stats['market_unsupported']+=not valid
        if end:break
    for phase in (0,1):
        es=[e for e in events if e['phase']==phase and e['supervised']]
        denominators=[len(es),sum(e['index']!=0 for e in es),sum('qindex' in e for e in es)]
        for e in (e for e in events if e['phase']==phase):
            active=[e['supervised'],e['supervised'] and e['index']!=0,e['supervised'] and 'qindex' in e]
            e['weights']=np.array([float(a)/max(1,d) for a,d in zip(active,denominators)],np.float32)
        for name,count in zip(('gate','command','quantity'),denominators):stats[f'{phase}_{name}']+=count
    remember_requests(history,requests)
    return dict(x=row,events=events,post_ledger=post_ledger,post_farm=post_farm,stats=dict(stats))


def score_event(model,phase,actor,prefix,context,candidate,ledger,extra):
    head=model.worker_head if phase==0 else model.market_head
    return head.score(actor,prefix,context,candidate['raw'],candidate['source'],candidate['target'],
                      candidate['items'],candidate['workers'],ledger,extra,candidate['legal'])


class ExactAgent:
    """Season-recurrent policy; teacher and live use identical candidate builders."""
    def __init__(self,model,worker_quantities,market_quantities,device='cpu',greedy=True):
        self.model=model.to(device).eval();self.wq=worker_quantities;self.mq=market_quantities
        self.device=device;self.greedy=greedy;self.reset()

    def reset(self):
        self.history=PublicHistory();self.state=tuple(torch.zeros(1,256,device=self.device) for _ in range(2))

    @classmethod
    def from_checkpoint(cls,path,device='cpu',greedy=True):
        from exact_identity import validate_identity
        from exact_model import ExactWorkerMarketPolicyV1
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        ci=checkpoint['cache_identity'];validate_identity(ci)
        if (checkpoint['worker_quantities']!=ci['worker_quantities'] or
                checkpoint['market_quantities']!=ci['market_quantities']):
            raise ValueError('Checkpoint quantity vocabulary mismatch')
        model=ExactWorkerMarketPolicyV1();model.load_exact(checkpoint)
        return cls(model,ci['worker_quantities'],ci['market_quantities'],device,greedy)

    def choose(self,lp):
        return int(lp.argmax(-1).item()) if self.greedy else int(torch.distributions.Categorical(logits=lp).sample().item())

    def choose_action(self,joint,gates,commands):
        if not self.greedy:return self.choose(joint)
        return 0 if gates[0,0]>=gates[0,1] else int(commands[0].argmax().item())+1

    @torch.no_grad()
    def act(self,obs):
        if obs['step']==0:self.reset()
        self.history.observe(obs);row=encode_exact(obs,self.history);m=self.model
        actor,critic,_,ctx=m.encode(batch_arrays([row],self.device),*self.state)
        self.state=(actor,critic);prefix=torch.zeros_like(actor);slots=[];density=0.
        def tensor(a):return torch.as_tensor(a,device=self.device).unsqueeze(0)
        n=1+len(obs['farms'][obs['player']]['hands'])
        for depth in range(n):
            provisional,_,_=resolve_worker_phase(obs,slots)
            data=worker_features(obs,provisional,slots,depth)
            c=expand_worker({k:tensor(v) for k,v in data.items()});led=tensor(data['ledger'])
            extra=ctx['workers'][:,depth]
            lp,gates,commands,emb=score_event(m,0,actor,prefix,ctx,c,led,extra)
            index=self.choose_action(lp,gates,commands)
            density+=float(lp[0,index]);q=0;has_q=needs_quantity(provisional,depth,index)
            if has_q:
                qf,qm=worker_quantity_features(provisional,depth,index,self.wq)
                qlp=m.worker_head.quantities(actor,prefix,emb[:,index],led,tensor(qf),tensor(qm))
                qi=self.choose(qlp);q=self.wq[qi];density+=float(qlp[0,qi])
            slots.append(worker_wire(index,q if has_q else None))
            prefix=m.worker_head.advance(prefix,emb[:,index],tensor(np.float32(q)).reshape(1),tensor(worker_delta(index,q)))
        resolved,_,_=resolve_worker_phase(obs,slots);ledger=market_state(resolved)
        extra=m.summarize_post_worker(tensor(ledger_vector(ledger)),tensor(encode_exact(resolved,self.history)['farms'][0]))
        prefix=torch.zeros_like(actor)
        for depth in range(10):
            c={k:tensor(v) for k,v in market_candidates(ledger).items()};before=ledger_vector(ledger);led=tensor(before)
            lp,gates,commands,emb=score_event(m,1,actor,prefix,ctx,c,led,extra)
            index=self.choose_action(lp,gates,commands)
            density+=float(lp[0,index])
            if index==0:break
            q=0
            if MARKET[index][1] is not None:
                qf,qm=quantity_features(ledger,index,self.mq)
                qlp=m.market_head.quantities(actor,prefix,emb[:,index],led,tensor(qf),tensor(qm))
                qi=self.choose(qlp);q=self.mq[qi];density+=float(qlp[0,qi])
            ledger.add_order(index,q)
            prefix=m.market_head.advance(prefix,emb[:,index],tensor(np.float32(q)).reshape(1),tensor(ledger_vector(ledger)-before))
        remember_requests(self.history,ledger.orders)
        return dict(farmer=slots[0],hands=slots[1:],market=ledger.orders),density
