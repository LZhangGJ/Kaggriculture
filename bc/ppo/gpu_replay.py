"""Build recurrent PPO minibatches directly from compact GPU rollouts."""
import torch
from ppo.gpu_features import GPUFeatures,worker_seed
from ppo.gpu_post_worker import GPUPostWorker
from ppo.worker_state import WorkerState


@torch.no_grad()
def batch_window(data,begin,end,games,wq,mq,outcomes,advantages=None,burn=0,seat_ids=None):
    device=data['logp'].device
    if seat_ids is None:
        games=list(games);seat_ids=[2*g+s for g in games for s in (0,1)]
    else:games=sorted({s//2 for s in seat_ids})
    positions=torch.tensor([2*games.index(s//2)+s%2 for s in seat_ids],device=device)
    games=torch.as_tensor(games,device=device,dtype=torch.long)
    seats=torch.as_tensor(seat_ids,device=device,dtype=torch.long)
    all_seats=(games[:,None]*2+torch.arange(2,device=device)).flatten()
    steps=end-begin;b=len(seats)
    state={k[6:]:v[begin:end][:,games].flatten(0,1) for k,v in data.items() if k.startswith('state/')}
    selected=(torch.arange(steps,device=device)[:,None]*len(all_seats)+positions).flatten()
    h=[data['history/'+k][begin:end][:,all_seats].flatten(0,1) for k in ('rows','valid','ema','variance','count')]
    x={k:v[selected] for k,v in GPUFeatures(device)(state,*h).items()}
    # Valid workers form a contiguous prefix (own workers, then opponents).
    # Masked padding cannot influence valid tokens. Keep the same representation
    # while avoiding attention/MLP work over unused worker slots during PPO.
    token_width=max(1,int(x['worker_valid'].sum(-1).max().item()))
    for key in ('workers','worker_valid'):x[key]=x[key][:,:token_width]
    seed={k:v[selected] for k,v in worker_seed(state).items()}
    width=int(seed['count'].max().item())
    for k in ('positions','inventory','order'):seed[k]=seed[k][:,:width]
    wires=[data[k][begin:end].permute(0,2,1,3)[:,seats].flatten(0,1).long() for k in ('workers','market')]
    resolved=WorkerState(seed)
    resolved.resolve(list(wires[0][:,:width,0].unbind(1)),list(wires[0][:,:width,1].unbind(1)))
    market,ledger,farm=GPUPostWorker(device)(resolved,state,selected)
    events=[]
    for phase,wire in enumerate(wires):
        for depth in range(width if phase==0 else 10):
            chosen=wire[:,depth]
            ids=chosen[:,4].bool().nonzero().flatten()
            if not len(ids):continue
            selected=chosen[ids];index=selected[:,0];quantity=selected[:,1]
            qr=selected[:,3].bool().nonzero().flatten()
            advance=(torch.ones_like(index,dtype=torch.bool) if phase==0 else index!=0).nonzero().flatten()
            events.append(dict(phase=phase,depth=depth,ids=ids,index=index,quantity=quantity,
                qr=qr,qindex=selected[qr,2],advance_indices=advance))
    outcome=outcomes[seats]
    mask=(torch.arange(steps,device=device)>=burn)[:,None].expand(-1,b).flatten()
    if 'learner_mask' in data:mask=mask & data['learner_mask'][begin:end][:,seats].flatten()
    batch=dict(batch=b,steps=steps,x=x,worker_seed=seed,market_seed=market,quantity_vocab=(wq,mq),
        post_ledger=ledger,post_farm=farm,events=events,outcome=outcome,outcome_turn=outcome.repeat(steps),
        old_logp=data['logp'][begin:end][:,seats].flatten(),factor_count=data['factors'][begin:end][:,seats].flatten(),loss_mask=mask)
    if advantages is not None:batch['advantages']=advantages[begin:end][:,seats].flatten()
    memory=tuple(data[k][begin,seats] for k in ('actor','critic'))
    return batch,memory


@torch.no_grad()
def terminal_advantages(data,cash,lam=.95,beta=1.,sigma=25000.):
    from ppo.replay import shaped_reward
    margin=cash-cash.flip(1)
    outcome=(margin.sign()+1).reshape(-1).long()
    reward=shaped_reward(margin.reshape(-1),beta,sigma).to(data['value'].dtype)
    values=data['value'];advantages=torch.empty_like(values)
    carry=torch.zeros_like(reward);following=torch.zeros_like(reward)
    for t in range(len(values)-1,-1,-1):
        delta=(reward if t==len(values)-1 else 0)+following-values[t]
        carry=delta+lam*carry;advantages[t]=carry;following=values[t]
    return outcome,advantages,reward
