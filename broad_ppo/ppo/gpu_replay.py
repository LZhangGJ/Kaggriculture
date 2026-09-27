"""Build recurrent PPO minibatches directly from compact GPU rollouts."""
import torch
from ppo.gpu_features import GPUFeatures,worker_seed
from ppo.gpu_post_worker import GPUPostWorker
from ppo.worker_state import WorkerState


@torch.no_grad()
def batch_window(data,begin,end,games,wq,mq,outcomes,advantages=None,burn=0,seat_ids=None,wires=None,features=True,skip_burn_events=False):
    # v40 skip_burn_events: burn-in turns (loss-masked) get no decoder events; they only advance the recurrent state
    # distill-v1: wires=(workers,market) [steps*b,W,5] replaces the recorded actions; features=False skips the encoder inputs
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
    h=[data['history/'+k][begin:end][:,all_seats].flatten(0,1) for k in ('rows','valid','ema','variance','count','long') if 'history/'+k in data]
    x=None
    if features:
        x={k:v[selected] for k,v in GPUFeatures(device)(state,*h).items()}
        # Valid workers form a contiguous prefix (own workers, then opponents).
        # Masked padding cannot influence valid tokens. Keep the same representation
        # while avoiding attention/MLP work over unused worker slots during PPO.
        token_width=max(1,int(x['worker_valid'].sum(-1).max().item()))
        for key in ('workers','worker_valid'):x[key]=x[key][:,:token_width]
    seed={k:v[selected] for k,v in worker_seed(state).items()}
    width=int(seed['count'].max().item())
    for k in ('positions','inventory','order'):seed[k]=seed[k][:,:width]
    wires=[data[k][begin:end].permute(0,2,1,3)[:,seats].flatten(0,1).long() for k in ('workers','market')] if wires is None else list(wires)
    resolved=WorkerState(seed)
    resolved.resolve(list(wires[0][:,:width,0].unbind(1)),list(wires[0][:,:width,1].unbind(1)))
    market,ledger,farm=GPUPostWorker(device)(resolved,state,selected)
    events=[]
    scored=(torch.arange(steps*b,device=device)>=burn*b) if skip_burn_events and burn else None  # time-major rows
    for phase,wire in enumerate(wires):
        for depth in range(width if phase==0 else 10):
            chosen=wire[:,depth]
            active=chosen[:,4].bool()
            if scored is not None:active=active&scored
            ids=active.nonzero().flatten()
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
def terminal_advantages(data,cash,lam=.95,beta=1.,sigma=25000.,cash_weight=0.,cash_center=100000.,shape='clip',dense=False,potential='cash',potential_weight=None):
    """Returns (outcome, advantages [T,N], total reward [N], reward-to-go returns [T,N]).

    networth-shaping-v2: potential_weight=None keeps exactly the v1 behaviour below. A number W_p scales ONLY the per-turn
    potential: Phi_t = W_p*squash((L_t-centre)/sigma) for t < T, Phi_T = cash_weight*squash((final_cash-centre)/sigma), so
      sum_t r_t = margin_terms(margin) + Phi_T - W_p*squash((L_0-centre)/sigma)   (L_0 = 3000 for every seat: a constant);
    the terminal objective is unchanged and the last-turn reward absorbs Phi_T - Phi_{T-1}.

    networth-shaping-v1: with dense=True, `potential` picks the per-turn level L_t fed to Phi: 'cash' = own money
    (shaped-reward-v6/v7, unchanged), 'networth'/'networth-growing' = ppo.networth mark-to-market worth. The final
    potential is ALWAYS Phi(terminal cash) (official score = money only), so per seat
      sum_t r_t = margin_terms(margin) + Phi(final_cash) - Phi(L_0),
    the same total as the cash potential except the constant Phi(L_0) (L_0 = 3000 for every potential at turn 0)."""
    from ppo.replay import shaped_reward,dense_rewards,reward_to_go
    margin=cash-cash.flip(1)
    outcome=(margin.sign()+1).reshape(-1).long()
    values=data['value']
    if dense:
        if potential=='cash':money=data['state/money'].reshape(len(values),-1)
        else:
            from ppo.networth import potential_level
            money=potential_level(data,potential,len(values))
        if potential_weight is None:
            rewards=dense_rewards(money,cash.reshape(-1),margin.reshape(-1),beta,sigma,cash_weight,cash_center,shape).to(values.dtype)
        else:
            from ppo.networth import dense_rewards_weighted
            rewards=dense_rewards_weighted(money,cash.reshape(-1),margin.reshape(-1),beta,sigma,cash_weight,cash_center,shape,potential_weight).to(values.dtype)
    else:
        if potential!='cash':raise ValueError('networth-shaping-v1: a non-cash potential requires dense rewards')
        if potential_weight is not None:raise ValueError('networth-shaping-v2: a potential weight requires dense rewards')
        rewards=torch.zeros_like(values)
        rewards[-1]=shaped_reward(margin.reshape(-1),beta,sigma,cash=cash.reshape(-1),cash_weight=cash_weight,cash_center=cash_center,shape=shape).to(values.dtype)
    advantages=torch.empty_like(values)
    carry=torch.zeros_like(rewards[-1]);following=torch.zeros_like(rewards[-1])
    for t in range(len(values)-1,-1,-1):
        delta=rewards[t]+following-values[t]
        carry=delta+lam*carry;advantages[t]=carry;following=values[t]
    return outcome,advantages,rewards.sum(0),reward_to_go(rewards)
