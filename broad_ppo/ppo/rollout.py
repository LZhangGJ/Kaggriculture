"""Complete-game collection and recurrent windows; no replay JSON hot path."""
from collections import Counter
import time
import numpy as np
import torch
from exact_training import pack_turns, merge_chunks
from ppo.actors import ActorPool
from ppo.sampler import BatchedExactSampler
from ppo.replay import gae, shaped_reward, dense_rewards, reward_to_go


def collect(assignments, models, reference, wq, mq, device, workers, bf16=False, arena_mib=32, burn=16, train=48, sampler=None, gae_lambda=.95, reference_sha=None, all_anchors=False, reward_beta=1., reward_sigma=25000., reward_cash_weight=0., reward_cash_center=100000., reward_shape='clip', reward_dense=False):
    start=time.perf_counter()
    pool=ActorPool(assignments,wq,mq,workers=workers,arena_mib=arena_mib, timeout=600 if any(a.get("arena_opponent") for a in assignments) else 120)
    startup_seconds=time.perf_counter()-start
    sampler=sampler or BatchedExactSampler(models,reference,wq,mq,device,bf16)
    sampler.reset_seasons()
    episodes={}
    active={a['game']:a for a in assignments}
    for a in assignments:
        for seat in a['learner_seats']:
            episodes[f"{a['game']}:{seat}"]=dict(assignment=a,seat=seat,turns=[],reference_sha=reference_sha)
    games=[]
    turn=0
    try:
        while active:
            keys=[f'{game}:{seat}' for game,a in active.items() for seat,p in enumerate(a['policies']) if not p.startswith('script:')]
            starts=pool.call('start',keys)
            policies={}
            for key in keys:
                game,seat=map(int,key.split(':'))
                policies.setdefault(active[game]['policies'][seat],[]).append(key)
            anchor=all_anchors or turn==0 or (turn+burn>=train and (turn+burn)%train==0)
            records=sampler.act_groups(policies,starts,pool,anchor=anchor)
            for key,row in records.items(): episodes[key]['turns'].append(row)
            status=pool.call('step',list(active))
            turn+=1
            for game,result in status.items():
                if 'money' in result:
                    for seat in active[game]['learner_seats']:episodes[f'{game}:{seat}'].setdefault('money',[]).append(result['money'][seat])
                if result['done']:
                    a=active.pop(game)
                    if not any(result['faults']) and (result['cash'] is None or len(result['cash'])!=2 or not np.isfinite(result['cash']).all()):
                        raise FloatingPointError(f'Nonfinite terminal cash: game={game}, result={result}')
                    games.append(dict(**a,**result))
                    for seat in a['learner_seats']:
                        ep=episodes[f'{game}:{seat}']
                        faults=result['faults']
                        if any(faults):
                            # Opponent policy faults do not supply positive learner rewards.
                            if not faults[seat]:
                                ep['discarded']='opponent_fault';continue
                            outcome=0;shaped=-1.
                        else:
                            cash=result['cash'];outcome=1 if cash[seat]==cash[1-seat] else 2 if cash[seat]>cash[1-seat] else 0
                            shaped=float(shaped_reward(cash[seat]-cash[1-seat],reward_beta,reward_sigma,cash=cash[seat],cash_weight=reward_cash_weight,cash_center=reward_cash_center,shape=reward_shape))
                        turns=ep['turns']
                        ep['outcome']=outcome
                        rewards=np.zeros(len(turns),np.float32);rewards[-1]=shaped
                        if reward_dense and not any(faults):
                            money=[3000.]+list(ep.get('money',[]))  # startingMoney 3000; money[t] = own cash before turn t
                            if len(money)!=len(turns)+1:raise RuntimeError(f'Money trace length {len(money)} for {len(turns)} turns')
                            dense=dense_rewards(torch.tensor(money[:-1])[:,None],torch.tensor([money[-1]]),torch.tensor([cash[seat]-cash[1-seat]]),reward_beta,reward_sigma,reward_cash_weight,reward_cash_center,reward_shape)[:,0].numpy()
                            if abs(float(money[-1])-float(cash[seat]))>1e-3:raise RuntimeError('Money trace does not end at terminal cash')
                            rewards=dense.astype(np.float32)
                        ep['shaped_return']=float(rewards.sum())
                        ep['returns']=reward_to_go(torch.tensor(rewards)).numpy()
                        terminated=np.zeros(len(turns),bool);terminated[-1]=True
                        ep['advantages']=gae(rewards,[r['value'] for r in turns],terminated,lam=gae_lambda).numpy()
                        if gae_lambda == 1.0 and not reward_dense:
                            expected = shaped-np.asarray([r['value'] for r in turns])
                            np.testing.assert_allclose(ep['advantages'], expected, atol=2e-5, rtol=0)
        kept=[e for e in episodes.values() if 'discarded' not in e]
        turns=sum(len(e['turns']) for e in kept)
        seconds=time.perf_counter()-start
        valid=sum(not any(g['faults']) for g in games)
        return kept,dict(games=games,complete_games=len(games),valid_games=valid,
                         learner_turns=turns,collection_seconds=seconds,actor_startup_seconds=startup_seconds,games_per_second=valid/seconds,
                         learner_turns_per_second=turns/seconds,sampler_seconds=sampler.seconds,gpu_seconds=sampler.gpu_timings(),
                         action_factors=sampler.factors,ipc_seconds=pool.ipc_seconds,ipc_bytes=pool.bytes,ipc_bytes_by_command=pool.bytes_by_command,ipc_calls_by_command=pool.calls_by_command,ipc_seconds_by_command=pool.seconds_by_command,
                         actor_timings=pool.timings(),
                         full_seasons=sum(g['turns']==719 and not any(g['faults']) for g in games),
                         families=dict(Counter(a['family'] for a in assignments)))
    finally:
        import sys,traceback
        failed=sys.exc_info()[0] is not None
        try:pool.close()
        except BaseException:
            if not failed:raise
            traceback.print_exc()


def windows(episodes,burn=16,train=48):
    out={}
    for ep in episodes:
        start=ep.get('train_from',0)  # start-state-v1: turns before the start turn are forced prefix, never trained
        for first in range(0,len(ep['turns']),train):
            begin=max(0,first-burn);end=min(len(ep['turns']),first+train)
            if start and end<=start:continue
            key=(begin,end-begin,first-begin,'gpu' if 'gpu_data' in ep else 'official')
            out.setdefault(key,[]).append((ep,begin,end,first-begin))
    return out


def batch_windows(rows, teacher=None):
    if len({'gpu_data' in row[0] for row in rows}) != 1:
        raise ValueError('Mixed rollout representations in one minibatch')
    if 'gpu_data' in rows[0][0]:
        from ppo.gpu_backend import batch_gpu_windows
        return batch_gpu_windows(rows, teacher) if teacher else batch_gpu_windows(rows)  # distill-v1
    for ep,begin,_,_ in rows:
        if ep['turns'][begin]['state'] is None or ep['turns'][begin]['reference_state'] is None:
            raise RuntimeError(f'Missing recurrent anchor at turn {begin}')
    chunks=[pack_turns(ep['turns'][begin:end],begin,ep['outcome']) for ep,begin,end,_ in rows]
    batch=merge_chunks(chunks)
    references={ep.get("reference_sha") for ep,_,_,_ in rows}
    if len(references)!=1:raise ValueError("Mixed reference identities")
    batch["reference_sha"]=references.pop()
    if 'market_seed' in rows[0][0]['turns'][rows[0][1]]:
        seats=[ep['turns'][begin:end] for ep,begin,end,_ in rows]
        turns=[seat[t] for t in range(len(seats[0])) for seat in seats]
        batch['market_seed']={k:torch.from_numpy(np.stack([turn['market_seed'][k] for turn in turns])) for k in turns[0]['market_seed']}
        batch['quantity_vocab']=turns[0]['quantity_vocab']
        if 'worker_seed' in turns[0]:
            from ppo.transport import BatchTransfer
            batch['worker_seed']=BatchTransfer('cpu').mapping([t['worker_seed'] for t in turns],pad_names=('positions','inventory','order'))

    # Loss order follows time-major merging, never concatenate whole episodes.
    batch['old_logp']=torch.tensor(np.stack([[r['old_logp'] for r in ep['turns'][begin:end]] for ep,begin,end,_ in rows],1).reshape(-1),dtype=torch.float32)
    batch['advantages']=torch.tensor(np.stack([ep['advantages'][begin:end] for ep,begin,end,_ in rows],1).reshape(-1),dtype=torch.float32)
    batch['outcome_turn']=batch['outcome'].repeat(batch['steps'])
    batch['return_turn']=(torch.tensor(np.stack([ep['returns'][begin:end] for ep,begin,end,_ in rows],1).reshape(-1),dtype=torch.float32) if all('returns' in ep for ep,_,_,_ in rows)
        else torch.tensor([ep['shaped_return'] for ep,_,_,_ in rows],dtype=torch.float32).repeat(batch['steps']))
    batch['factor_count']=torch.tensor(np.stack([[r['factor_count'] for r in ep['turns'][begin:end]] for ep,begin,end,_ in rows],1).reshape(-1))
    batch['loss_mask']=torch.tensor(np.stack([np.arange(end-begin)>=burn for _,begin,end,burn in rows],1).reshape(-1))
    state=torch.tensor(np.stack([ep['turns'][begin]['state'] for ep,begin,_,_ in rows]))
    ref=torch.tensor(np.stack([ep['turns'][begin]['reference_state'] for ep,begin,_,_ in rows]))
    return batch,tuple(state[:,i] for i in (0,1)),tuple(ref[:,i] for i in (0,1))
