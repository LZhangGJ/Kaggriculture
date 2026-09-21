"""Adapter from the existing league/trainer to compact GPU trajectories."""
import time
import json
import os
from collections import Counter
import torch
from ppo.gpu_collect import GPUCollector
from ppo.gpu_events import prepare_events
from ppo.gpu_roles import NeuralRole
from ppo.gpu_replay import terminal_advantages,batch_window


class GPUBackend:
    def __init__(self,models,reference,wq,mq,device,workers):
        self.models=models;self.reference=reference;self.wq=wq;self.mq=mq
        self.device=device;self.workers=workers;self.collector=None;self.rollout=None
        self.role_cache={}
        self.reference_sha=None
        self.gae_lambda=.95
        self.reward_beta=1.;self.reward_sigma=25000.
        self.comparison_recorded=False

    def collect(self,jobs):
        start=time.perf_counter();policies={};learner=[]
        for game,job in enumerate(jobs):
            for seat,policy in enumerate(job['policies']):
                if policy not in self.models:
                    raise ValueError(f'GPU collection requires a batched neural opponent: {policy}; use the official collector for Python agents')
                policies.setdefault(policy,[]).append(2*game+seat)
            learner.extend(2*game+s for s in job['learner_seats'])
        seeds=[j['seed'] for j in jobs]
        policies=dict(sorted(policies.items()))
        if any(j.get('reference_sha256')!=self.reference_sha for j in jobs):
            raise ValueError('Collection reference mismatch')
        events=prepare_events(seeds,workers=self.workers)
        prep=time.perf_counter()-start
        if self.collector is None:
            roles=[(self.models[p],seats) for p,seats in policies.items()]
            self.collector=GPUCollector(self.models['learner'],self.wq,self.mq,seeds,str(self.device),
                events=events,reference=self.reference,roles=roles,learner_seats=learner)
            self.role_cache=dict(zip(policies,self.collector.roles))
        else:
            self.collector.reset(seeds,events)
            roles=[]
            for policy,seats in policies.items():
                role=self.role_cache.get(policy)
                if role is None or len(role.seats)!=len(seats):
                    role=NeuralRole(self.models[policy],seats,self.collector.quantities,self.device,True)
                    self.role_cache[policy]=role
                else:
                    if role.model is not self.models[policy]:raise RuntimeError('Cached role model was replaced')
                    role.seats.copy_(torch.tensor(seats,device=self.device))
                roles.append(role)
            self.collector.roles=roles
            for stale in set(self.role_cache)-set(policies):del self.role_cache[stale]
            self.collector.learner_mask.zero_();self.collector.learner_mask[learner]=True
            self.collector.learner_ids=torch.tensor(learner,device=self.device,dtype=torch.long)
        def progress(turn):
            print(json.dumps(dict(stage='gpu_collection',rank=int(os.environ.get('RANK',0)),
                turn=turn,games=len(jobs),seconds=time.perf_counter()-start)),flush=True)
        self.rollout,final=self.collector.collect(rollout=self.rollout,progress=progress)
        outcome,adv,reward=terminal_advantages(self.rollout.data,final['money'],lam=self.gae_lambda,beta=self.reward_beta,sigma=self.reward_sigma)
        comparison=None
        if not self.comparison_recorded:
            _,comparison,_=terminal_advantages(self.rollout.data,final['money'],lam=.95,beta=self.reward_beta,sigma=self.reward_sigma)
            expected=reward.reshape(1,-1)-self.rollout.data['value']
            if self.gae_lambda==1.0 and not torch.allclose(adv,expected,atol=2e-5,rtol=2e-5):raise RuntimeError('Lambda1 terminal-return identity failed')
            self.comparison_recorded=True
        cash=final['money'].cpu().tolist();games=[];episodes=[]
        for game,job in enumerate(jobs):
            games.append(dict(**job,cash=cash[game],done=True,turns=719,faults=[False,False]))
            for seat in job['learner_seats']:
                flat=2*game+seat
                episodes.append(dict(gpu_data=self.rollout.data,gpu_outcomes=outcome,seat_index=flat,
                    reference_sha=self.reference_sha,
                    quantity_vocab=(self.wq,self.mq),turns=range(719),advantages=adv[:,flat],
                    assignment=job,seat=seat,outcome=int(outcome[flat]),shaped_return=float(reward[flat])))
        if comparison is not None:
            for ep in episodes:ep['comparison_advantages']=comparison[:,ep['seat_index']]
        seconds=time.perf_counter()-start
        return episodes,dict(games=games,complete_games=len(games),valid_games=len(games),full_seasons=len(games),
            learner_turns=len(learner)*719,collection_seconds=seconds,event_preparation_seconds=prep,
            games_per_second=len(games)/seconds,gpu_seconds=self.collector.profile(),
            worker_bucket_turns=dict(self.collector.worker_bucket_turns),
            actor_batches={p:len(s) for p,s in policies.items()},
            reference_graphs=len(self.collector.reference_graphs),
            caches={p:dict(encoders=len(r.encoders),workers=len(r.workers),market=int(r.market is not None)) for p,r in self.role_cache.items()},
            memory=dict(allocated=torch.cuda.memory_allocated(self.device),reserved=torch.cuda.memory_reserved(self.device),free=torch.cuda.mem_get_info(self.device)[0]),
            rollout_bytes=self.rollout.bytes,families=dict(Counter(j['family'] for j in jobs)))


def batch_gpu_windows(rows):
    first,begin,end,burn=rows[0];data=first['gpu_data']
    if any(ep['gpu_data'] is not data or (b,e,k)!=(begin,end,burn) for ep,b,e,k in rows):
        raise ValueError('GPU recurrent minibatch must share rollout and window')
    seats=[ep['seat_index'] for ep,_,_,_ in rows]
    batch,memory=batch_window(data,begin,end,[],*first['quantity_vocab'],first['gpu_outcomes'],burn=burn,seat_ids=seats)
    batch['advantages']=torch.stack([ep['advantages'][begin:end] for ep,_,_,_ in rows],1).flatten()
    batch['return_turn']=torch.tensor([ep['shaped_return'] for ep,_,_,_ in rows],dtype=torch.float32,device=batch['advantages'].device).repeat(end-begin)
    if any(ep.get('reference_sha')!=first.get('reference_sha') for ep,_,_,_ in rows):raise ValueError('Mixed reference versions')
    batch['reference_sha']=first.get('reference_sha')
    reference=tuple(data[k][begin,seats] for k in ('reference_actor','reference_critic'))
    return batch,memory,reference
