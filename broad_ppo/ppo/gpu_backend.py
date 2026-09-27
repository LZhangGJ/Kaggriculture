"""Adapter from the existing league/trainer to compact GPU trajectories."""
import time
import threading
import json
import os
from collections import Counter
import torch
from ppo.gpu_collect import GPUCollector
from ppo.gpu_events import prepare_events
from ppo.gpu_roles import NeuralRole
from ppo.gpu_replay import terminal_advantages,batch_window


def _bot_ms_by_opponent(arena,game_dt):
    """source-v52 bot-timing-v1: per arena opponent id[:12] -> [mean ms per turn, max ms, games]."""
    agg={}
    for a in arena:
        d=game_dt.get(a['game'])
        if not d:continue
        x=agg.setdefault(a['job']['arena_opponent']['id'][:12],[0.,0.,0,0]);x[0]+=d[0];x[1]=max(x[1],d[1]);x[2]+=d[2];x[3]+=1
    return {k:[round(1000*v[0]/max(v[2],1),2),round(1000*v[1],1),v[3]] for k,v in agg.items()}


class GPUBackend:
    def __init__(self,models,reference,wq,mq,device,workers):
        self.models=models;self.reference=reference;self.wq=wq;self.mq=mq
        self.device=device;self.workers=workers;self.collector=None;self.rollout=None
        self.role_cache={}
        self.reference_sha=None
        self.gae_lambda=.95
        self.reward_beta=1.;self.reward_sigma=25000.
        self.reward_cash_weight=0.;self.reward_cash_center=100000.  # shaped-reward-v5
        self.reward_shape='clip';self.reward_dense=False  # shaped-reward-v6
        self.reward_potential='cash'  # networth-shaping-v1: 'cash' = shaped-reward-v6/v7 potential
        self.reward_potential_weight=None  # networth-shaping-v2: None = the dense potential uses reward_cash_weight (v42)
        self.comparison_recorded=False
        self.arena=None  # arena-in-gpu-v1: dict(workers, parity_games, parity_dir) enables script:arena:* seats
        self.distill=None  # distill-v1: dict(games, teachers, workers, label_workers, arena_config, arena_repo)


    # prep-ahead-v1 (source-v65) ------------------------------------------------------------------------------------
    def _arena_list(self,jobs):
        return [dict(game=g,seat=s,job=j) for g,j in enumerate(jobs) for s,p in enumerate(j['policies'])
                if self.arena is not None and p.startswith('script:arena:')]

    def _prep_key(self,seeds,arena):
        return (tuple(seeds),tuple((a['game'],a['seat'],a['job']['arena_opponent']['id']) for a in arena))

    def prepare_ahead(self,jobs):
        """Start preparing events + arena hook for `jobs` on a helper thread (called while the trainer updates)."""
        self._drop_prepared()
        seeds=[j['seed'] for j in jobs];arena=self._arena_list(jobs);box={}
        def run():
            try:
                t0=time.perf_counter()
                box['events']=prepare_events(seeds,workers=self.workers)
                box['hook']=None
                if arena:
                    from ppo.arena_gpu import ArenaHook
                    box['hook']=ArenaHook(arena,self.device,int(self.arena.get('workers',16)),int(self.arena.get('parity_games',0)),
                                          parity_seed=hash((tuple(seeds),int(os.environ.get('RANK',0))))%(2**31))
                box['seconds']=time.perf_counter()-t0
            except BaseException as exc:box['error']=exc
        th=threading.Thread(target=run,name='prep-ahead',daemon=True);th.start()
        self._prepared=dict(key=self._prep_key(seeds,arena),thread=th,box=box)

    def _drop_prepared(self):
        p=getattr(self,'_prepared',None);self._prepared=None
        if p:
            p['thread'].join()
            if p['box'].get('hook') is not None:
                try:p['box']['hook'].close()
                except Exception:pass

    def _take_prepared(self,seeds,arena):
        p=getattr(self,'_prepared',None)
        if not p:return None
        if p['key']!=self._prep_key(seeds,arena):
            print(json.dumps(dict(stage='prep_ahead_mismatch',rank=os.environ.get('RANK'))),flush=True)
            self._drop_prepared();return None
        self._prepared=None;p['thread'].join()
        if 'error' in p['box']:
            print(json.dumps(dict(stage='prep_ahead_error',error=repr(p['box']['error'])[:300])),flush=True)
            if p['box'].get('hook') is not None:p['box']['hook'].close()
            return None
        return p['box']

    def _capture_shard(self,hook,arena,final):
        import numpy as np,threading as _th
        n=hook.capture_n;cap=arena[:n];games=torch.tensor([a['game'] for a in cap],device=self.device)
        seats=torch.tensor([2*a['game']+a['seat'] for a in cap],device=self.device)
        data=self.rollout.data
        shard=dict(version='surrogate-capture-v1',update=getattr(self,'iteration',None),rank=int(os.environ.get('RANK',0)),
            bot_seat=[a['seat'] for a in cap],opponent_id=[a['job']['arena_opponent']['id'] for a in cap],
            family=[a['job'].get('arena_family') for a in cap],seed=[a['job']['seed'] for a in cap],
            faulted=[a['game'] in hook.bots.faults for a in cap],final_cash=final['money'][games].cpu().numpy(),
            actions=np.stack(hook.capture_rows,0),actions_json=hook.capture_json,
            state={k[6:]:v[:,games].cpu() for k,v in data.items() if k.startswith('state/')},
            history={k[8:]:v[:,seats].cpu() for k,v in data.items() if k.startswith('history/')})
        out=Path(os.environ.get('PPO_CAPTURE_DIR','/home/keith/surrogate-data'));out.mkdir(parents=True,exist_ok=True)
        path=out/('u%s-r%d.pt'%(shard['update'],shard['rank']))
        def save():
            try:tmp=path.with_suffix('.tmp');torch.save(shard,tmp);tmp.replace(path)
            except Exception as exc:print(json.dumps(dict(stage='surrogate_capture_save_error',error=repr(exc)[:300])),flush=True)
        _th.Thread(target=save,name='surrogate-capture',daemon=True).start()

    def collect(self,jobs):
        start=time.perf_counter();policies={};learner=[];external=[];arena=[]
        for game,job in enumerate(jobs):
            for seat,policy in enumerate(job['policies']):
                if self.arena is not None and policy.startswith('script:arena:'):  # arena-in-gpu-v1: real bot seat
                    if policy!='script:arena:'+job['arena_opponent']['id'] or seat in job['learner_seats']:raise ValueError('Malformed arena job')
                    external.append(2*game+seat);arena.append(dict(game=game,seat=seat,job=job));continue
                if policy not in self.models:
                    raise ValueError(f'GPU collection requires a batched neural opponent: {policy}; use the official collector for Python agents')
                policies.setdefault(policy,[]).append(2*game+seat)
            learner.extend(2*game+s for s in job['learner_seats'])
        seeds=[j['seed'] for j in jobs]
        policies=dict(sorted(policies.items()))
        if any(j.get('reference_sha256')!=self.reference_sha for j in jobs):
            raise ValueError('Collection reference mismatch')
        _pre=self._take_prepared(seeds,arena)  # prep-ahead-v1 (source-v65): resources prepared during the last update
        if _pre is not None:
            events=_pre['events'];_hook_box={'hook':_pre['hook'],'seconds':0.} if arena else {}
            _hook_thread=threading.Thread(target=lambda:None);_hook_thread.start()
        else:
            _hook_box={}  # prep-overlap-v1 (source-v62): bot sandboxes start while the simulator events are prepared
            if arena:
                def _make_hook():
                    try:
                        from ppo.arena_gpu import ArenaHook
                        t0=time.perf_counter()
                        _hook_box['hook']=ArenaHook(arena,self.device,int(self.arena.get('workers',16)),int(self.arena.get('parity_games',0)),
                                                    parity_seed=hash((tuple(seeds),int(os.environ.get('RANK',0))))%(2**31))
                        _hook_box['seconds']=time.perf_counter()-t0
                    except BaseException as exc:_hook_box['error']=exc
                _hook_thread=threading.Thread(target=_make_hook,name='arena-hook-setup',daemon=True);_hook_thread.start()
            try:events=prepare_events(seeds,workers=self.workers)
            except BaseException:
                if arena:
                    _hook_thread.join()
                    if 'hook' in _hook_box:_hook_box['hook'].close()
                raise
        prep=time.perf_counter()-start
        if self.collector is None:
            roles=[(self.models[p],seats) for p,seats in policies.items()]
            self.collector=GPUCollector(self.models['learner'],self.wq,self.mq,seeds,str(self.device),
                events=events,reference=self.reference,roles=roles,learner_seats=learner,external_seats=external)
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
            if sorted([s for r in roles for s in r.seats.tolist()]+external)!=list(range(2*len(jobs))):raise ValueError('Roles must cover each seat exactly once')
            self.collector.external_seats=external
        starts=[(game,job['start_state']) for game,job in enumerate(jobs) if job.get('start_state')]
        forcer=None
        if starts:  # start-state-v1
            from ppo.start_states import Bank,Forcer
            spec=starts[0][1]
            if any(s['bank']!=spec['bank'] or s['bank_sha256']!=spec['bank_sha256'] for _,s in starts):raise ValueError('Mixed start-state banks')
            if any(jobs[g]['family']!='selfplay' or jobs[g]['seed']!=s['seed'] or sorted(jobs[g]['learner_seats'])!=[0,1] for g,s in starts):
                raise ValueError('Start states apply to self-play jobs with the recorded seed only')
            if getattr(self,'start_bank',None) is None or self.start_bank.sha!=spec['bank_sha256'] or self.start_bank.path!=spec['bank']:
                self.start_bank=Bank(spec['bank'],spec['bank_sha256'])
            if any(self.start_bank.meta['games'][s['index']]['seed']!=s['seed'] for _,s in starts):raise ValueError('Start-state seed/index mismatch')
            forcer=Forcer(self.start_bank,[(g,s['index'],s['turn']) for g,s in starts],self.device)
        self.collector.forcer=forcer
        def progress(turn):
            print(json.dumps(dict(stage='gpu_collection',rank=int(os.environ.get('RANK',0)),
                turn=turn,games=len(jobs),seconds=time.perf_counter()-start)),flush=True)
        hook=None;arena_metrics=None;thook=None;labeled=[];distill_metrics=None
        if arena:
            _hook_thread.join()
            if 'error' in _hook_box:raise _hook_box['error']
            hook=_hook_box['hook'];hook_start=_hook_box['seconds']
        try:
            if self.distill and int(self.distill.get('games',0))>0:  # distill-v1: teachers shadow sampled learner seats
                from ppo.distill_gpu import TeacherHook,HookChain,sample_labeled
                labeled=sample_labeled(jobs,int(self.distill['games']),hash(('distill',tuple(seeds),int(os.environ.get('RANK',0))))%(2**31),self.distill)
                thook=TeacherHook(labeled,self.distill['teachers'],self.device,int(self.distill.get('workers',16)),int(self.distill.get('label_workers',4)),self.wq,self.mq)
            chain=HookChain([hook,thook]) if thook is not None else hook
            self.rollout,final=self.collector.collect(rollout=self.rollout,progress=progress,hook=chain)
            if thook is not None:
                tdata,distill_metrics=thook.finish()
                slot=torch.full((2*len(jobs),),-1,device=self.device,dtype=torch.long)
                for j,a in enumerate(labeled):slot[2*a['game']+a['seat']]=j
                tdata['teacher/slot']=slot
                self.rollout.data.update(tdata)
            if hook is not None:
                money=final['money'].cpu().tolist()
                parity=None
                try:parity=hook.launch_parity(money,self.arena.get('parity_dir'),'r%s-%d'%(os.environ.get('RANK','0'),int(time.time())))
                except Exception as exc:parity=dict(error=repr(exc)[:300])
                bots=hook.bots
                arena_metrics=dict(games=len(arena),faulted=len(bots.faults),faults={str(g):v for g,v in bots.faults.items()},
                    workers=len(bots.groups),games_per_worker=[len(g) for g in bots.groups],max_worker_turn_ms=[round(x,1) for x in bots.max_worker_ms],
                    start_seconds=round(hook_start,3),container_start_seconds=round(bots.seconds['start'],3),bot_load_seconds=round(bots.seconds['load'],3),
                    submit_seconds=round(bots.seconds['submit'],3),wait_seconds=round(bots.seconds['wait'],3),
                    host_copy_seconds=round(hook.seconds['host'],3),inject_seconds=round(hook.seconds['inject'],3),record_seconds=round(hook.seconds['record'],3),
                    parity=parity,turn_ms=bots.turn_stats,bot_ms_by_opponent=_bot_ms_by_opponent(arena,bots.game_dt))  # source-v52 bot-timing-v1
        finally:
            if hook is not None:hook.close()
            if thook is not None:thook.close()
        faulted={a['game'] for a in arena if hook is not None and a['game'] in hook.bots.faults}
        if hook is not None and getattr(hook,'capture_n',0):  # surrogate-capture-v1 (source-v67): never raises
            try:self._capture_shard(hook,arena,final)
            except Exception as exc:print(json.dumps(dict(stage='surrogate_capture_error',error=repr(exc)[:300])),flush=True)
        start_metrics=forcer.verify() if forcer is not None else None  # start-state-v1: fail closed on any start-turn state difference
        outcome,adv,reward,returns=terminal_advantages(self.rollout.data,final['money'],lam=self.gae_lambda,beta=self.reward_beta,sigma=self.reward_sigma,cash_weight=self.reward_cash_weight,cash_center=self.reward_cash_center,shape=self.reward_shape,dense=self.reward_dense,potential=self.reward_potential,potential_weight=self.reward_potential_weight)
        comparison=None
        if not self.comparison_recorded:
            _,comparison,_,_=terminal_advantages(self.rollout.data,final['money'],lam=.95,beta=self.reward_beta,sigma=self.reward_sigma,cash_weight=self.reward_cash_weight,cash_center=self.reward_cash_center,shape=self.reward_shape,dense=self.reward_dense,potential=self.reward_potential,potential_weight=self.reward_potential_weight)
            expected=reward.reshape(1,-1)-self.rollout.data['value']
            if self.gae_lambda==1.0 and not self.reward_dense and not torch.allclose(adv,expected,atol=2e-5,rtol=2e-5):raise RuntimeError('Lambda1 terminal-return identity failed')
            if not torch.isfinite(adv).all() or not torch.isfinite(returns).all():raise FloatingPointError('Nonfinite GPU advantages or returns')
            self.comparison_recorded=True
        cash=final['money'].cpu().tolist();games=[];episodes=[]
        for game,job in enumerate(jobs):
            faults=[False,False]
            if game in faulted:faults[1-job['learner_seats'][0]]=True  # arena-in-gpu-v1: opponent fault, as the official collector reports it
            games.append(dict(**job,cash=cash[game],done=True,turns=719,faults=faults))
            if game in faulted:continue  # opponent-fault game: no learner episode (official collector: discarded='opponent_fault')
            for seat in job['learner_seats']:
                flat=2*game+seat
                episodes.append(dict(gpu_data=self.rollout.data,gpu_outcomes=outcome,seat_index=flat,
                    reference_sha=self.reference_sha,
                    quantity_vocab=(self.wq,self.mq),turns=range(719),advantages=adv[:,flat],returns=returns[:,flat],
                    assignment=job,seat=seat,outcome=int(outcome[flat]),shaped_return=float(reward[flat])))
                if job.get('start_state'):episodes[-1]['train_from']=int(job['start_state']['turn'])  # start-state-v1
        if comparison is not None:
            for ep in episodes:ep['comparison_advantages']=comparison[:,ep['seat_index']]
        seconds=time.perf_counter()-start
        metrics=dict(games=games,complete_games=len(games),valid_games=len(games)-len(faulted),full_seasons=len(games)-len(faulted),
            learner_turns=sum(len(e['turns']) for e in episodes),arena_in_gpu=arena_metrics,collection_seconds=seconds,event_preparation_seconds=prep,
            games_per_second=len(games)/seconds,gpu_seconds=self.collector.profile(),
            worker_bucket_turns=dict(self.collector.worker_bucket_turns),
            actor_batches={p:len(s) for p,s in policies.items()},
            reference_graphs=len(self.collector.reference_graphs),
            caches={p:dict(encoders=len(r.encoders),workers=len(r.workers),market=int(r.market is not None)) for p,r in self.role_cache.items()},
            memory=dict(allocated=torch.cuda.memory_allocated(self.device),reserved=torch.cuda.memory_reserved(self.device),free=torch.cuda.mem_get_info(self.device)[0]),
            rollout_bytes=self.rollout.bytes,families=dict(Counter(j['family'] for j in jobs)),distill=distill_metrics)
        if forcer is not None:metrics['start_states']=dict(start_metrics,trained_turns=sum(719-e.get('train_from',0) for e in episodes if 'train_from' in e),episodes=sum('train_from' in e for e in episodes))
        return episodes,metrics


def batch_gpu_windows(rows,teacher=None):
    first,begin,end,burn=rows[0];data=first['gpu_data']
    if any(ep['gpu_data'] is not data or (b,e,k)!=(begin,end,burn) for ep,b,e,k in rows):
        raise ValueError('GPU recurrent minibatch must share rollout and window')
    seats=[ep['seat_index'] for ep,_,_,_ in rows]
    batch,memory=batch_window(data,begin,end,[],*first['quantity_vocab'],first['gpu_outcomes'],burn=burn,seat_ids=seats,
        skip_burn_events=os.environ.get('PPO_SKIP_BURN_EVENTS','1')=='1')  # v40
    batch['burn_events_skipped']=os.environ.get('PPO_SKIP_BURN_EVENTS','1')=='1' and burn>0
    batch['advantages']=torch.stack([ep['advantages'][begin:end] for ep,_,_,_ in rows],1).flatten()
    batch['return_turn']=torch.stack([ep['returns'][begin:end] for ep,_,_,_ in rows],1).flatten().to(torch.float32)
    if any(ep.get('reference_sha')!=first.get('reference_sha') for ep,_,_,_ in rows):raise ValueError('Mixed reference versions')
    batch['reference_sha']=first.get('reference_sha')
    reference=tuple(data[k][begin,seats] for k in ('reference_actor','reference_critic'))
    if teacher and 'teacher/workers' in data:  # distill-v1: teacher turns for the labelled rows of this window
        from ppo.distill_replay import build_teacher_batch
        slots=[int(data['teacher/slot'][s]) for s in seats]
        tb=build_teacher_batch(data,begin,end,seats,slots,*first['quantity_vocab'],burn)
        if tb is not None:batch['teacher']=tb
    return batch,memory,reference
