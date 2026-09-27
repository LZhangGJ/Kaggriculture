"""Experimental device-only full-season collection with the exact BC decoder.

No Python observations or actions cross the per-turn boundary. Each policy role
must own a separate instance because CUDA graphs retain model/buffer pointers.
"""
import torch
import os
from ppo.gpu_env import GpuGameBatch
from ppo.gpu_features import GPUFeatures,GPUHistory,worker_seed
from ppo.gpu_post_worker import GPUPostWorker
from ppo.fast_features import QuantityFeatures
from ppo.encoder_graph import EncoderGraph
from ppo.worker_phase_graph import WorkerPhaseGraph
from ppo.market_graph import MarketGraph
from ppo.replay import utility

WORKER_BUCKETS=((1,2,4,6,8,10,12,14,16,20,24,28,33)
                if os.environ.get('PPO_FINE_WORKER_BUCKETS')=='1' else (1,2,4,8,16,33))


class GPUActionCodec:
    def __init__(self,device):
        from exact_actions import WORKER
        from bc_runtime import MARKET,ITEMS
        from kaggriculture_jax.constants import UnitOp,MarketOp
        self.worker_op=torch.tensor([int(UnitOp[op]) for op,_ in WORKER],device=device,dtype=torch.int8)
        self.worker_item=torch.tensor([ITEMS.index(it) if it else -1 for _,it in WORKER],device=device,dtype=torch.int8)
        self.market_op=torch.tensor([int(MarketOp[op]) if op in MarketOp.__members__ else 0 for op,_ in MARKET],device=device,dtype=torch.int8)
        self.market_item=torch.tensor([ITEMS.index(it) if it else -1 for _,it in MARKET],device=device,dtype=torch.int8)

    def __call__(self,workers,market,count):
        n=workers.shape[1];games=n//2
        ix=workers[:,:,0].T;need=workers[:,:,3].T.bool()
        mi=market[:,:,0].T;active=market[:,:,4].T.bool()&(mi!=0)
        def shape(x):return x.reshape(games,2,*x.shape[1:]).contiguous()
        return dict(unit_op=shape(self.worker_op[ix]),unit_item=shape(self.worker_item[ix]),
            unit_amount=shape(torch.where(need,workers[:,:,1].T,1).int()),unit_count=shape(count.to(torch.int8)),
            market_op=shape(torch.where(active,self.market_op[mi],0)),
            market_item=shape(torch.where(active,self.market_item[mi],-1)),
            market_amount=shape(torch.where(active,market[:,:,1].T,0).int()),
            market_count=shape(active.sum(-1).to(torch.int8)))


class TensorRollout:
    """One allocation per field; numeric state replaces Python event objects."""
    def __init__(self,length):self.length=length;self.data={};self.used=0

    def append(self,turn,values):
        for name,value in values.items():
            if name not in self.data:
                self.data[name]=torch.empty((self.length,*value.shape),dtype=value.dtype,device=value.device)
            self.data[name][turn].copy_(value)
        self.used=turn+1

    @property
    def bytes(self):return sum(x.numel()*x.element_size() for x in self.data.values())


class GPUCollector:
    def __init__(self,model,wq,mq,seeds,device='cuda:0',compiled=True,greedy=False,events=None,reference=None,roles=None,learner_seats=None,external_seats=None):
        # Fused eval-only Transformer kernels differ from the differentiable
        # path enough to corrupt PPO ratios. Use the same SDPA path in both.
        torch.backends.mha.set_fastpath_enabled(False)
        self.model=model;self.device=torch.device(device);self.n=2*len(seeds)
        self.compiled=compiled
        self.env=GpuGameBatch(seeds,device,events);self.history=self.new_history()
        self.features=GPUFeatures(device);self.post=GPUPostWorker(device);self.codec=GPUActionCodec(device)
        self.seed_fn=worker_seed;self.greedy=greedy
        if compiled:
            opts={'triton.cudagraphs':False}
            self.features=torch.compile(self.features.__call__,fullgraph=True,dynamic=False,options=opts)
            self.post=torch.compile(self.post.__call__,fullgraph=True,dynamic=False,options=opts)
            self.seed_fn=torch.compile(worker_seed,fullgraph=True,dynamic=False,options=opts)
            self.codec=torch.compile(self.codec.__call__,fullgraph=True,dynamic=False,options=opts)
        self.quantities=QuantityFeatures(wq,mq,device)
        self.memory=(torch.zeros(self.n,256,device=device),torch.zeros(self.n,256,device=device))
        self.encoder=None;self.workers=None;self.market=None
        self.encoder_graphs={};self.reference_graphs={}
        self.worker_graphs={}
        self.reference=reference;self.reference_encoder=None
        self.reference_memory=tuple(torch.zeros_like(v) for v in self.memory)
        self.timings=[]
        self.worker_bucket_turns={}
        self.roles=None
        self.learner_mask=torch.ones(self.n,device=device,dtype=torch.bool)
        self.learner_ids=torch.arange(self.n,device=device)
        self.external_seats=list(external_seats or [])
        self.forcer=None;self.force_turn=0  # start-state-v1: ppo.start_states.Forcer, or None (unchanged collection)
        if roles is not None:
            from ppo.gpu_roles import NeuralRole
            covered=[seat for _,seats in roles for seat in seats]
            # arena-in-gpu-v1: external seats (real arena bots) get their actions from a per-turn hook, not a role.
            if sorted(covered+list(external_seats or []))!=list(range(self.n)):raise ValueError('Roles must cover each seat exactly once')
            if learner_seats is None:raise ValueError('Mixed collection requires explicit learner seats')
            self.roles=[NeuralRole(m,seats,self.quantities,device,compiled) for m,seats in roles if seats]
            self.learner_mask.zero_();self.learner_mask[learner_seats]=True
            self.learner_ids=torch.tensor(learner_seats,device=device,dtype=torch.long)

    def new_history(self):
        history=GPUHistory(self.n//2,self.device)
        if self.compiled:
            for name in ('observe','remember'):
                setattr(history,name,torch.compile(getattr(history,name),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False}))
        return history

    def timed(self,name,fn,*args):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=fn(*args);end.record()
        self.timings.append((name,start,end))
        return result

    def profile(self):
        torch.cuda.synchronize(self.device)
        result={}
        for name,start,end in self.timings:result[name]=result.get(name,0)+start.elapsed_time(end)/1000
        return result

    def reset(self,seeds,events=None):
        if len(seeds)*2!=self.n:raise ValueError('Reusable collector requires a fixed batch size')
        self.env.reset(seeds,events)
        self.history=self.new_history()
        self.memory=tuple(torch.zeros_like(v) for v in self.memory)
        self.reference_memory=tuple(torch.zeros_like(v) for v in self.reference_memory)
        self.timings.clear()
        self.worker_bucket_turns.clear()

    @torch.no_grad()
    def act(self,state,history):
        x=self.timed('features',self.features,state,*history)
        seed=self.timed('worker_seed',self.seed_fn,state)
        width=33
        if os.environ.get('PPO_WORKER_BUCKETS')=='1':
            # One control scalar; observations and worker resolution stay on GPU.
            # Benchmark against fixed33: the saved inactive decoder work must
            # outweigh this synchronization before enabling it for training.
            if self.roles is not None and os.environ.get('PPO_ROLE_WORKER_BUCKETS')=='1':
                counts=torch.stack([seed['count'][role.seats].max() for role in self.roles]).cpu().tolist()
                self.role_widths=[next(v for v in WORKER_BUCKETS if v>=c) for c in counts]
                count=max(counts)
            else:
                count=int(seed['count'].max().item())
                self.role_widths=None
            width=next(v for v in WORKER_BUCKETS if v>=count)
            bucket=str(self.role_widths or [width])
            self.worker_bucket_turns[bucket]=self.worker_bucket_turns.get(bucket,0)+1
            for key in ('positions','inventory','order'):seed[key]=seed[key][:,:width]
            for key in ('workers','worker_valid'):x[key]=x[key][:,:2*width]
        if self.roles is not None:
            # Only a missing graph needs a host completion boundary. Normal
            # replay retains the asynchronous DLPack simulator path.
            if any((width not in role.encoders or
                    (self.role_widths[i] if getattr(self,'role_widths',None) else width) not in role.workers or
                    role.market is None) for i,role in enumerate(self.roles)) or (
                    self.reference is not None and (width,len(self.learner_ids)) not in self.reference_graphs):
                import jax
                jax.block_until_ready(self.env.state)
            return self.act_roles(x,seed,state,width)
        if width not in self.encoder_graphs:self.encoder_graphs[width]=EncoderGraph(self.model,x,self.memory)
        self.encoder=self.encoder_graphs[width]
        actor,critic,logits,ctx=self.timed('encoder',self.encoder,x,self.memory)
        self.memory=actor,critic
        if self.reference is not None:
            if width not in self.reference_graphs:self.reference_graphs[width]=EncoderGraph(self.reference,x,self.reference_memory)
            self.reference_encoder=self.reference_graphs[width]
            ra,rc,_,_=self.timed('reference_encoder',self.reference_encoder,x,self.reference_memory)
            self.reference_memory=ra,rc
        if width not in self.worker_graphs:
            self.worker_graphs[width]=WorkerPhaseGraph(self.model,self.quantities,actor,ctx,seed,self.greedy,finalize=True,retain_trace=False)
        self.workers=self.worker_graphs[width]
        wd,wf,wire,_=self.timed('workers',self.workers,actor,ctx,seed)
        market_seed,ledger,farm=self.timed('post_worker',self.post,self.workers.state,state)
        if self.market is None:
            self.market=MarketGraph(self.model,self.quantities,actor,ctx,market_seed,ledger,farm,self.greedy,retain_trace=False)
        md,mf,orders,_=self.timed('market',self.market,actor,ctx,market_seed,ledger,farm)
        self.timed('history_requests',self.history.remember,orders[:,:,0],orders[:,:,1],orders[:,:,4].bool()&(orders[:,:,0]!=0))
        if width<33:
            wire=torch.cat((wire,torch.zeros(33-width,*wire.shape[1:],device=wire.device,dtype=wire.dtype)),0)
        action=self.timed('action_encoding',self.codec,wire,orders,seed['count'])
        return action,wire,orders,wd+md,wf+mf,utility(logits)+self.model.value_shaped(critic).float().squeeze(-1)

    def act_roles(self,x,seed,state,width):
        if getattr(self,'external_seats',None):  # arena-in-gpu-v1: seats no role writes stay finite (zeros)
            actor=torch.zeros_like(self.memory[0]);critic=torch.zeros_like(self.memory[1])
            value=torch.zeros(self.n,device=self.device);density=torch.zeros_like(value);factors=torch.zeros_like(value)
        else:
            actor=torch.empty_like(self.memory[0]);critic=torch.empty_like(self.memory[1])
            value=torch.empty(self.n,device=self.device);density=torch.empty_like(value);factors=torch.empty_like(value)
        wire=torch.zeros(33,self.n,5,device=self.device,dtype=torch.long)
        orders=torch.zeros(10,self.n,5,device=self.device,dtype=torch.long)
        for role_i,role in enumerate(self.roles):
            role_width=None if not getattr(self,'role_widths',None) else self.role_widths[role_i]
            if (role_width or width) not in role.workers:
                import json
                print(json.dumps(dict(stage='worker_capture_role',rank=os.environ.get('RANK'),role=role_i,batch=len(role.seats),width=role_width or width)),flush=True)
            a,c,logits,w,m,lp,f=self.timed('role_decode',role,x,self.memory,seed,state,self.greedy,self.timed,role_width)
            ids=role.seats
            actor[ids]=a;critic[ids]=c;value[ids]=utility(logits)+role.model.value_shaped(c).float().squeeze(-1)
            wire[:w.shape[0],ids]=w;orders[:,ids]=m;density[ids]=lp;factors[ids]=f
        self.memory=actor,critic
        if self.reference is not None:
            # Frozen opponent trajectories never enter PPO; no KL anchors are
            # needed for those seats. Preserve full-size scatter storage only.
            ids=self.learner_ids;key=(width,len(ids))
            rx={k:v[ids] for k,v in x.items()}
            rm=tuple(v[ids] for v in self.reference_memory)
            if key not in self.reference_graphs:self.reference_graphs[key]=EncoderGraph(self.reference,rx,rm)
            a,c,_,_=self.timed('reference_encoder',self.reference_graphs[key],rx,rm)
            self.reference_memory[0][ids]=a;self.reference_memory[1][ids]=c
        if self.forcer is None:
            self.timed('history_requests',self.history.remember,orders[:,:,0],orders[:,:,1],orders[:,:,4].bool()&(orders[:,:,0]!=0))
        else:  # start-state-v1: forced prefix seats remember the recorded requests
            ho=self.forcer.history_orders(self.force_turn,orders)
            self.timed('history_requests',self.history.remember,ho[:,:,0],ho[:,:,1],ho[:,:,4].bool()&(ho[:,:,0]!=0))
        return self.timed('action_encoding',self.codec,wire,orders,seed['count']),wire,orders,density,factors,value

    @torch.no_grad()
    def collect(self,retain=True,progress=None,rollout=None,hook=None):
        rollout=TensorRollout(719) if rollout is None else rollout
        forcer=self.forcer
        if forcer is not None and self.roles is None:raise ValueError('start-state-v1 requires role collection')
        _prof=None;_ptrig='/home/keith/bench-20260924/PROFILE_COLLECT'  # profile-v1 (source-v57)
        for turn in range(719):
            if turn==200 and os.path.exists(_ptrig):
                import time as _time
                torch.cuda.synchronize(self.device);_prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);_prof.__enter__();_pt0=_time.perf_counter()
            if turn==230 and _prof is not None:
                import time as _time
                from ppo.train import _profile_report
                torch.cuda.synchronize(self.device);_pw=_time.perf_counter()-_pt0;_prof.__exit__(None,None,None)
                _profile_report(_prof,'/home/keith/bench-20260924/profiles/collect-%d-%d.txt'%(os.getpid(),int(_time.time())),_pw,'collection turns 200-230, pid %d, %d seats'%(os.getpid(),self.n));_prof=None
            state=self.env.tensors()
            if forcer is not None:self.force_turn=turn;forcer.observe(turn,state)
            if hook is not None:hook.before_act(turn,state)  # arena-in-gpu-v1: bots start on this state while the GPU acts
            history=self.timed('history',self.history.observe,state)
            if retain:
                values={f'state/{k}':v for k,v in state.items()}
                values.update({f'history/{k}':v for k,v in zip(('rows','valid','ema','variance','count','long'),history)})
                # Save pre-action recurrent state. FP32 anchors preserve density.
                values.update(actor=self.memory[0],critic=self.memory[1])
                values['learner_mask']=self.learner_mask if forcer is None else forcer.mask(turn,self.learner_mask)
                if self.reference is not None:values.update(reference_actor=self.reference_memory[0],reference_critic=self.reference_memory[1])
                self.timed('storage',rollout.append,turn,values)
            action,workers,market,logp,factors,value=self.act(state,history)
            if forcer is not None:  # start-state-v1: recorded actions drive the prefix; the sampled prefix decisions leave no trace
                action=forcer.force(turn,action)
                workers,market,logp,factors=forcer.scrub(turn,workers,market,logp,factors)
            if retain:self.timed('storage',rollout.append,turn,dict(workers=workers.int(),market=market.int(),logp=logp,factors=factors,value=value))
            # Release exported simulator views before optional XLA donation.
            # Rollout storage owns copies; feature/decoder buffers own theirs.
            if retain:del values
            del state
            if hook is not None:action=hook.after_act(turn,action)  # arena-in-gpu-v1: bot actions into the external seats
            self.timed('simulator',self.env.step,action,False)
            if progress is not None and (turn+1)%120==0:progress(turn+1)
        torch.cuda.synchronize(self.device)
        final=self.env.tensors()
        for key in ('hand_cap_hits','market_loop_cap_hits','price_lut_oob'):
            if final[key].bool().any().item():raise RuntimeError(f'Simulator limit reached: {key}')
        if not final['done'].all().item():raise RuntimeError('Nonterminal games after 719 transitions')
        return rollout,final
