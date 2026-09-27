"""Batched sampling at each exact worker/market depth, with numeric PPO traces."""
from contextlib import nullcontext
import time
import copy
import os
import numpy as np
import torch
from ppo.fast_features import expand_worker_cached as expand_worker
from exact_decoder import score_event, worker_delta
from features_v2 import batch_arrays
from ppo.replay import utility
from ppo.fast_features import QuantityFeatures
from ppo.gpu_market import MarketBatch
from ppo.market_graph import MarketGraph
from ppo.worker_graph import WorkerGraph
from ppo.worker_phase_graph import WorkerPhaseGraph
from ppo.transport import BatchTransfer, cpu_tree
from ppo.encoder_graph import EncoderGraph


def stack(rows, device):
    return {k: torch.as_tensor(np.stack([r[k] for r in rows]), device=device) for k in rows[0]}


class BatchedExactSampler:
    def __init__(self, models, reference, wq, mq, device='cpu', bf16=False, generator=None):
        if os.environ.get('PPO_COMPACT_WORKERS')=='1' and (os.environ.get('PPO_COMPACT_MARKET')!='1' or os.environ.get('PPO_GPU_WORKERS')!='1'):
            raise ValueError('Compact worker replay requires GPU workers and compact market replay')
        if bf16:
            raise ValueError('BF16 collection is disabled until complete-turn sampler/replay density parity passes')
        if os.environ.get('PPO_COMPILE_HEADS')=='1':
            # Fixed CUDA graph shapes include policy batch, worker width and phase.
            # These measured finite variants exceed Dynamo's default cache of eight.
            torch._dynamo.config.recompile_limit=64
        self.models, self.reference = models, reference
        self.wq, self.mq, self.device = wq, mq, torch.device(device)
        self.bf16 = bf16
        self.quantities = QuantityFeatures(wq,mq,self.device)
        self.transfer = BatchTransfer(self.device)
        self.zero_state = torch.zeros(1,256,device=self.device)
        self.states, self.ref_states = {}, {}
        self.gpu_events = []
        self.encoder_graphs = {}
        self.market_graphs = {}
        self.worker_graphs = {}
        self.worker_phase_graphs = {}
        self.seconds, self.turns, self.factors = 0., 0, 0
        # collection-overlap-v1: all official-collection draws use this generator (None keeps the default RNG).
        self.generator = generator

    def reset_seasons(self):
        self.states.clear(); self.ref_states.clear()
        self.gpu_events.clear()
        self.seconds, self.turns, self.factors = 0., 0, 0

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16) if self.bf16 else nullcontext()

    def initial(self, keys, store):
        return tuple(torch.cat([store.get(k, (self.zero_state,)*2)[i]
                                for k in keys]) for i in (0,1))

    def timed_gpu(self, name, function, *args):
        if os.environ.get('PPO_PROFILE') != '1' or self.device.type != 'cuda':
            return function(*args)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=function(*args);end.record()
        self.gpu_events.append((name,start,end))
        return result

    def gpu_timings(self):
        totals={}
        for name,start,end in self.gpu_events:
            end.synchronize()
            totals[name]=totals.get(name,0.)+start.elapsed_time(end)/1000
        return totals

    def encode(self, scope, model, x, state):
        if self.device.type != 'cuda' or os.environ.get('PPO_GRAPH_ENCODER') != '1':
            return model.encode(x,*state)
        key=(scope,id(model),tuple((k,tuple(v.shape),v.dtype) for k,v in x.items()))
        if key not in self.encoder_graphs:
            self.encoder_graphs[key]=EncoderGraph(model,x,state)
        return self.encoder_graphs[key](x,state)

    def _act(self, policy, keys, starts, greedy=False, retain=True, anchor=True):
        m = self.models[policy]
        n = len(keys)
        state = self.initial(keys, self.states)
        pre = torch.stack(state, 1).float().cpu().numpy() if retain and anchor else None
        x = self.transfer.observations([starts[k]['x'] for k in keys])
        with self.autocast():
            actor, critic, logits, ctx = self.timed_gpu('encoder',self.encode,('policy',policy),m,x,state)
        for i,k in enumerate(keys): self.states[k] = (actor[i:i+1],critic[i:i+1])
        ref_pre = None
        if retain:
            rs = self.initial(keys,self.ref_states)
            ref_pre = torch.stack(rs,1).float().cpu().numpy() if anchor else None
            with self.autocast():
                ra,rc,_,_ = self.timed_gpu('reference_encoder',self.encode,('reference',policy),self.reference,x,rs)
            for i,k in enumerate(keys): self.ref_states[k] = (ra[i:i+1],rc[i:i+1])
        records = {k:dict(x={name:np.array(value,copy=True) for name,value in starts[k]['x'].items()}, events=[], stats={}, state=pre[i] if retain and anchor else None,
                         reference_state=ref_pre[i] if retain and anchor else None) for i,k in enumerate(keys)} if retain else {}
        density = torch.zeros(n,device=self.device)
        factors = torch.zeros(n,device=self.device)
        worker_counts = [starts[k]['workers'] for k in keys]
        prefix = torch.zeros_like(actor)
        post = None
        if self.device.type == 'cuda' and os.environ.get('PPO_GPU_WORKERS') == '1':
            wd,wf,post_rows = yield from self._workers(m,keys,actor,ctx,starts,greedy,retain,records)
            md,mf = yield from self._market(m,keys,actor,ctx,post_rows,greedy,retain,records)
            density += wd+md; factors += wf+mf
        for phase in (0,1):
            if self.device.type == 'cuda' and os.environ.get('PPO_GPU_WORKERS') == '1': break
            if phase:
                post_rows = (yield ('market_begin',keys))
                md,mf = yield from self._market(m,keys,actor,ctx,post_rows,greedy,retain,records)
                density += md; factors += mf
                break
            active = list(range(n))
            next_workers = None
            for depth in range(10 if phase else max(worker_counts)):
                if phase == 0: active = [i for i,c in enumerate(worker_counts) if c>depth]
                if not active: break
                active_keys = [keys[i] for i in active]
                response = next_workers if next_workers is not None else (yield ('market' if phase else 'worker',active_keys))
                data = [response[k] for k in active_keys]
                ids = torch.tensor(active,device=self.device)
                r = torch.arange(len(ids),device=self.device)
                branch = 'candidates' if phase else 'compact'
                packed = self.transfer.mapping([d[branch] for d in data])
                led = self.transfer.array([d['ledger'] if phase else d['compact']['ledger'] for d in data])
                context = {k:v[ids] for k,v in ctx.items()}
                c = packed if phase else expand_worker(packed)
                extra = post[ids] if phase else context['workers'][r,packed['worker']]
                head = m.market_head if phase else m.worker_head
                use_graph = self.device.type == 'cuda' and phase == 0
                if use_graph:
                    need_table=self.transfer.array([d['need'] for d in data])
                    stats_table=self.transfer.array([d['qstats'] for d in data])
                    graph_key=(id(m),len(active),greedy)
                    if graph_key not in self.worker_graphs:
                        self.worker_graphs[graph_key]=WorkerGraph(m,self.quantities,actor[ids],prefix[ids],context,packed,need_table,stats_table,greedy,generator=self.generator)
                    wire,selected,count,nxt,qf,qm=self.worker_graphs[graph_key](actor[ids],prefix[ids],context,packed,need_table,stats_table)
                    if retain:
                        density.index_add_(0,ids,selected)
                        factors.index_add_(0,ids,count)
                    prefix.index_copy_(0,ids,nxt)
                    choices=wire.cpu().numpy()
                else:
                    with self.autocast():
                        lp,gates,commands,emb = score_event(m,phase,actor[ids],prefix[ids],context,c,led,extra)
                        if greedy:
                            index = torch.where(gates[:,0]>=gates[:,1],0,commands.argmax(-1)+1)
                        else: index = torch.multinomial(lp.exp(),1,generator=self.generator).squeeze(1)
                        need = self.transfer.array([d['need'] for d in data])[r,index]
                        stats = self.transfer.array([d['qstats'] for d in data])[r,index]
                        qr = torch.where(need)[0]
                        qi = torch.zeros(len(ids),device=self.device,dtype=torch.long)
                        quantity = torch.zeros_like(qi)
                        selected = lp[r,index].clone()
                        qf,qm = self.quantities(phase,index[qr],stats[qr])
                        if len(qr):
                            qlp = head.quantities(actor[ids[qr]],prefix[ids[qr]],emb[r[qr],index[qr]],led[qr],qf,qm)
                            qindex = qlp.argmax(-1) if greedy else torch.multinomial(qlp.exp(),1,generator=self.generator).squeeze(1)
                            qi[qr] = qindex
                            quantity[qr] = self.quantities.vocab[phase][qindex]
                            selected[qr] += qlp[torch.arange(len(qr),device=self.device),qindex]
                    if retain:
                        density.index_add_(0,ids,selected.float())
                        factors.index_add_(0,ids,1.+need.float())
                    # One batched host transfer for choices, never item() per action.
                    choices = torch.stack((index,quantity,qi,need.long()),1).cpu().numpy()
                # Retain data before the next command overwrites the shared arena.
                if retain:
                    data = [{branch:{name:np.array(value,copy=True) for name,value in d[branch].items()},
                             **({'ledger':d['ledger'].copy()} if phase else {})} for d in data]
                    qrows = list(range(len(active))) if use_graph else qr.cpu().tolist()
                    qcpu = qf.cpu().numpy(); mcpu = qm.cpu().numpy()
                    quantity_rows = {j:(qcpu[t],mcpu[t]) for t,j in enumerate(qrows)}
                if phase:
                    deltas = (yield ('market_apply',{k:(int(a[0]),int(a[1])) for k,a in zip(active_keys,choices)}))
                else:
                    next_workers = (yield ('worker_next',{k:(int(a[0]),int(a[1]),bool(a[3])) for k,a in zip(active_keys,choices)}))
                    deltas = {k:worker_delta(int(a[0]),int(a[1])) for k,a in zip(active_keys,choices)}
                advance_local = []
                for j,(k,a,d) in enumerate(zip(active_keys,choices,data)):
                    ix,q,qindex,has_q = map(int,a)
                    advance = not phase or ix!=0
                    if retain:
                        event = dict(phase=phase,depth=depth,index=ix,quantity=q,supervised=True,
                                     weights=np.ones(3,np.float32),delta=deltas[k],advance=advance,
                                     ledger=d['ledger'] if phase else d['compact']['ledger'],**{branch:d[branch]})
                        if has_q:
                            event.update(qindex=qindex,qfeatures=quantity_rows[j][0],qmask=quantity_rows[j][1])
                        records[k]['events'].append(event)
                    if advance: advance_local.append(j)
                if advance_local and not use_graph:
                    ai = torch.tensor(advance_local,device=self.device)
                    delta = torch.as_tensor(np.stack([deltas[k] for k in active_keys]),device=self.device)
                    with self.autocast():
                        nxt = head.advance(prefix[ids[ai]],emb[r[ai],index[ai]],quantity[ai],delta[ai])
                    prefix.index_copy_(0,ids[ai],nxt.to(prefix.dtype))
                if phase: active = [active[j] for j,a in enumerate(choices) if a[0]!=0]
        yield ('finish',keys)
        if retain:
            values = (utility(logits) + m.value_shaped(critic).float().squeeze(-1)).cpu().numpy()
            totals = torch.stack((density,factors),1).cpu().numpy()
            for i,k in enumerate(keys):
                records[k].update(old_logp=float(totals[i,0]),factor_count=int(totals[i,1]),value=float(values[i]))
        self.turns += n
        if retain: self.factors += int(totals[:,1].sum())
        return records if retain else {}


    @torch.no_grad()
    def act(self, policy, keys, starts, backend, greedy=False, retain=True):
        begin=time.perf_counter()
        gen=self._act(policy,keys,starts,greedy,retain,True)
        try:
            command,payload=next(gen)
            while True:
                command,payload=gen.send(backend.call(command,payload))
        except StopIteration as done:
            self.seconds += time.perf_counter()-begin
            return done.value

    @torch.no_grad()
    def act_groups(self, groups, starts, backend, anchor=True):
        begin=time.perf_counter()
        pending={}; generators={}; result={}
        order=['worker','worker_apply','worker_next','worker_sequence','market_begin','market','market_apply','market_sequence','finish']
        for policy,keys in groups.items():
            gen=self._act(policy,keys,starts,policy!='learner',policy=='learner',anchor)
            generators[policy]=gen; pending[policy]=next(gen)
        while pending:
            command=min((v[0] for v in pending.values()),key=order.index)
            members=[p for p,v in pending.items() if v[0]==command]
            payloads=[pending[p][1] for p in members]
            merged={} if isinstance(payloads[0],dict) else []
            for payload in payloads:
                if isinstance(merged,dict): merged.update(payload)
                else: merged.extend(payload)
            response=backend.call(command,merged)
            for policy,payload in zip(members,payloads):
                part={k:response[k] for k in payload if k in response}
                try: pending[policy]=generators[policy].send(part)
                except StopIteration as done:
                    result.update(done.value); del pending[policy]
        self.seconds += time.perf_counter()-begin
        return result

    def _workers(self,m,keys,actor,ctx,starts,greedy,retain,records):
        seed_rows=[starts[k]['worker_seed'] for k in keys]
        compact_workers=os.environ.get('PPO_COMPACT_WORKERS')=='1'
        if retain and compact_workers:
            for key,seed_row in zip(keys,seed_rows):
                records[key]['worker_seed']={name:np.array(v,copy=True) for name,v in seed_row.items()}
        seed=self.transfer.mapping(seed_rows,pad_names=('positions','inventory','order'))
        graph_key=(id(m),len(keys),seed['positions'].shape[1],greedy)
        if graph_key not in self.worker_phase_graphs:
            self.worker_phase_graphs[graph_key]=WorkerPhaseGraph(m,self.quantities,actor,ctx,seed,greedy)
        density,factors,wire,trace=self.timed_gpu('workers',self.worker_phase_graphs[graph_key],actor,ctx,seed)
        choices=wire.cpu().numpy();orders={k:[] for k in keys}
        if retain and not compact_workers:trace=cpu_tree(trace)
        for depth,rows in enumerate(choices):
            if retain and not compact_workers:
                compact,delta,qf,qm=trace[depth]
            for j,(index,quantity,qi,need,active) in enumerate(rows):
                if not active:continue
                orders[keys[j]].append((int(index),int(quantity),bool(need)))
                if retain:
                    event=dict(phase=0,depth=depth,index=int(index),quantity=int(quantity),supervised=True,
                        weights=np.ones(3,np.float32),advance=True)
                    if compact_workers:
                        event.update(delta=np.empty(0,np.float32),ledger=np.empty(0,np.float32),compact={})
                        if need:event.update(qindex=int(qi),qfeatures=np.zeros((1,12),np.float32),qmask=np.ones(1,bool))
                    else:
                        event.update(delta=delta[j],ledger=compact['ledger'][j],compact={k:v[j] for k,v in compact.items()})
                        if need:event.update(qindex=int(qi),qfeatures=qf[j],qmask=qm[j])
                    records[keys[j]]['events'].append(event)
        post_rows=yield ('worker_sequence',orders)
        return density,factors,post_rows

    def _market(self,m,keys,actor,ctx,post_rows,greedy,retain,records):
        n=len(keys);r=torch.arange(n,device=self.device)
        compact_market=os.environ.get('PPO_COMPACT_MARKET')=='1'
        for k in records:
            if compact_market:
                records[k]['market_seed']={name:np.array(v,copy=True) for name,v in post_rows[k]['market_seed'].items()}
                records[k]['quantity_vocab']=[self.wq,self.mq]

            records[k].update(post_ledger=post_rows[k]['post_ledger'],post_farm=post_rows[k]['post_farm'])
        seed=self.transfer.mapping([post_rows[k]['market_seed'] for k in keys])
        post_ledger=self.transfer.array([post_rows[k]['post_ledger'] for k in keys])
        post_farm=self.transfer.array([post_rows[k]['post_farm'] for k in keys])
        if self.device.type == 'cuda':
            graph_key=(id(m),n,greedy)
            if graph_key not in self.market_graphs:
                self.market_graphs[graph_key]=MarketGraph(m,self.quantities,actor,ctx,seed,post_ledger,post_farm,greedy,generator=self.generator)
            density,factors,wire,trace=self.timed_gpu('market',self.market_graphs[graph_key],actor,ctx,seed,post_ledger,post_farm)
            choices=wire.cpu().numpy()
        else:
            state=MarketBatch(seed)
            post=m.summarize_post_worker(post_ledger,post_farm)
            prefix=torch.zeros_like(actor);active=torch.ones(n,device=self.device,dtype=torch.bool)
            density=torch.zeros(n,device=self.device);factors=density.clone();trace=[];wire=[]
            for depth in range(10):
                led=state.vector();c=state.candidates(led)
                lp,gates,commands,emb=score_event(m,1,actor,prefix,ctx,c,led,post)
                index=torch.where(gates[:,0]>=gates[:,1],0,commands.argmax(-1)+1) if greedy else torch.multinomial(lp.exp(),1,generator=self.generator).squeeze(1)
                index=torch.where(active,index,0)
                need=state.item[index]>=0
                qf,qm=self.quantities(1,index,state.qstats(index))
                qlp=m.market_head.quantities(actor,prefix,emb[r,index],led,qf,qm)
                qi=qlp.argmax(-1) if greedy else torch.multinomial(qlp.exp(),1,generator=self.generator).squeeze(1)
                quantity=torch.where(need,self.quantities.vocab[1][qi],0)
                if retain:
                    density+=(lp[r,index]+torch.where(need,qlp[r,qi],0))*active
                    factors+=(1+need.float())*active
                state.apply(index,quantity,active);delta=state.vector()-led
                advance=active&(index!=0)
                nxt=m.market_head.advance(prefix,emb[r,index],quantity,delta)
                prefix=torch.where(advance[:,None],nxt,prefix)
                choice=torch.stack((index,quantity,qi,need.long(),active.long()),-1)
                wire.append(choice)
                if retain: trace.append((led,c,delta,qf,qm))
                active=advance
                if not bool(active.any()): break
            choices=torch.stack(wire).cpu().numpy()
        used=int(choices[:,:,4].any(axis=1).sum())
        choices=choices[:used]
        if retain and not compact_market:trace=cpu_tree(trace[:used])
        orders={k:[] for k in keys}
        for depth,rows in enumerate(choices):
            if retain and not compact_market:
                led,c,delta,qf,qm=trace[depth]
            for j,(index,quantity,qi,need,valid) in enumerate(rows):
                if not valid: continue
                index,quantity,qi=int(index),int(quantity),int(qi)
                if index: orders[keys[j]].append((index,quantity))
                if retain:
                    event=dict(phase=1,depth=depth,index=index,quantity=quantity,supervised=True,
                        weights=np.ones(3,np.float32),advance=index!=0)
                    if compact_market:
                        # Empty feature slots are filled before any PPO model call.
                        event.update(delta=np.empty(0,np.float32),ledger=np.empty(0,np.float32),candidates={})
                        if need:event.update(qindex=qi,qfeatures=np.zeros((1,12),np.float32),qmask=np.ones(1,bool))
                    else:
                        event.update(delta=delta[j],ledger=led[j],candidates={name:v[j] for name,v in c.items()})
                        if need:event.update(qindex=qi,qfeatures=qf[j],qmask=qm[j])
                    records[keys[j]]['events'].append(event)
        yield ('market_sequence',orders)
        return density,factors
