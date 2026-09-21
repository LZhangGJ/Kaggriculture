"""Full worker request program on-device; experimental until full-game parity."""
import torch
import os
import json
from ppo.fast_heads import score_event
from ppo.worker_state import WorkerScenarios
from ppo.worker_features_device import WorkerFeatures
from ppo.fast_features import expand_worker_cached


class WorkerPhaseGraph:
    def __init__(self,model,quantities,actor,context,seed,greedy,finalize=False,retain_trace=True):
        self.actor=actor.clone();self.seed={k:v.clone() for k,v in seed.items()}
        width=seed['positions'].shape[1]
        self.context={k:v.clone() for k,v in context.items() if k in ('cells','market')}
        self.context['workers']=context['workers'][:,:width].clone()
        scenarios=WorkerScenarios(self.seed);state=scenarios.state;features=WorkerFeatures(state)
        self.scenarios=scenarios
        self.state=state;self.features=features
        n=len(actor);rows=state.rows
        if os.environ.get('PPO_COMPILE_WORKERS') == '1':
            features=torch.compile(features.__call__,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
        score=score_event;quantity_head=model.worker_head.quantities;advance_head=model.worker_head.advance
        if os.environ.get('PPO_COMPILE_HEADS')=='1':
            options={'triton.cudagraphs':False}
            score=torch.compile(score,fullgraph=True,dynamic=False,options=options)
            quantity_head=torch.compile(quantity_head,fullgraph=True,dynamic=False,options=options)
        if os.environ.get('PPO_COMPILE_FEEDBACK')=='1':
            from ppo.compiled_feedback import compiled_feedback
            advance_head=compiled_feedback(model.worker_head)
        def decode():
            actor=self.actor;prefix=torch.zeros_like(actor)
            density=torch.zeros(n,device=actor.device);factors=density.clone()
            trace=[];choices=[]
            scenarios.reset()
            counts=torch.zeros(n,5,device=actor.device,dtype=torch.long)
            for depth in range(width):
                compact,need_table,stats=features(state.workers[depth].expand(n),counts)
                ledger=compact['ledger'];candidate=expand_worker_cached(compact)
                lp,gates,commands,emb=score(model,0,actor,prefix,self.context,candidate,ledger,self.context['workers'][:,depth])
                active=state.seed['count'].reshape(n)>depth
                index=(torch.where(gates[:,0]>=gates[:,1],0,commands.argmax(-1)+1)
                       if greedy else torch.multinomial(lp.exp(),1).squeeze(1))
                index=torch.where(active,index,0);need=need_table[rows,index]&active
                qf,qm=quantities(0,index,stats[rows,index])
                qlp=quantity_head(actor,prefix,emb[rows,index],ledger,qf,qm)
                qi=qlp.argmax(-1) if greedy else torch.multinomial(qlp.exp(),1).squeeze(1)
                qi=torch.where(need,qi,0);quantity=torch.where(need,quantities.vocab[0][qi],0)
                density+=(lp[rows,index]+torch.where(need,qlp[rows,qi],0))*active
                factors+=(1+need.float())*active
                delta=torch.zeros(n,128,device=actor.device);delta.scatter_(1,index[:,None],1)
                delta[:,44]=quantities.scaled(quantity.double())
                nxt=advance_head(prefix,emb[rows,index],quantity,delta)
                prefix=torch.where(active[:,None],nxt,prefix)
                counts.scatter_add_(1,(index-15).clamp(0,4)[:,None],((index>=15)&(index<20)).long()[:,None])
                if depth+1<width or finalize:scenarios.append(depth,index,quantity)
                choices.append(torch.stack((index,quantity,qi,need.long(),active.long()),-1))
                if retain_trace:trace.append((compact,delta,qf,qm))
            return density,factors,torch.stack(choices),trace
        # Initialize static cached operation tables outside capture.
        state.reset();expand_worker_cached(features(state.workers[0].expand(n),torch.zeros(n,5,device=actor.device,dtype=torch.long))[0])
        rng=torch.cuda.get_rng_state(actor.device)
        stream=torch.cuda.Stream(device=actor.device)
        stream.wait_stream(torch.cuda.current_stream(actor.device))
        with torch.cuda.stream(stream):
            for _ in range(2):decode()
        torch.cuda.current_stream(actor.device).wait_stream(stream)
        # Lazy buckets can follow asynchronous simulator work. Finish all work
        # before capture and reuse the stream on which decode was warmed.
        torch.cuda.synchronize(actor.device)
        expected=None
        if n in (320,384) and width==14:
            # Eager scenario updates replace Python tensor attributes. Run the
            # oracle BEFORE capture so final attributes remain graph-owned.
            torch.cuda.set_rng_state(rng,actor.device)
            expected=tuple(v.clone() for v in decode()[:3])
            torch.cuda.synchronize(actor.device)
            torch.cuda.set_rng_state(rng,actor.device)
        self.graph=torch.cuda.CUDAGraph()
        # The simulator output is complete and capture inputs are owned clones.
        # Retain same-thread checks while independent runtime threads remain live.
        print(json.dumps(dict(stage='worker_capture_begin',rank=os.environ.get('RANK'),batch=n,width=width,greedy=greedy,warmup_complete=True,capture_mode='thread_local')),flush=True)
        with torch.cuda.graph(self.graph,stream=stream,capture_error_mode='thread_local'):self.output=decode()
        torch.cuda.current_stream(actor.device).wait_stream(stream)
        torch.cuda.set_rng_state(rng,actor.device)
        if expected is not None:
            # Check the formerly failing learner bucket on its actual inputs.
            # This runs only on construction and restores the action RNG.
            self.graph.replay()
            torch.testing.assert_close(self.output[0],expected[0],rtol=0,atol=.002)
            torch.testing.assert_close(self.output[1],expected[1],rtol=0,atol=0)
            torch.testing.assert_close(self.output[2],expected[2],rtol=0,atol=0)
            torch.cuda.set_rng_state(rng,actor.device)
            print(json.dumps(dict(stage='worker_capture_parity',rank=os.environ.get('RANK'),batch=n,width=width,actions_exact=True,factors_exact=True,density_tolerance=.002)),flush=True)
        if not torch.equal(torch.cuda.get_rng_state(actor.device),rng):raise RuntimeError('Worker capture changed RNG')
        print(json.dumps(dict(stage='worker_capture_complete',rank=os.environ.get('RANK'),batch=n,width=width,rng_preserved=True)),flush=True)

    def __call__(self,actor,context,seed):
        self.actor.copy_(actor)
        for k in ('cells','market'):self.context[k].copy_(context[k])
        self.context['workers'].copy_(context['workers'][:,:self.context['workers'].shape[1]])
        for k in self.seed:self.seed[k].copy_(seed[k])
        self.graph.replay()
        return self.output
