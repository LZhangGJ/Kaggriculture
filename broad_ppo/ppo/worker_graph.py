"""One fixed-shape worker decision, including quantity and prefix feedback."""
import torch
from exact_decoder import score_event
from ppo.fast_features import expand_worker_cached


class WorkerGraph:
    def __init__(self, model, quantities, actor, prefix, context, compact, need, stats, greedy, generator=None):
        self.actor=actor.clone();self.prefix=prefix.clone()
        self.context={k:v.clone() for k,v in context.items() if k in ('cells','market')}
        rows=torch.arange(len(actor),device=actor.device)
        self.context['workers']=context['workers'][rows,compact['worker']][:,None].clone()
        self.compact={k:v.clone() for k,v in compact.items()}
        self.need=need.clone();self.stats=stats.clone()
        n=len(actor);rows=torch.arange(n,device=actor.device)
        self.rows=rows
        # Populate cached static tables before capture.
        expand_worker_cached(self.compact)
        def decode():
            c=expand_worker_cached(self.compact);ledger=self.compact['ledger']
            c['workers']=torch.zeros_like(c['workers'])
            extra=self.context['workers'][:,0]
            lp,gates,commands,emb=score_event(model,0,self.actor,self.prefix,self.context,c,ledger,extra)
            index=(torch.where(gates[:,0]>=gates[:,1],0,commands.argmax(-1)+1)
                   if greedy else torch.multinomial(lp.exp(),1,generator=generator).squeeze(1))
            need=self.need[rows,index]
            qf,qm=quantities(0,index,self.stats[rows,index])
            qlp=model.worker_head.quantities(self.actor,self.prefix,emb[rows,index],ledger,qf,qm)
            qi=qlp.argmax(-1) if greedy else torch.multinomial(qlp.exp(),1,generator=generator).squeeze(1)
            qi=torch.where(need,qi,0)
            quantity=torch.where(need,quantities.vocab[0][qi],0)
            density=lp[rows,index]+torch.where(need,qlp[rows,qi],0)
            delta=torch.zeros(n,128,device=actor.device)
            delta.scatter_(1,index[:,None],1)
            delta[:,44]=quantities.scaled(quantity.double())
            nxt=model.worker_head.advance(self.prefix,emb[rows,index],quantity,delta)
            return torch.stack((index,quantity,qi,need.long()),1),density,1+need.float(),nxt,qf,qm
        # collection-overlap-v1: with a dedicated generator this graph never touches the default CUDA RNG
        # and captures in thread_local mode so another collection thread cannot fail the capture.
        rng=None if generator is not None else torch.cuda.get_rng_state(actor.device)
        stream=torch.cuda.Stream(device=actor.device)
        stream.wait_stream(torch.cuda.current_stream(actor.device))
        with torch.cuda.stream(stream):
            for _ in range(3):decode()
        torch.cuda.current_stream(actor.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        if generator is not None:
            self.graph.register_generator_state(generator)
            with torch.cuda.graph(self.graph,capture_error_mode='thread_local'):self.output=decode()
        else:
            with torch.cuda.graph(self.graph):self.output=decode()
            torch.cuda.set_rng_state(rng,actor.device)

    def __call__(self,actor,prefix,context,compact,need,stats):
        self.actor.copy_(actor);self.prefix.copy_(prefix)
        for k in ('cells','market'):self.context[k].copy_(context[k])
        rows=torch.arange(len(actor),device=actor.device)
        self.context['workers'].copy_(context['workers'][rows,compact['worker']][:,None])
        for k in self.compact:self.compact[k].copy_(compact[k])
        self.need.copy_(need);self.stats.copy_(stats)
        self.graph.replay()
        return self.output
