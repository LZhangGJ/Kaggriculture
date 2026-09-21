"""Independent graph ownership for mixed, batched neural opponents."""
import torch
from ppo.encoder_graph import EncoderGraph
from ppo.worker_phase_graph import WorkerPhaseGraph
from ppo.market_graph import MarketGraph
from ppo.gpu_post_worker import GPUPostWorker


class NeuralRole:
    def __init__(self,model,seats,quantities,device,compiled):
        self.model=model;self.seats=torch.tensor(seats,device=device,dtype=torch.long)
        self.quantities=quantities;self.encoders={};self.workers={};self.market=None
        self.post=GPUPostWorker(device)
        if compiled:self.post=torch.compile(self.post.__call__,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})

    def __call__(self,x,memory,seed,simulation,greedy,timed=None,worker_width=None):
        if timed is None:timed=lambda name,fn,*args:fn(*args)
        ids=self.seats
        x={k:v[ids] for k,v in x.items()};memory=tuple(v[ids] for v in memory)
        seed={k:v[ids] for k,v in seed.items()};width=seed['positions'].shape[1]
        if width not in self.encoders:self.encoders[width]=EncoderGraph(self.model,x,memory)
        actor,critic,logits,ctx=timed('role_encoder',self.encoders[width],x,memory)
        if worker_width is not None:
            width=worker_width
            for key in ('positions','inventory','order'):seed[key]=seed[key][:,:width]
        if width not in self.workers:
            self.workers[width]=WorkerPhaseGraph(self.model,self.quantities,actor,ctx,seed,greedy,finalize=True,retain_trace=False)
        worker=self.workers[width]
        wd,wf,wire,_=timed('role_workers',worker,actor,ctx,seed)
        market,ledger,farm=timed('role_post_worker',self.post,worker.state,simulation,ids)
        if self.market is None:self.market=MarketGraph(self.model,self.quantities,actor,ctx,market,ledger,farm,greedy,retain_trace=False)
        md,mf,orders,_=timed('role_market',self.market,actor,ctx,market,ledger,farm)
        return actor,critic,logits,wire,orders,wd+md,wf+mf
