"""Compact recurrent policy with an ordered, resource-aware command decoder."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import native


class Policy(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.encoder=nn.Sequential(nn.Linear(1328,width),nn.Tanh())
        self.memory=nn.GRU(width,width,batch_first=True)
        self.candidate=nn.Sequential(nn.Linear(59,width),nn.Tanh())
        self.query=nn.Linear(width*2,width)
        self.quantity=nn.Linear(width*2,1)
        self.prefix=nn.GRUCell(width+1,width)
        self.value=nn.Linear(width,1)

    @staticmethod
    def inputs(tiles,glob):
        tiles=torch.as_tensor(tiles,dtype=torch.float32)
        glob=torch.as_tensor(glob,dtype=torch.float32)
        pooled=F.avg_pool2d(tiles.flatten(0,-4) if tiles.ndim>4 else tiles,2).flatten(1)
        return torch.cat((pooled,glob.reshape(-1,128)),-1)

    def step(self,x,memory):
        y,next_memory=self.memory(self.encoder(x)[:,None],memory[None])
        h=y[:,0];return h,next_memory[0],self.value(h)[:,0]

    def sequence(self,x):
        # Complete 719-turn sequences. Hidden state resets at each game start.
        t,b,_=x.shape
        y,_=self.memory(self.encoder(x).transpose(0,1),torch.zeros(1,b,self.width))
        h=y.transpose(0,1);return h,self.value(h)[...,0]

    def score(self,h,prefix,features,owners,bounds,offsets,selected=None,quantity=None,greedy=False):
        n=len(h);c=self.candidate(features);q=torch.tanh(self.query(torch.cat((h,prefix),-1)))
        logits=(c*q[owners]).sum(-1)/(self.width**.5)
        maxlen=int((offsets[1:]-offsets[:-1]).max())
        pos=torch.arange(len(features))-offsets[:-1][owners]
        padded=logits.new_full((n,maxlen),-torch.inf)
        padded[owners,pos]=logits
        dist=torch.distributions.Categorical(logits=padded)
        if selected is None: selected=dist.logits.argmax(-1) if greedy else dist.sample()
        row=offsets[:-1]+selected;chosen=c[row]
        count=(bounds[row]-1).float()
        qlogits=self.quantity(torch.cat((q,chosen),-1))[:,0].clamp(-8,8)
        qtydist=torch.distributions.Binomial(total_count=count.double(),logits=qlogits.double())
        if quantity is None:
            quantity=1+(torch.round(count*torch.sigmoid(qlogits)) if greedy else qtydist.sample()).long()
        lp=dist.log_prob(selected)+qtydist.log_prob((quantity-1).double()).float()
        # Binomial entropy is expensive for large supports; report command entropy.
        entropy=dist.entropy()
        next_prefix=self.prefix(torch.cat((chosen,torch.log1p(quantity.float())[:,None]/10),-1),prefix)
        return selected,quantity,lp,entropy,next_prefix


@dataclass
class Stage:
    features: torch.Tensor
    owners: torch.Tensor
    bounds: torch.Tensor
    offsets: torch.Tensor
    samples: torch.Tensor
    selected: torch.Tensor
    quantity: torch.Tensor


class PolicyAgent:
    def __init__(self,model,greedy=True):
        self.model=model.eval();self.memory=torch.zeros(1,model.width);self.greedy=greedy

    @torch.no_grad()
    def __call__(self,obs):
        actor=native.Actor(obs);tiles,glob=actor.encode()
        x=self.model.inputs(tiles[None],glob[None]);h,self.memory,_=self.model.step(x,self.memory)
        prefix=torch.zeros_like(h)
        for depth in range(29):
            f,b=actor.menu()
            if len(b)==0:break
            f=torch.from_numpy(f);b=torch.from_numpy(b)
            selected,qty,_,_,prefix=self.model.score(h,prefix,f,torch.zeros(len(b),dtype=torch.long),b,
                torch.tensor([0,len(b)]),greedy=self.greedy)
            actor.apply(int(selected[0]),int(qty[0]))
        else: raise RuntimeError('actor exceeded full joint command depth')
        return actor.action()


def save(model,path,metadata):
    torch.save({'width':model.width,'state_dict':model.state_dict(),'metadata':metadata},path)


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True);p=Policy(d['width']);p.load_state_dict(d['state_dict'])
    return p.eval(),d['metadata']
