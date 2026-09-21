"""Device-side request ledger. This predicts no market fills or engine transitions."""
import numpy as np
import torch
from bc_runtime import MARKET,ITEMS,CROPS
from ppo.fast_features import QuantityFeatures
from exact_decoder import ledger_vector

if len(MARKET) != 23:
    raise ValueError('GPU market ledger requires the frozen 23-entry market grammar')


def market_seed(ledger):
    o=ledger.obs;pr=o['private'];f=o['farms'][o['player']]
    costs=np.asarray([ledger.cost(i) for i in range(23)],np.float64)
    hires=[];lands=[]
    for n in range(11):
        ledger.hires=n;ledger.lands=n
        hires.append(ledger.cost(2));lands.append(ledger.cost(3))
    ledger.hires=ledger.lands=0
    return dict(base=ledger_vector(ledger),costs=costs,hires=np.asarray(hires,np.float64),lands=np.asarray(lands,np.float64),
                money=np.asarray(f['money'],np.float64),shed=np.asarray([pr['shed'].get(i,0) for i in ITEMS],np.float64),
                seeds=np.asarray([pr['seeds'].get(i,0) for i in ITEMS],np.float64),
                prices_present=np.asarray([i in o['market']['prices'] for i in ITEMS]),
                room=np.asarray(max(0,100-sum(pr['shed'].values())),np.float64))


class MarketBatch:
    def __init__(self,seed):
        self.seed=dict(seed);self.n=len(seed['base']);self.device=seed['base'].device
        for name in ('money','room'): self.seed[name]=seed[name].reshape(self.n)
        self.rows=torch.arange(self.n,device=self.device)
        self.item=torch.tensor([ITEMS.index(i) if i else -1 for _,i in MARKET],device=self.device)
        self.buyseed=torch.tensor([op=='BUY_SEED' for op,_ in MARKET],device=self.device)
        self.sell=torch.tensor([op=='SELL' for op,_ in MARKET],device=self.device)
        self.ops=torch.tensor([('STOP','NOOP','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL').index(op) for op,_ in MARKET],device=self.device)
        self.count=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.hires=self.count.clone();self.lands=self.count.clone()
        self.spend=torch.zeros(self.n,device=self.device,dtype=torch.float64);self.income=self.spend.clone()
        self.buys=torch.zeros(self.n,12,device=self.device,dtype=torch.float64);self.sells=self.buys.clone();self.seed_buys=self.buys.clone()
        self.template=torch.zeros(23,128,device=self.device)
        ids=torch.arange(1,23,device=self.device)
        self.template[ids,self.ops[1:]]=1
        self.template[torch.arange(4,23,device=self.device),8+self.item[4:]]=1
        self.template[1:,63]=1;self.template[1:,70]=np.log(2)/12
        self.template[1:,78]=1;self.template[1:,115:117]=1
        self.template[1:,119:122]=1;self.template[1:,124]=1;self.template[1:,126:128]=1
        self.minus=torch.full((self.n,23),-1,device=self.device,dtype=torch.long)

    def costs(self):
        cost=self.seed['costs'].clone()
        cost[:,2]=self.seed['hires'][self.rows,self.hires]
        cost[:,3]=self.seed['lands'][self.rows,self.lands]
        return cost

    def vector(self):
        v=self.seed['base'].clone();scale=QuantityFeatures.scaled
        v[:,1]=scale(self.spend);v[:,2]=scale(self.income)
        v[:,3]=scale(self.seed['money']-self.spend+self.income)
        v[:,5]=(10-self.count).double()/10;v[:,6]=self.hires.double()/20;v[:,7]=self.lands.double()/3
        v[:,20:32]=scale(self.buys);v[:,32:44]=scale(self.sells)
        v[:,86]=self.count>0;v[:,90]=self.count>0;v[:,96:101]=scale(self.seed_buys[:,:5])
        return v

    def candidates(self,vector):
        scale=QuantityFeatures.scaled;cost=self.costs()
        raw=self.template[None].expand(self.n,-1,-1).clone();v=vector
        raw[:,1:,69]=scale(cost[:,1:]);raw[:,1:,72]=v[:,88,None];raw[:,1:,73]=v[:,0,None]
        raw[:,1:,74]=scale((cost[:,1:]-self.seed['money'][:,None]).clamp_min(0))
        raw[:,1:,77]=self.count[:,None].double()/10
        raw[:,1:,79]=v[:,1,None];raw[:,1:,80]=v[:,2,None]
        ii=self.item[4:]
        raw[:,4:,105]=v[:,8+ii];raw[:,4:,106]=scale(self.seed['seeds'][:,ii])
        raw[:,4:,108]=scale(torch.where(self.buyseed[None,4:],self.seed_buys[:,ii],self.buys[:,ii]))
        raw[:,4:,109]=v[:,32+ii];raw[:,4:,110]=v[:,61+ii];raw[:,4:,111]=v[:,73+ii]
        raw[:,4:,112]=self.seed['prices_present'][:,ii]
        raw[:,1:,113]=v[:,3,None];raw[:,1:,117:119]=(self.count<10)[:,None,None]
        legal=(self.count<10)[:,None].expand(-1,23).clone();legal[:,0]=True;legal[:,1]=False
        return dict(raw=raw,source=self.minus,target=self.minus,workers=self.minus,items=self.item[None].expand(self.n,-1),legal=legal)

    def qstats(self,index):
        ii=self.item[index].clamp_min(0)
        available=torch.where(self.buyseed[index],self.seed['seeds'][self.rows,ii],self.seed['shed'][self.rows,ii])
        return torch.stack((self.costs()[self.rows,index],available,self.seed['money'],self.seed['room']),-1)

    def apply(self,index,quantity,active):
        do=active&(index!=0);has=do&(self.item[index]>=0)
        sell=has&self.sell[index];seed=has&self.buyseed[index];buy=has&~sell&~seed
        price=self.costs()[self.rows,index];q=quantity.double()
        self.income+=torch.where(sell,price*q,0)
        self.spend+=torch.where(has&~sell,price*q,torch.where(do&((index==2)|(index==3)),price,0))
        ii=self.item[index].clamp_min(0)[:,None]
        self.sells.scatter_add_(1,ii,torch.where(sell,q,0)[:,None]);self.buys.scatter_add_(1,ii,torch.where(buy,q,0)[:,None]);self.seed_buys.scatter_add_(1,ii,torch.where(seed,q,0)[:,None])
        self.hires+=do&(index==2);self.lands+=do&(index==3);self.count+=do
