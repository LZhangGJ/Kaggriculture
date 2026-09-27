"""Batched own-side worker projection; no market settlement or day transition.

This module is under differential test and is not yet used by collection.
"""
import numpy as np
import torch
import os
from bc_runtime import ITEMS, CROPS
from exact_actions import WORKER
from worker_phase import engine

KIND, ITEM, BIRTH, YIELD, WATER, CARE, FERT, NEGLECT, AVAILABLE, LIFE, BONUS = range(11)
KINDS = (None, 'LOCKED', 'WEED', 'PLANT', 'COOP', 'PASTURE')


def pack(obs):
    farm=obs['farms'][obs['player']]; private=obs['private']
    positions=[farm['farmer'],*farm['hands']]
    tiles=np.zeros((100,11),np.int64)
    tiles[:,ITEM]=-1;tiles[:,BIRTH]=obs['day'];tiles[:,FERT]=-1;tiles[:,LIFE]=-1
    for y,row in enumerate(farm['tiles']):
        for x,t in enumerate(row):
            a=tiles[y*10+x]
            if not isinstance(t,dict):a[KIND]=KINDS.index(t);continue
            a[KIND]=KINDS.index(t['kind'])
            name=t.get('crop',t.get('animal'))
            a[ITEM]=ITEMS.index(name) if name else -1
            a[BIRTH]=t.get('planted_day',t.get('placed_day',obs['day']))
            a[YIELD]=t.get('yield_units',0)
            a[WATER]=t.get('watered_today',t.get('fed_today',False))
            a[CARE]=t.get('cared_today',False);a[FERT]=t.get('fertilized_until_day',-1)
            a[NEGLECT]=t.get('consecutive_unwatered',t.get('consecutive_unfed',0))
            a[AVAILABLE]=t.get('fertilizer_available',False)
            a[LIFE]=t.get('max_lifespan_step',-1);a[BONUS]=t.get('pending_care_bonus',0)
    inventory=np.zeros((len(positions),12),np.int64);order=np.full_like(inventory,-1)
    for worker,inv in enumerate(private['inventories']):
        for rank,(name,amount) in enumerate(inv.items()):
            i=ITEMS.index(name);inventory[worker,i]=amount;order[worker,i]=rank
    return dict(tiles=tiles,positions=np.asarray(positions,np.int64),inventory=inventory,order=order,
                shed=np.asarray([private['shed'].get(i,0) for i in ITEMS],np.int64),
                seeds=np.asarray([private['seeds'].get(i,0) for i in CROPS],np.int64),
                day=np.asarray(obs['day'],np.int64),hour=np.asarray(obs['hour'],np.int64),
                step=np.asarray(obs['step'],np.int64),money=np.asarray(farm['money'],np.int64),
                count=np.asarray(len(positions),np.int64),
                prices=np.asarray([obs['market']['prices'].get(i,0) for i in ITEMS],np.int64))


class WorkerState:
    def __init__(self,seed):
        self.seed=seed;self.device=seed['tiles'].device;self.n=len(seed['tiles'])
        self.day=seed['day'].reshape(self.n);self.rows=torch.arange(self.n,device=self.device)
        self.workers=torch.arange(seed['positions'].shape[1],device=self.device)
        official=engine()
        self.moves=torch.tensor([(0,0),(0,-1),(0,1),(1,0),(-1,0)],device=self.device)
        self.action_item=torch.tensor([ITEMS.index(i) if i else -1 for _,i in WORKER],device=self.device)
        self.access=torch.zeros(100,device=self.device,dtype=torch.bool)
        for x,y in official._shed_access_tiles(10):self.access[y*10+x]=True
        rules=np.zeros((12,5),np.int64)
        for i,name in enumerate(ITEMS):
            if name in official.CROPS:
                r=official.CROPS[name]
                rules[i]=[r['first_yield_day'],r['max_yield_day'],r['max_yield'],r['ongoing'],i]
            elif name in official.ANIMALS:
                r=official.ANIMALS[name]
                rules[i]=[r['first_yield_day'],0,0,1,ITEMS.index(r['product'])]
        self.rules=torch.tensor(rules,device=self.device)
        self.reset()
        if os.environ.get('PPO_COMPILE_WORKERS') == '1':
            self.apply=torch.compile(self.apply,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})

    def reset(self):
        for key in ('tiles','positions','inventory','order','shed','seeds'):
            setattr(self,key,self.seed[key].clone())

    def _add(self,inv,order,item,amount,mask):
        rows=self.rows;old=inv[rows,item]
        new_rank=order.max(-1).values+1
        order[rows,item]=torch.where(mask & (order[rows,item]<0),new_rank,order[rows,item])
        inv[rows,item]=old+torch.where(mask,amount,0)

    def _take(self,inv,order,item,amount,mask):
        rows=self.rows;valid=mask & (inv[rows,item]>=amount)
        inv[rows,item]-=torch.where(valid,amount,0)
        order[rows,item]=torch.where(valid & (inv[rows,item]==0),-1,order[rows,item])
        return valid

    def apply(self,worker,index,quantity,active=None):
        rows=self.rows
        active=torch.ones(self.n,device=self.device,dtype=torch.bool) if active is None else active
        pos=self.positions[rows,worker];cell=pos[:,1]*10+pos[:,0]
        tile=self.tiles[rows,cell].clone();inv=self.inventory[rows,worker].clone();order=self.order[rows,worker].clone()
        item=self.action_item[index].clamp_min(0)
        move=(index>=1)&(index<=4)&active
        dest=pos+self.moves[index.clamp(0,4)]
        move &= ((dest>=0)&(dest<10)).all(-1)
        self.positions[rows,worker]=torch.where(move[:,None],dest,pos)
        adjacent=self.access[cell]
        # DROP follows Python inventory insertion order, including capacity loss.
        drop=(index==14)&adjacent&active
        ix=order.argsort(dim=-1,stable=True);amount=inv.gather(1,ix).clamp_min(0)
        room=(100-self.shed.sum(-1)).clamp_min(0)
        before=amount.cumsum(-1)-amount
        take=torch.minimum(amount,(room[:,None]-before).clamp_min(0))
        delivered=torch.zeros_like(inv).scatter(1,ix,take)
        self.shed+=torch.where(drop[:,None],delivered,0)
        inv=torch.where(drop[:,None],0,inv);order=torch.where(drop[:,None],-1,order)

        pickup=(index>=32)&adjacent&(quantity>0)&active
        amount=torch.minimum(quantity,self.shed[rows,item]);pickup &= amount>0
        self.shed[rows,item]-=torch.where(pickup,amount,0)
        self._add(inv,order,item,amount,pickup)

        place=(index>=20)&(index<32)&active
        structure=torch.where(item==9,4,5)
        animal_slot=place&(item>=9)&(tile[:,KIND]==structure)&(tile[:,ITEM]<0)
        placed=self._take(inv,order,item,torch.ones_like(item),animal_slot)
        new=torch.zeros_like(tile);new[:,KIND]=structure;new[:,ITEM]=item;new[:,BIRTH]=self.day
        new[:,FERT]=-1;new[:,LIFE]=-1
        tile=torch.where(placed[:,None],new,tile)
        deposit=place&~animal_slot&adjacent&(quantity>0)
        amount=torch.minimum(quantity,inv[rows,item])
        amount=torch.minimum(amount,(100-self.shed.sum(-1)).clamp_min(0))
        deposit &= amount>0
        self._take(inv,order,item,amount,deposit)
        self.shed[rows,item]+=torch.where(deposit,amount,0)

        owned=active&(tile[:,KIND]!=1)
        plant=(index>=15)&(index<20)&owned&(tile[:,KIND]==0)
        crop=item.clamp(0,4);plant &= self.seeds[rows,crop]>0
        self.seeds[rows,crop]-=plant.long()
        rule=self.rules[crop]
        new=torch.zeros_like(tile);new[:,KIND]=3;new[:,ITEM]=crop;new[:,BIRTH]=self.day
        new[:,YIELD]=1-rule[:,3];new[:,NEGLECT]=1;new[:,FERT]=-1
        new[:,LIFE]=torch.where(rule[:,3]>0,-1,(self.day+rule[:,1]+1)*24)
        tile=torch.where(plant[:,None],new,tile)

        rule=self.rules[tile[:,ITEM].clamp_min(0)]
        water=(index==5)&owned&(tile[:,KIND]==3)&(tile[:,WATER]==0)
        age=self.day-tile[:,BIRTH]
        bonus=water&(rule[:,3]==0)&(age>=(rule[:,1]+1)//2)&(age<=rule[:,1])
        tile[:,YIELD]=torch.where(bonus,torch.minimum(rule[:,2],tile[:,YIELD]+1+(tile[:,FERT]>=self.day)),tile[:,YIELD])
        tile[:,WATER]=torch.where(water,1,tile[:,WATER])
        animal=tile[:,ITEM]>=9
        harvest=(index==6)&owned&(tile[:,YIELD]>0)&(animal|((tile[:,KIND]==3)&(age>=rule[:,0])))
        self._add(inv,order,rule[:,4],tile[:,YIELD],harvest)
        clear=harvest&(tile[:,KIND]==3)&(rule[:,3]==0)
        tile[:,YIELD]=torch.where(harvest,0,tile[:,YIELD])
        fertilizer=torch.full_like(item,8)
        fertilize=self._take(inv,order,fertilizer,torch.ones_like(item),(index==7)&owned&(tile[:,KIND]==3))
        tile[:,FERT]=torch.where(fertilize,torch.maximum(tile[:,FERT],self.day+2),tile[:,FERT])
        clear |= (index==8)&owned&~animal
        empty=torch.zeros_like(tile);empty[:,ITEM]=-1;empty[:,BIRTH]=self.day;empty[:,FERT]=-1;empty[:,LIFE]=-1
        tile=torch.where(clear[:,None],empty,tile)
        build=((index==9)|(index==10))&owned&(tile[:,KIND]==0)
        tile[:,KIND]=torch.where(build,index-5,tile[:,KIND])
        feed=self._take(inv,order,torch.zeros_like(item),torch.ones_like(item),(index==11)&owned&animal&(tile[:,WATER]==0))
        tile[:,WATER]=torch.where(feed,1,tile[:,WATER])
        collect=(index==13)&owned&animal&(tile[:,AVAILABLE]>0)
        self._add(inv,order,fertilizer,torch.ones_like(item),collect)
        tile[:,AVAILABLE]=torch.where(collect,0,tile[:,AVAILABLE])
        tile[:,CARE]=torch.where((index==12)&owned&animal,1,tile[:,CARE])
        self.tiles[rows,cell]=tile;self.inventory[rows,worker]=inv;self.order[rows,worker]=order

    def resolve(self,indices,quantities):
        self.reset()
        count=torch.zeros(self.n,5,device=self.device,dtype=torch.long)
        for index in indices:
            count.scatter_add_(1,(index-15).clamp(0,4)[:,None],((index>=15)&(index<20)).long()[:,None])
        blocked=count>self.seed['seeds']
        for worker,(index,quantity) in enumerate(zip(indices,quantities)):
            plant=(index>=15)&(index<20)
            cancelled=plant&blocked[self.rows,(index-15).clamp(0,4)]
            self.apply(self.workers[worker].expand(self.n),torch.where(cancelled,0,index),quantity)
        return self


class WorkerScenarios:
    """Track all 32 crop-cancellation sets to avoid replaying each prefix.

    Every scenario receives the same selected requests. Only its five PLANT
    cancellation bits differ. Selecting the actual demand mask after each
    request is therefore identical to rebuilding that prefix from scratch.
    """
    def __init__(self,seed):
        self.state=WorkerState(seed);self.n=self.state.n;self.seed=seed
        self.scenarios=WorkerState({k:v.repeat_interleave(32,dim=0) for k,v in seed.items()})
        self.bits=2**torch.arange(5,device=self.state.device)
        self.scenario_id=torch.arange(32,device=self.state.device).repeat(self.n)
        self.reset()

    def reset(self):
        for key,value in self.seed.items():
            self.scenarios.seed[key].copy_(value.repeat_interleave(32,dim=0))
        self.scenarios.reset()
        self.counts=torch.zeros(self.n,5,device=self.state.device,dtype=torch.long)
        self.select()

    def select(self):
        mask=((self.counts>self.seed['seeds'])*self.bits).sum(-1)
        rows=self.state.rows*32+mask
        for key in ('tiles','positions','inventory','order','shed','seeds'):
            setattr(self.state,key,getattr(self.scenarios,key)[rows])

    def append(self,worker,index,quantity):
        plant=(index>=15)&(index<20);crop=(index-15).clamp(0,4)
        self.counts.scatter_add_(1,crop[:,None],plant.long()[:,None])
        ix=index.repeat_interleave(32);amount=quantity.repeat_interleave(32)
        cancelled=plant.repeat_interleave(32)&((self.scenario_id & self.bits[crop].repeat_interleave(32))!=0)
        self.scenarios.apply(self.scenarios.workers[worker].expand(self.n*32),torch.where(cancelled,0,ix),amount)
        self.select()

    def resolve(self,indices,quantities):
        self.reset()
        for worker,(index,quantity) in enumerate(zip(indices,quantities)):
            self.append(worker,index,quantity)
        return self.state
