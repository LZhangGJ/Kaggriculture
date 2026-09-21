"""Exact post-worker market inputs from the GPU provisional resolver."""
import torch
import torch.nn.functional as F
from ppo.gpu_features import scaled


class GPUPostWorker:
    def __init__(self,device):
        from bc_runtime import MARKET,ITEMS
        self.item=torch.tensor([ITEMS.index(i) if i else 0 for _,i in MARKET],device=device)
        self.seedbuy=torch.tensor([op=='BUY_SEED' for op,_ in MARKET],device=device)
        self.animalbuy=torch.tensor([op=='BUY_ANIMAL' for op,_ in MARKET],device=device)
        self.none=torch.tensor([op in ('STOP','NOOP') for op,_ in MARKET],device=device)
        self.static=torch.tensor([10,20,50,100,80,0,0,0,0,300,400,500],device=device,dtype=torch.float64)
        fib=[1,1]
        for _ in range(62):fib.append(fib[-1]+fib[-2])
        self.fib=torch.tensor(fib,device=device,dtype=torch.float64)
        self.land=torch.tensor([1000,2000,4000]+[0]*12,device=device,dtype=torch.float64)
        self.offset=torch.arange(11,device=device)
        self.first=torch.tensor([2,2,8,10,10,0,0,0,0,4,8,6],device=device)

    def __call__(self,state,simulation,seat_indices=None):
        seed=state.seed;n=state.n;device=state.device
        shed,seeds=state.shed,state.seeds
        money,step,count=seed['money'],seed['step'],seed['count']
        prices=seed['prices'];stocks=F.pad(simulation['market_inventory'].repeat_interleave(2,0).long(),(0,3))
        hire_count=simulation['hires_today'].reshape(-1).long()
        land_count=simulation['unlocked_count'].reshape(-1).long()
        if seat_indices is not None:
            stocks=stocks[seat_indices];hire_count=hire_count[seat_indices];land_count=land_count[seat_indices]
        hires=self.fib[hire_count[:,None]+self.offset]
        lands=self.land[land_count[:,None]-1+self.offset]
        base=torch.zeros(n,112,device=device)
        base[:,0]=base[:,3]=scaled(money)
        base[:,5]=1
        base[:,8:20],base[:,56:61]=scaled(shed),scaled(seeds)
        base[:,61:73],base[:,73:85]=scaled(stocks),scaled(prices)
        base[:,85],base[:,87],base[:,91],base[:,95]=1,1,1,1
        base[:,88]=(719-step).double()/719
        base[:,89]=scaled(count)
        costs=prices[:,self.item].double()
        costs=torch.where((self.seedbuy|self.animalbuy)[None,:],self.static[self.item],costs)
        costs[:,self.none]=0
        costs[:,2],costs[:,3]=hires[:,0],lands[:,0]
        present=torch.arange(12,device=device)[None,:].expand(n,-1)<9
        market=dict(base=base,costs=costs,hires=hires,lands=lands,money=money.double(),
            shed=shed.double(),seeds=F.pad(seeds.double(),(0,7)),prices_present=present,
            room=(100-shed.sum(-1)).clamp_min(0).double())
        farm=torch.zeros(n,96,device=device)
        farm[:,0],farm[:,1]=scaled(money),scaled(count)
        farm[:,2]=scaled(hire_count)
        farm[:,3]=land_count.float()/4
        farm[:,4:6]=1;farm[:,7]=scaled(market['room'])
        farm[:,8:20],farm[:,20:32],farm[:,32:37]=scaled(shed),scaled(state.inventory.sum(1)),scaled(seeds)
        tiles=state.tiles;item=tiles[:,:,1];known=item>=0;animal=item>=9
        product=torch.where(animal,item-4,item).clamp_min(0)
        left=((tiles[:,:,2]+self.first[item.clamp_min(0)]-seed['day'][:,None])*24-seed['hour'][:,None]).clamp_min(0)
        ready=tiles[:,:,3]*(left==0)*known
        farm[:,37:49].scatter_add_(1,item.clamp_min(0),known.float())
        farm[:,65:77].scatter_add_(1,product,ready.float())
        remaining=(719-step).clamp_min(0)
        for hi,span in enumerate((24,72,168,720)):
            horizon=remaining.clamp_max(span)
            farm[:,49+hi]=((left<=horizon[:,None])&known).sum(-1)
            demand=torch.zeros(n,device=device)
            for cell in range(100):
                demand=(demand.double()+horizon.double()/24*(1+animal[:,cell].long())*known[:,cell]).float()
            farm[:,53+hi]=demand
            farm[:,57+hi]=count*horizon
        farm[:,37:77]=scaled(farm[:,37:77])
        farm[:,89],farm[:,91],farm[:,92],farm[:,94]=1,1,1,1
        return market,base,farm
