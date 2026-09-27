"""Exact BC observations from simulator tensors, without host observations.

Rows are game-major, seat-minor. Opponent private data never enters a row.
This adapter leaves checkpoint-hashed BC representation code unchanged.
"""
import torch
import torch.nn.functional as F


class GPUHistory:
    """Causal public history. Only chosen requests enter request channels."""
    def __init__(self, games, device):
        from bc_runtime import MARKET, ITEMS
        n = games*2
        self.rows = torch.zeros(n,8,48,device=device)
        self.valid = torch.zeros(n,8,device=device,dtype=torch.bool)
        self.ema = torch.zeros(n,12,device=device,dtype=torch.float64)
        self.variance = torch.zeros_like(self.ema)
        self.count = torch.zeros(n,device=device,dtype=torch.long)
        self.requests = torch.zeros(n,48,device=device)
        self.previous = (torch.zeros(n,12,device=device,dtype=torch.long),
                         torch.zeros(n,12,device=device,dtype=torch.long),
                         torch.zeros(n,device=device,dtype=torch.long))
        self.item = torch.tensor([ITEMS.index(it) if it else 0 for op,it in MARKET],device=device)
        self.sign = torch.tensor([-1 if op=='SELL' else 1 if it else 0 for op,it in MARKET],device=device)
        import long_market as LM  # long-market-v1
        self.lm_c = LM.consts(device); self.lm_state = LM.new_state(n, device)
        self.req_sell = torch.zeros(n, 9, device=device, dtype=torch.float64); self.req_buy = torch.zeros_like(self.req_sell)
        self.shed_prev = torch.zeros_like(self.req_sell)
        self.is_sell = torch.tensor([op=='SELL' for op,it in MARKET],device=device)
        self.is_buyprod = torch.tensor([op=='BUY_PRODUCT' for op,it in MARKET],device=device)
        self.special = torch.tensor([44 if op=='BUY_SEED' and it=='WHEAT' else
            45 if op=='BUY_PRODUCT' and it=='WHEAT' else 46 if op=='BUY_ANIMAL' else
            47 if op in ('HIRE','BUY_LAND') else 0 for op,it in MARKET],device=device)

    def remember(self, indices, quantities, active):
        row = torch.zeros_like(self.requests)
        batch = torch.arange(len(row),device=row.device)
        rs = torch.zeros_like(self.req_sell); rb = torch.zeros_like(self.req_buy)
        for depth in range(indices.shape[0]):
            index, q, valid = indices[depth],quantities[depth],active[depth]
            amount = ((q.double().abs()+1).log1p()/12).float()
            col = 24+self.item[index]
            row[batch,col] += amount*self.sign[index]*valid
            special = self.special[index]
            value = torch.where(special==47,.1,amount)*valid*(special>0)
            row[batch,special] += value
            it9 = self.item[index].clamp_max(8); live = valid & (self.item[index] < 9)
            rs[batch,it9] += torch.where(live & self.is_sell[index], q.double().clamp_min(0), 0.)
            rb[batch,it9] += torch.where(live & self.is_buyprod[index], q.double().clamp_min(0), 0.)
        self.requests.copy_(row)
        self.req_sell.copy_(rs); self.req_buy.copy_(rb)

    def observe(self, state):
        prices = F.pad(state['market_price'].repeat_interleave(2,0).long(),(0,3))
        stocks = F.pad(state['market_inventory'].repeat_interleave(2,0).long(),(0,3))
        cash = state['money'].reshape(-1).long()
        observed=self.count>0
        oldprice,oldstock,oldcash = self.previous
        delta = prices-self.ema
        self.ema.copy_(torch.where(observed[:,None],self.ema+.125*delta,prices.double()))
        self.variance.copy_(torch.where(observed[:,None],.875*(self.variance+.125*delta.square()),0))
        row = self.requests.clone()
        row[:,:12],row[:,12:24] = scaled(prices-oldprice),scaled(stocks-oldstock)
        row[:,36],row[:,37] = (self.count-1).double()/719,1/719
        cashdelta = cash-oldcash
        row[:,38],row[:,39] = scaled(cashdelta),scaled(cashdelta.reshape(-1,2).flip(1).reshape(-1))
        row[:,42:44] = 1
        rows=torch.where((self.count>8)[:,None,None],self.rows.roll(-1,1),self.rows)
        slot=(self.count-1).clamp(0,7)
        index=torch.arange(len(rows),device=rows.device)
        rows[index,slot]=torch.where(observed[:,None],row,rows[index,slot])
        self.rows.copy_(rows)
        self.valid[index,slot]|=observed
        oldprice.copy_(prices);oldstock.copy_(stocks);oldcash.copy_(cash)
        import long_market as LM  # long-market-v1
        shops = state['town_shops'].repeat_interleave(2,0).long()
        shops = torch.where(torch.arange(8,device=shops.device)[None] < state['town_count'].repeat_interleave(2,0).long()[:,None], shops, -1)
        long = LM.step(self.lm_state, state['market_price'].repeat_interleave(2,0), state['market_inventory'].repeat_interleave(2,0),
                       self.shed_prev, self.req_sell, self.req_buy, self.count == 0, shops, self.lm_c)
        self.shed_prev.copy_(state['shed'].reshape(len(self.count), -1)[:, :9].double())
        self.req_sell.zero_(); self.req_buy.zero_()
        self.count.add_(1)
        self.requests.zero_()
        return self.rows,self.valid,self.ema,self.variance,self.count,long


def worker_seed(state):
    """Map simulator storage to the existing exact GPU worker resolver."""
    n = state['money'].numel()
    def own(key): return state[key].reshape(n,*state[key].shape[2:]).long()
    step = state['step'].repeat_interleave(2).long()
    day = step//24
    kind,crop,animal = (own(k).flatten(1) for k in ('tile_kind','tile_crop','tile_animal'))
    iscrop,isan = kind==3,animal>=0
    known = iscrop|isan
    flags = own('tile_flags').flatten(1)
    def field(key): return own(key).flatten(1)
    tiles = torch.stack((kind,torch.where(iscrop,crop,torch.where(isan,animal+9,-1)),
        torch.where(known,field('tile_origin_day'),day[:,None]),field('tile_yield')*known,
        torch.where(iscrop,(flags&1)!=0,isan&((flags&2)!=0)).long(),
        (isan&((flags&4)!=0)).long(),torch.where(iscrop,field('tile_fertilized_until'),-1),
        field('tile_neglect')*known,(isan&((flags&8)!=0)).long(),
        torch.where(iscrop,field('tile_max_lifespan'),-1),field('tile_pending_care')*isan),-1)
    return dict(tiles=tiles,positions=own('unit_pos'),inventory=own('unit_inventory'),
        order=own('unit_inventory_order'),shed=own('shed'),seeds=own('seeds'),day=day,
        hour=step%24,step=step,money=own('money'),count=own('unit_active').sum(-1),
        prices=F.pad(state['market_price'].repeat_interleave(2,0).long(),(0,3)))


def scaled(x):
    x = x.double()
    return (x.sign() * x.abs().log1p() / 12).float()


class GPUFeatures:
    def __init__(self, device):
        self.device = device
        self.xy = torch.tensor([(x, y) for y in range(10) for x in range(10)], device=device)
        self.distance = (self.xy[:, None] - self.xy[None]).abs().sum(-1)
        self.shed = ((self.xy[:, 0] >= 4) & (self.xy[:, 0] <= 5)
                     & (self.xy[:, 1] >= 4) & (self.xy[:, 1] <= 5))
        self.first = torch.tensor([2,2,8,10,10,0,0,0,0,4,8,6], device=device)
        self.cost = torch.tensor([10,20,50,100,80,0,0,0,0,300,400,500], device=device)
        self.rules = torch.zeros(12,8,device=device)
        self.rules[:5] = torch.tensor([[2/30,0,.6,0,0,1,0,1], [2/30,0,.4,0,0,1,0,1],
            [8/30,1/30,.4,1,0,1,0,1], [10/30,2/30,.4,1,0,1,0,1],
            [10/30,0,.6,0,0,1,0,1]],device=device)
        self.rules[9:] = torch.tensor([[4/30,1/30,.4,1,1,0,1,1],
            [8/30,2/30,.6,1,1,0,1,1], [6/30,3/30,.6,1,1,0,1,1]],device=device)

    def __call__(self, s, hist, hv, ema, variance, history_count, long=None):
        games = s['money'].shape[0]
        n = games * 2
        # Canonical public own/opponent order for each seat.
        def sides(x):
            return torch.stack((x, x.flip(1)), dim=1).reshape(n,2,*x.shape[2:])
        def own(x): return x.reshape(n,*x.shape[2:])
        def shared(x): return x.repeat_interleave(2,dim=0)
        def zeros(*shape): return torch.zeros(*shape,device=self.device)
        turn = shared(s['step']).long()
        day, hour = turn // 24, turn % 24
        remaining = (719-turn).clamp_min(0)
        kind = sides(s['tile_kind']).long().flatten(-2)
        crop = sides(s['tile_crop']).long().flatten(-2)
        animal = sides(s['tile_animal']).long().flatten(-2)
        flags = sides(s['tile_flags']).long().flatten(-2)
        iscrop, isan = (kind == 3), (animal >= 0)
        known = iscrop | isan
        asset = torch.where(iscrop, crop, animal+9).clamp(0,11)
        product = torch.where(iscrop, crop, animal+5).clamp(0,11)
        origin = sides(s['tile_origin_day']).long().flatten(-2)
        left = ((origin + self.first[asset] - day[:,None,None])*24-hour[:,None,None]).clamp_min(0)
        yields = sides(s['tile_yield']).long().flatten(-2)
        ready = yields * (left == 0) * known
        prices = F.pad(shared(s['market_price']).long(),(0,3))
        stocks = F.pad(shared(s['market_inventory']).long(),(0,3))
        active = sides(s['unit_active']).bool()
        pos = sides(s['unit_pos']).long()
        count = active.sum(-1)
        cells = (pos[...,1]*10+pos[...,0]).clamp(0,99)
        occupancy = zeros(n,2,100).scatter_add(2,cells,active.float())
        hands = zeros(n,2,100).scatter_add(2,cells[:,:,1:],active[:,:,1:].float())
        b = zeros(n,2,36,100)
        for channel, value in enumerate((kind == 1,kind == 0,kind == 2)):
            b[:,:,channel] = value
        b[:,:,3:8] = (F.one_hot(crop.clamp(0,4),5)*iscrop[...,None]).transpose(-1,-2)
        b[:,:,8], b[:,:,9] = kind == 5, kind == 4
        for channel, aid in ((10,1),(11,2),(12,0)): b[:,:,channel] = animal == aid
        b[:,:,13] = scaled(yields) * known
        b[:,:,14] = (day[:,None,None]-origin).float()/30 * known
        b[:,:,15] = torch.where(iscrop, (flags&1)!=0, ((flags&2)!=0)&isan)
        b[:,:,16] = ((flags&4)!=0)&isan
        b[:,:,17] = (sides(s['tile_fertilized_until']).long().flatten(-2)-day[:,None,None]+1).clamp_min(0).float()/4 * iscrop
        b[:,:,18] = sides(s['tile_neglect']).flatten(-2).float()/10 * known
        b[:,:,19].scatter_(2,cells[:,:,:1],1)
        b[:,:,20] = hands*.1
        b[:,:,21] = ((flags&8)!=0)&isan
        b[:,:,22] = (sides(s['tile_max_lifespan']).long().flatten(-2)-turn[:,None,None]).clamp_min(0).float()/720 * iscrop
        b[:,:,23] = sides(s['tile_pending_care']).flatten(-2).float()/10 * isan
        b[:,:,24] = occupancy*.1
        b[:,:,30] = left.float()/719 * known
        b[:,:,31] = ready > 0
        b[:,:,32] = iscrop & ((flags&1)==0)
        b[:,:,33] = isan & ((flags&2)==0)
        b[:,:,34] = scaled(ready * prices[:,None,:].expand(-1,2,-1).gather(2,product))
        b[:,:,35] = known

        private_inv = own(s['unit_inventory']).long()
        carry = (private_inv * own(s['unit_active'])[:,:,None]).sum(1)
        shed, seeds = own(s['shed']).long(), own(s['seeds']).long()
        farms = zeros(n,2,96)
        farms[:,:,0] = scaled(sides(s['money']))
        farms[:,:,1] = scaled(count)
        farms[:,:,2] = scaled(sides(s['hires_today']))
        farms[:,:,3] = sides(s['unlocked_count']).float()/4
        farms[:,0,4:6] = 1
        farms[:,0,7] = scaled((100-shed.sum(-1)).clamp_min(0))
        farms[:,0,8:20], farms[:,0,20:32], farms[:,0,32:37] = scaled(shed),scaled(carry),scaled(seeds)
        farms[:,:,37:49].scatter_add_(2,asset,known.float())
        farms[:,:,65:77].scatter_add_(2,product,ready.float())
        # Match sequential float32 accumulation used by the BC encoder.
        for hi, span in enumerate((24,72,168,720)):
            horizon = remaining.clamp_max(span)
            farms[:,:,49+hi] = ((left <= horizon[:,None,None]) & known).sum(-1)
            demand = zeros(n,2)
            for cell in range(100):
                demand = (demand.double() + horizon[:,None].double()/24 *
                          (1+isan[:,:,cell].long()) * known[:,:,cell]).float()
            farms[:,:,53+hi] = demand
            farms[:,:,57+hi] = count * horizon[:,None]
        farms[:,:,37:77] = scaled(farms[:,:,37:77])
        farms[:,0,89] = 1
        farms[:,:,91:93] = 1
        farms[:,:,94] = 1

        opportunities = torch.stack((self.shed.expand(n,2,-1),
            (yields>0)&(isan|(known&(left==0))), iscrop&((flags&1)==0),
            isan&((flags&2)==0), isan&((flags&4)==0), isan&((flags&8)!=0),
            iscrop, kind==0, (kind>=2)&~isan, (kind==4)&~isan, (kind==5)&~isan),-1)
        worker_dist = self.distance[cells]
        near = torch.where(opportunities[:,:,None,:,:],worker_dist[:,:,:,:,None],100).amin(-2)
        present = opportunities.any(-2)
        near = torch.where(present[:,:,None,:],near,0)
        w = zeros(n,2,33,64)
        w[:,1,:,0] = 1
        w[:,:,0,1] = 1
        w[:,:,:,2] = scaled(torch.arange(33,device=self.device))
        w[:,:,:,3:5] = pos.float()/9
        w[:,0,:,5] = 1
        w[:,0,:,8:20] = scaled(private_inv)
        w[:,0,:,20] = scaled(private_inv.sum(-1))
        w[:,:,:,21] = near[:,:,:,0].float()/18
        w[:,:,:,33] = 1
        w[:,:,:,34:45] = near.float()/18
        w[:,:,:,45:56] = present[:,:,None,:]
        w[:,0,:,56] = (seeds>0).any(-1)[:,None]
        w[:,0,:,57] = private_inv[:,:,0]>0
        w[:,0,:,58] = private_inv[:,:,8]>0
        w[:,0,:,59] = (private_inv[:,:,:9]>0).any(-1)
        w[:,0,:,60] = (private_inv[:,:,9:]>0).any(-1)
        wi = torch.arange(66,device=self.device)[None,:].expand(n,-1)
        src = torch.where(wi<count[:,0,None],wi,wi-count[:,0,None]+33).clamp_max(65)
        valid = wi<count.sum(-1)[:,None]
        workers = w.flatten(1,2).gather(1,src[:,:,None].expand(-1,-1,64))*valid[:,:,None]

        market = zeros(n,12,64)
        market[:,:,:12] = torch.eye(12,device=self.device)
        market[:,:,12],market[:,:,13] = scaled(prices),scaled(stocks)
        market[:,:,14],market[:,:5,15],market[:,:,16] = scaled(shed),scaled(seeds),scaled(carry)
        market[:,:9,17:19] = 1
        market[:,:,19],market[:,:,20:28] = scaled(self.cost),self.rules
        hc = hv.sum(-1)
        latest = hist.gather(1,(hc-1).clamp_min(0)[:,None,None].expand(-1,1,48))[:,0]
        market[:,:,28:31] = latest[:,:36].reshape(n,3,12).transpose(1,2)
        market[:,:,31] = hist[:,:,24:36].sum(1)
        market[:,:,32],market[:,:,33] = hist[:,:,:12].sum(1),hist[:,:,12:24].sum(1)
        market[:,:,34],market[:,:,35] = scaled(ema),scaled(variance.sqrt())
        market[:,:,36] = history_count.clamp_max(719)[:,None].float()/719
        market[:,:,37] = hist[:,:,24:36].clamp_min(0).sum(1)
        market[:,:,38] = hist[:,:,24:36].clamp_max(0).sum(1)
        market[:,:,39] = hc[:,None].float()/8
        market[:,:,40:43] = 1
        market[:,:,28:44] *= (hc>0)[:,None,None]
        g = zeros(n,64)
        g[:,0],g[:,1],g[:,2],g[:,3] = turn.float()/719,day.float()/30,hour.float()/24,remaining.float()/719
        g[:,4:6] = (turn>0)[:,None].float()/719
        g[:,6],g[:,8] = turn==0,2
        shops = shared(s['town_shops']).long()
        g[:,9:17] = (F.one_hot(shops.clamp_min(0),8)*(shops>=0)[:,:,None]).sum(1).float()/10
        extra = {} if long is None else dict(market_long=long)  # long-market-v1
        return dict(**extra, boards=b.reshape(n,2,36,10,10),global_features=g,farms=farms,
            workers=workers,worker_valid=valid,market=market,programs=zeros(n,16,96),
            program_valid=torch.zeros(n,16,device=self.device,dtype=torch.bool),history=hist,history_valid=hv)
