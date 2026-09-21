"""PPO-only exact feature adapters; checkpoint-hashed BC sources stay unchanged."""
import numpy as np
import torch
from bc_runtime import ITEMS, CROPS, MARKET, scale
from exact_actions import WORKER
from features_v2 import lifecycle, HORIZONS, clock
from worker_phase import engine, resolve_worker_phase


class WorkerPrefix:
    def __init__(self, obs):
        self.original = obs
        self.slots = []
        self._points = None
        self.resolved, self.demand, self.blocked = resolve_worker_phase(obs, [])

    def points(self):
        if self._points is None: self._points = opportunity_map(opportunity_points_fast(self.resolved,self.resolved["player"]))
        return self._points

    def append(self, action):
        self.slots.append(action)
        if len(action) >= 2 and action[0] == 'PLANT':
            crop = action[1]
            self.demand[crop] = self.demand.get(crop, 0) + 1
        blocked = {c for c,n in self.demand.items() if n > self.original['private']['seeds'].get(c,0)}
        if blocked != self.blocked:
            self.resolved, self.demand, self.blocked = resolve_worker_phase(self.original, self.slots)
            self._points = None
        else:
            wire = ['PASS'] if len(action)>=2 and action[0]=='PLANT' and action[1] in blocked else action
            o = self.resolved
            farm=o['farms'][o['player']]; x,y=([farm['farmer']]+farm['hands'])[len(self.slots)-1]
            before=copy.deepcopy(farm['tiles'][y][x])
            engine()._apply_unit_action(o['farms'][o['player']], o['private'], len(self.slots)-1,
                                       wire, 10, o['day'], 24, 100)
            if before != farm['tiles'][y][x]: self._points = None


def own_farm(obs):
    # Exact own-farm subset of encode_exact(encode_v2); no queue in this policy.
    farm=obs['farms'][obs['player']]; pr=obs['private']; f=np.zeros(96,np.float32)
    remaining=max(0,719-clock(obs))
    f[:8]=[scale(farm['money']),scale(len(farm['hands'])+1),scale(farm.get('hires_today',0)),
           len(farm['unlocked_quadrants'])/4,1,1,0,scale(max(0,100-sum(pr['shed'].values())))]
    f[8:20]=[scale(pr['shed'].get(x,0)) for x in ITEMS]
    f[20:32]=[scale(sum(inv.get(x,0) for inv in pr['inventories'])) for x in ITEMS]
    f[32:37]=[scale(pr['seeds'].get(x,0)) for x in CROPS]
    for row in farm['tiles']:
        for t in row:
            left,_,_,known,product=lifecycle(t,obs['day'],obs['hour'])
            if not known: continue
            f[37+ITEMS.index(t.get('crop',t.get('animal')))]+=1
            f[65+ITEMS.index(product)]+=t.get('yield_units',0) if left==0 else 0
            for h,span in enumerate(HORIZONS):
                horizon=min(span,remaining); f[49+h]+=int(left<=horizon)
                f[53+h]+=horizon/24*(1+bool(t.get('animal')))
    for h,span in enumerate(HORIZONS): f[57+h]=(len(farm['hands'])+1)*min(span,remaining)
    f[37:77]=[scale(v) for v in f[37:77]]
    f[61:65]=0; f[89:94]=[1,0,1,1,0]; f[94]=1
    return f


def worker_stats(obs, worker):
    pr=obs['private']; inv=pr['inventories'][worker]; room=max(0,100-sum(pr['shed'].values()))
    return np.asarray([[inv.get(item,0),pr['shed'].get(item,0),room] for _,item in WORKER],np.float64)


def market_stats(ledger):
    o=ledger.obs; pr=o['private']; cash=o['farms'][o['player']]['money']
    room=max(0,100-sum(pr['shed'].values()))
    return np.asarray([[ledger.cost(i) if i>1 and item is not None else 0,
                        (pr['seeds'] if op=='BUY_SEED' else pr['shed']).get(item,0),cash,room]
                       for i,(op,item) in enumerate(MARKET)],np.float64)


class QuantityFeatures:
    def __init__(self,wq,mq,device):
        self.vocab=[torch.tensor(v,device=device,dtype=torch.long) for v in (wq,mq)]
        self.q=[v.double()[None,:] for v in self.vocab]
        self.pickup=torch.tensor([op=='PICKUP' for op,_ in WORKER],device=device)
        self.place=torch.tensor([op=='PLACE' for op,_ in WORKER],device=device)
        self.sell=torch.tensor([op=='SELL' for op,_ in MARKET],device=device)
        self.static=[self.scaled(q) for q in self.q]
        self.normal=[self.q[0]/max(1,max(abs(x) for x in wq)),self.q[1]/max(1,max(mq))]

    @staticmethod
    def scaled(x): return x.sign()*x.abs().log1p()/12

    def __call__(self,phase,index,stats):
        q=self.q[phase]; n=len(index); out=torch.zeros(n,q.shape[1],12,device=q.device,dtype=torch.float64)
        out[:,:,0]=self.static[phase];out[:,:,1]=self.normal[phase]
        if phase==0:
            inv,shed,room=stats.unbind(-1);pickup=self.pickup[index];place=self.place[index]
            amount=torch.where(pickup,shed,inv)[:,None]
            effect=torch.minimum(q.clamp_min(0),amount)
            effect=torch.where(place[:,None],torch.minimum(effect,room[:,None]),effect)
            out[:,:,2]=self.scaled(effect);out[:,:,3]=self.scaled(inv[:,None]);out[:,:,4]=self.scaled(shed[:,None])
            out[:,:,5]=self.scaled(room[:,None]);out[:,:,6]=self.scaled((q-amount).clamp_min(0))
            out[:,:,7]=pickup[:,None];out[:,:,8]=place[:,None];out[:,:,9]=q<=0;out[:,:,10]=effect>0;out[:,:,11]=1
            mask=torch.ones((n,q.shape[1]),device=q.device,dtype=torch.bool)
        else:
            price,available,cash,room=stats.unbind(-1);sell=self.sell[index][:,None];cost=q*price[:,None]
            out[:,:,2]=torch.where(sell,0.,self.scaled(cost));out[:,:,3]=torch.where(sell,self.scaled(cost),0.)
            out[:,:,4]=self.scaled(available[:,None]);out[:,:,5]=self.scaled(q/available.clamp_min(1)[:,None])
            out[:,:,6]=torch.where(sell,0.,self.scaled((cost-cash[:,None]).clamp_min(0)))
            out[:,:,7]=self.scaled((q-available[:,None]).clamp_min(0));out[:,:,8]=self.scaled(room[:,None])
            out[:,:,9]=price[:,None]>0;out[:,:,10]=1;out[:,:,11]=q>0
            mask=(q>0).expand(n,-1)
        return out.float(),mask


from functools import lru_cache
from exact_actions import OPS

@lru_cache(None)
def worker_tables(device):
    op=torch.tensor([OPS.index(o) for o,_ in WORKER],device=device)
    it=torch.tensor([ITEMS.index(i) if i else -1 for _,i in WORKER],device=device)
    direction=torch.tensor([{'NORTH':1,'SOUTH':2,'EAST':3,'WEST':4}.get(o,0) for o,_ in WORKER],device=device)
    return op,it,direction,torch.arange(44,device=device),torch.nn.functional.one_hot(it.clamp_min(0),12),op==OPS.index('PLANT'),it.clamp(0,4)

def expand_worker_cached(data):
    """Expand static templates on-device; cache stores no repeated 44-row matrix."""
    led=data['ledger'];b=len(led);device=led.device
    op,it,direction,rows,onehot,isplant,crop = worker_tables(device)
    raw=torch.zeros(b,44,128,device=device,dtype=led.dtype)
    raw.scatter_(2,op[None,:,None].expand(b,-1,1),1)
    raw[:,:,18:30]=onehot.to(raw.dtype)[None]*(it>=0)[None,:,None]
    raw[:,:,30:54]=data['tiles'][:,0,None];raw[:,:,54:78]=data['tiles'][:,direction]
    raw[:,:,78:111]=data['nav'][:,direction].flatten(-2)
    raw[:,:,111:119]=data['items'][:,it.clamp_min(0)]*(it>=0)[None,:,None]
    raw[:,:,119]=led[:,92,None];raw[:,:,120]=data['bounds'][:,direction]
    initial=data['plant_initial'][:,crop];count=data['plant_count'][:,crop]+1
    raw[:,:,121]=torch.sign(initial-count)*torch.log1p((initial-count).abs())/12*isplant
    raw[:,:,122]=(count>initial)*isplant
    raw[:,:,123]=((count+led[:,5,None]*32)>initial)*isplant
    raw[:,:,124]=led[:,5,None];raw[:,:,125]=1
    raw[:,:,126]=led[:,6,None];raw[:,:,127]=led[:,7,None]
    source=data['cells'][:,0,None].expand(-1,44);target=data['cells'][:,direction]
    return dict(raw=raw,source=source,target=target,items=it[None].expand(b,-1),
                workers=data['worker'][:,None].expand(-1,44),legal=torch.ones(b,44,device=device,dtype=torch.bool))



import copy
from bc_runtime import ANIMALS
from exact_features import OPPORTUNITIES, MOVES, directional_opportunities
from exact_actions import tile_vector, scaled
from features_v2 import encode as encode_v2
from kgrl.commitments.mechanics import nearest_shed

@lru_cache(None)
def shed_points():
    return [(x,y) for y in range(10) for x in range(10) if tuple(nearest_shed((x,y)))==(x,y)]

def opportunity_points_fast(obs, player):
    """Public targets, with separate validity when a target class is absent."""
    points = [[] for _ in OPPORTUNITIES]
    points[0] = list(shed_points())
    # Movement is unobstructed in this engine, including through LOCKED cells.
    # Derive all access tiles from the pinned support function, not an assumed
    # corner/center shed layout.
    for y in range(10):
        for x in range(10):
            tile = obs['farms'][player]['tiles'][y][x]
            flags = [False] * 10
            flags[6] = tile is None
            if isinstance(tile, dict):
                crop, animal = tile.get('crop'), tile.get('animal')
                left, _, _, known, _ = lifecycle(tile, obs['day'], obs['hour'])
                has_animal_slot = 'animal' in tile
                flags[0] = tile.get('yield_units', 0) > 0 and (animal in ANIMALS or known and left == 0)
                flags[1] = crop in CROPS and not tile.get('watered_today', False)
                flags[2] = animal in ANIMALS and not tile.get('fed_today', False)
                flags[3] = animal in ANIMALS and not tile.get('cared_today', False)
                flags[4] = animal in ANIMALS and bool(tile.get('fertilizer_available', False))
                flags[5] = crop in CROPS
                flags[7] = not has_animal_slot
                flags[8] = tile.get('kind') == 'COOP' and not has_animal_slot
                flags[9] = tile.get('kind') == 'PASTURE' and not has_animal_slot
            for index, yes in enumerate(flags, 1):
                if yes:
                    points[index].append((x, y))
    return [np.asarray(p, dtype=np.int16).reshape(-1, 2) for p in points]

@lru_cache(None)
def distance_table():
    cells=np.asarray([(x,y) for y in range(10) for x in range(10)],np.int16)
    return np.abs(cells[:,None]-cells[None]).sum(-1)

def opportunity_map(points):
    distance=np.zeros((100,len(OPPORTUNITIES)),np.int16)
    known=np.asarray([len(p)>0 for p in points])
    table=distance_table()
    for i,p in enumerate(points):
        if len(p):distance[:,i]=table[:,p[:,1]*10+p[:,0]].min(-1)
    return distance,known

def directional_cached(position,field):
    cells=[]
    for dx,dy in MOVES:
        x,y=position[0]+dx,position[1]+dy
        if not (0<=x<10 and 0<=y<10):x,y=position
        cells.append(y*10+x)
    distance,known=field;d=distance[cells]
    result=np.zeros((5,len(OPPORTUNITIES),3),np.float32)
    result[:,:,0]=d/18;result[:,:,1]=(d[0]-d)/18;result[:,:,2]=known
    return result

def encode_exact_fast(obs, history):
    x = encode_v2(obs, [], history)
    # An absent queue is not an empty queue with free capacity.
    x['programs'].fill(0)
    x['program_valid'].fill(False)
    x['boards'][:, 25:30] = 0
    x['global_features'][7] = 0
    x['farms'][:, 6] = 0
    x['farms'][:, 61:65] = 0
    x['farms'][:, 77:89] = 0
    x['farms'][:, 94] = 1
    x['workers'][:, 6:8] = 0
    x['workers'][:, 22:33] = 0
    # Reuse previously empty worker columns for causal navigation context.
    index = 0
    for player in (obs['player'], 1 - obs['player']):
        farm = obs['farms'][player]
        points = opportunity_map(opportunity_points_fast(obs, player))
        for worker, position in enumerate([farm['farmer'], *farm['hands']]):
            near = directional_cached(position, points)[0]
            x['workers'][index, 34:45] = near[:, 0]
            x['workers'][index, 45:56] = near[:, 2]
            if player == obs['player']:
                inv = obs['private']['inventories'][worker]
                x['workers'][index, 56:61] = [
                    any(obs['private']['seeds'].get(c, 0) > 0 for c in CROPS),
                    inv.get('WHEAT', 0) > 0, inv.get('FERTILIZER', 0) > 0,
                    any(inv.get(item, 0) > 0 for item in ITEMS[:9]),
                    any(inv.get(item, 0) > 0 for item in ANIMALS)]
            index += 1
    return x
def worker_features_fast(start,provisional,prefix,worker,points):
    p=start['player'];farm=provisional['farms'][p];pr=provisional['private']
    positions=[farm['farmer']]+farm['hands'];x,y=positions[worker];inv=pr['inventories'][worker]
    counts=np.zeros(5)
    for a in prefix:
        if isinstance(a,list) and len(a)>1 and a[0]=='PLANT' and a[1] in CROPS:counts[CROPS.index(a[1])]+=1
    initial=np.array([start['private']['seeds'].get(c,0) for c in CROPS]);remaining=len(positions)-worker-1
    ledger=np.zeros(128,np.float32)
    ledger[:8]=[x/9,y/9,worker==0,worker/32,len(positions)/32,remaining/32,
                (x,y) in engine()._shed_access_tiles(10),scale(max(0,100-sum(pr['shed'].values())))]
    ledger[8:20]=scaled([inv.get(i,0) for i in ITEMS]);ledger[20:32]=scaled([pr['shed'].get(i,0) for i in ITEMS])
    ledger[32:37]=scaled([pr['seeds'].get(c,0) for c in CROPS]);ledger[37:61]=tile_vector(farm['tiles'][y][x],provisional)
    ledger[61:66]=scaled(initial);ledger[66:71]=scaled(counts);ledger[71:76]=scaled(initial-counts)
    ledger[76:81]=counts>initial;ledger[81:86]=counts+remaining<=initial
    ledger[86:91]=(counts<=initial)&(counts+remaining>initial)
    ledger[91:96]=[1,(719-start['step'])/719,start['day']/30,start['hour']/24,scale(farm['money'])]
    tiles=[];cells=[];bounds=[]
    for dx,dy in MOVES:
        nx,ny=x+dx,y+dy;valid=0<=nx<10 and 0<=ny<10
        if not valid:nx,ny=x,y
        cells.append(ny*10+nx);bounds.append(valid);tiles.append(tile_vector(farm['tiles'][ny][nx],provisional))
    item=np.zeros((12,8),np.float32)
    for i,name in enumerate(ITEMS):
        rule=engine().CROPS.get(name) or engine().ANIMALS.get(name)
        item[i,:4]=[scale(inv.get(name,0)),scale(pr['shed'].get(name,0)),scale(pr['seeds'].get(name,0)),scale(provisional['market']['prices'].get(name,0))]
        if rule:item[i,4:]=[scale(rule.get('seed',rule.get('cost',0))),rule['first_yield_day']/30,
                           scale(rule.get('max_yield',rule.get('max_held',0))),rule.get('ongoing',name in ANIMALS)]
    return dict(ledger=ledger,tiles=np.asarray(tiles),cells=np.asarray(cells,dtype=np.int64),
                bounds=np.asarray(bounds),nav=directional_cached(positions[worker],points),
                items=item,worker=np.int64(worker),plant_initial=initial.astype(np.float32),plant_count=counts.astype(np.float32))
