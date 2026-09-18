"""Canonical requests and compact, causal worker-option features."""
import numpy as np
import torch
from bc_runtime import ITEMS, CROPS, ANIMALS, MARKET, scale
from features_v2 import scaled
from exact_features import directional_opportunities, MOVES
from worker_phase import engine

OPS = ('PASS','NORTH','SOUTH','EAST','WEST','WATER','HARVEST','FERTILIZE','DIG',
       'BUILD_COOP','BUILD_PASTURE','FEED','CARE','COLLECT_FERTILIZER','DROP','PLANT','PLACE','PICKUP')
WORKER = [(op,None) for op in OPS[:15]] + [('PLANT',c) for c in CROPS]
WORKER += [(op,item) for op in ('PLACE','PICKUP') for item in ITEMS]
assert len(WORKER)==44


def needs_quantity(obs, worker, index):
    op,item=WORKER[index]
    if op not in ('PICKUP','PLACE'):return False
    farm=obs['farms'][obs['player']]; pos=([farm['farmer']]+farm['hands'])[worker]
    tile=farm['tiles'][pos[1]][pos[0]]
    if op=='PLACE' and item in ANIMALS and isinstance(tile,dict):
        if tile.get('kind')==engine().ANIMALS[item]['structure'] and 'animal' not in tile:
            return False  # Amount is ignored even when no animal is carried.
    return tuple(pos) in engine()._shed_access_tiles(10)


def worker_target(raw,obs,worker):
    if raw is None or raw==[]:return 0,0,True,False
    if not isinstance(raw,list) or not raw or raw[0] not in OPS:return 0,0,False,False
    op=raw[0];item=raw[1] if op in ('PLANT','PLACE','PICKUP') and len(raw)>1 else None
    if (op,item) not in WORKER:return 0,0,False,False
    index=WORKER.index((op,item));has_q=needs_quantity(obs,worker,index)
    if not has_q:return index,0,True,False
    try:q=int(raw[2]) if len(raw)>2 else 1
    except (ValueError,TypeError,OverflowError):return index,0,False,False
    return index,q,True,True


def worker_wire(index,quantity=None):
    op,item=WORKER[index];out=[op]
    if item is not None:out.append(item)
    if quantity is not None:out.append(int(quantity))
    return out


def market_target(raw):
    try:parsed=engine()._parse_order(raw)
    except (ValueError,TypeError,OverflowError):return 0,0,False
    if parsed is None:return 0,0,False
    op=parsed['type'];item=parsed.get('item')
    q=parsed.get('remaining',0)
    if (op,item) not in MARKET[2:] or (item is not None and q<=0):return 0,0,False
    return MARKET.index((op,item)),q,True


def tile_vector(tile,obs):
    v=np.zeros(24,np.float32)
    if tile=='LOCKED':v[0]=1;return v
    if tile is None:v[1]=1;return v
    v[2]=tile.get('kind')=='WEED'
    crop=tile.get('crop');animal=tile.get('animal')
    if crop in CROPS:v[3+CROPS.index(crop)]=1
    v[8]=tile.get('kind')=='PASTURE';v[9]=tile.get('kind')=='COOP'
    if animal in ('COW','SHEEP','GOOSE'):v[10+('COW','SHEEP','GOOSE').index(animal)]=1
    v[13]=scale(tile.get('yield_units',0));v[14]=(obs['day']-tile.get('planted_day',tile.get('placed_day',obs['day'])))/30
    v[15]=tile.get('watered_today',tile.get('fed_today',False));v[16]=tile.get('cared_today',False)
    v[17]=max(0,tile.get('fertilized_until_day',-1)-obs['day']+1)/4
    v[18]=tile.get('consecutive_unwatered',tile.get('consecutive_unfed',0))/10
    v[21]=tile.get('fertilizer_available',False)
    v[22]=max(0,tile.get('max_lifespan_step',-1)-obs['step'])/720
    v[23]=tile.get('pending_care_bonus',0)/10
    return v


def worker_features(start,provisional,prefix,worker):
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
                bounds=np.asarray(bounds),nav=directional_opportunities(provisional,p,positions[worker]),
                items=item,worker=np.int64(worker),plant_initial=initial.astype(np.float32),plant_count=counts.astype(np.float32))


def expand_worker(data):
    """Expand static templates on-device; cache stores no repeated 44-row matrix."""
    led=data['ledger'];b=len(led);device=led.device
    op=torch.tensor([OPS.index(o) for o,_ in WORKER],device=device)
    it=torch.tensor([ITEMS.index(i) if i else -1 for _,i in WORKER],device=device)
    direction=torch.tensor([{'NORTH':1,'SOUTH':2,'EAST':3,'WEST':4}.get(o,0) for o,_ in WORKER],device=device)
    raw=torch.zeros(b,44,128,device=device,dtype=led.dtype)
    raw[:,torch.arange(44,device=device),op]=1
    raw[:,:,18:30]=torch.nn.functional.one_hot(it.clamp_min(0),12).to(raw.dtype)[None]*(it>=0)[None,:,None]
    raw[:,:,30:54]=data['tiles'][:,0,None];raw[:,:,54:78]=data['tiles'][:,direction]
    raw[:,:,78:111]=data['nav'][:,direction].flatten(-2)
    raw[:,:,111:119]=data['items'][:,it.clamp_min(0)]*(it>=0)[None,:,None]
    raw[:,:,119]=led[:,92,None];raw[:,:,120]=data['bounds'][:,direction]
    crop=it.clamp(0,4);isplant=(op==OPS.index('PLANT'))
    initial=data['plant_initial'][:,crop];count=data['plant_count'][:,crop]+1
    raw[:,:,121]=torch.sign(initial-count)*torch.log1p((initial-count).abs())/12*isplant
    raw[:,:,122]=(count>initial)*isplant
    raw[:,:,123]=((count+led[:,5,None]*32)>initial)*isplant
    raw[:,:,124]=led[:,5,None];raw[:,:,125]=1
    raw[:,:,126]=led[:,6,None];raw[:,:,127]=led[:,7,None]
    source=data['cells'][:,0,None].expand(-1,44);target=data['cells'][:,direction]
    return dict(raw=raw,source=source,target=target,items=it[None].expand(b,-1),
                workers=data['worker'][:,None].expand(-1,44),legal=torch.ones(b,44,device=device,dtype=torch.bool))


def worker_quantity_features(obs,worker,index,quantities):
    op,item=WORKER[index];pr=obs['private'];inv=pr['inventories'][worker]
    q=np.asarray(quantities,dtype=np.float64);room=max(0,100-sum(pr['shed'].values()))
    amount=pr['shed'].get(item,0) if op=='PICKUP' else inv.get(item,0)
    effect=np.minimum(np.maximum(q,0),amount)
    if op=='PLACE':effect=np.minimum(effect,room)
    out=np.zeros((len(q),12),np.float32);out[:,0]=scaled(q);out[:,1]=q/max(1,np.abs(q).max())
    out[:,2]=scaled(effect);out[:,3]=scale(inv.get(item,0));out[:,4]=scale(pr['shed'].get(item,0))
    out[:,5]=scale(room);out[:,6]=scaled(np.maximum(q-amount,0));out[:,7]=op=='PICKUP';out[:,8]=op=='PLACE'
    out[:,9]=q<=0;out[:,10]=effect>0;out[:,11]=1
    return out,np.ones(len(q),bool)
