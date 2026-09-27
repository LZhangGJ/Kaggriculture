"""Causal FeatureSpecV2, shared candidate admission, and ordered request ledger.

Exact facts, hypothetical reservations, and price-based estimates occupy separate
fields. Market inventory is NOT a stock limit in engine 1.32.7. Market fills are
unknown until both players' orders settle; projected fills never enter job masks.
"""
from copy import deepcopy
from collections import deque
import numpy as np
from bc_runtime import encode as encode_v1, ITEMS, CROPS, ANIMALS, MARKET, SHOPS, JOB_OPS, job_key, scale
from bc_policy import job_catalog
from kgrl.commitments.mechanics import CROPS as CROP_RULES, ANIMALS as ANIMAL_RULES, distance, nearest_shed, UnitLedger
from kgrl.contracts import Command

SCHEMA = 'macro-bc-v2'
ENGINE = '1.32.7'
JOBS, BASE_JOB = job_catalog()
JOB_INDEX = {job_key(j): i for i, j in enumerate(JOBS) if j}
HORIZONS = (24, 72, 168, 720)
JOB_XY = np.array([(j['x'],j['y']) for j in JOBS[1:]])
JOB_ITEMS = np.array([ITEMS.index(j['item']) if j['item'] else -1 for j in JOBS[1:]])
JOB_TYPES = np.repeat(np.arange(11),100)
JOB_COSTS = np.array([CROP_RULES[j['item']][0] if j['item'] in CROPS else ANIMAL_RULES[j['item']][0] if j['item'] in ANIMALS else 0 for j in JOBS[1:]])
JOB_MATURITY = np.array([CROP_RULES[j['item']][1] if j['item'] in CROPS else ANIMAL_RULES[j['item']][2] if j['item'] in ANIMALS else 0 for j in JOBS[1:]])


def scaled(values):
    v=np.asarray(values);return np.sign(v)*np.log1p(np.abs(v))/12

# Every occupied slice below has a fixed meaning; unused trailing columns stay 0.
# Kinds: observed, exact-derived, executor-state, estimate, unknown-with-mask.
REGISTRY = {
 'boards': {'shape': [2,36,10,10], '0:24': 'v1 causal board channels',
   '24:36': ['worker_count','assigned_count','pending_count','deadline_fraction',
   'progress_known_value','estimated_blocked','maturity_turns_fraction','harvest_ready',
             'water_due','feed_due','ready_value_at_current_price','lifecycle_known']},
 'global_features': {'shape':[64], '0:9':['turn','day','hour','remaining','elapsed_observation',
   'elapsed_decision','reset','queue_free','schema_version'], '9:17':'public shop counts'},
 'farms': {'shape':[2,96], '0:8':'cash,workers,hires,land,own,private_known,queue_free,shed_room',
   '8:20':'shed', '20:32':'carried', '32:37':'seeds', '37:49':'asset counts',
   '49:53':'conditional first-maturity counts by horizon', '53:57':'care/feed demand by horizon',
   '57:61':'labor supply by horizon', '61:65':'queued travel/work estimate by horizon',
   '65:77':'ready output quantities', '77:89':'reserved job inputs', '89:94':'unknown flags and projected values'},
 'workers': {'shape':['W',64], '0:8':'side,farmer,id,x,y,inventory_known,assignment_known,assignment_estimate',
   '8:20':'individual inventory', '20:25':'total carried,shed distance,target distance,target x,target y',
   '25:30':'job operation', '30:34':'available,assignment estimate,travel estimate,capacity unbounded'},
 'market': {'shape':[12,64], '0:12':'item one-hot', '12:20':'price,inventory,shed,seeds,carried,price_known,inventory_known,static_purchase_cost',
   '20:28':'first maturity,interval,maximum yield,ongoing,animal,seed,feed needs,known',
   '28:40':'price/stock changes and past own requested buy/sell counts', '40:44':'public-history validity,request-only flags'},
 'programs': {'shape':[16,96], '0:20':'v1 program identity/location/deadline',
   '20:32':'status,estimated resource blocker,estimated target blocker,travel,estimated worker,age,unknown progress,dependency index,assignment/progress known,estimate flag,reserved zero',
   '32:44':'synthetic material requirements (not observed teacher reservations)', '44:56':'resource shortfall',
   '56:64':'maturity,labor,care/feed,remaining,estimated value and known flags','64:68':'teacher reservation/assignment/blocker unknown, synthetic requirements valid'},
 'history': {'shape':[8,48], '0:12':'past price deltas','12:24':'past stock deltas',
   '24:36':'signed request presence and magnitude, buy minus sell','36:44':'turn,elapsed,cash changes,public assets,request valid,observation valid'},
 'candidate': {'shape':['N',128], '0:33':'stable v1 action identity/type/target',
   '33:69':'target board36', '69:81':'resource/economic/phase fields',
   '81:93':'material vector','93:105':'shortfall vector','105:115':'current item/queue/ledger fields',
   '115:117':'job current-price maximum-yield value estimate and replacement input cost; market phase/known flags',
   '117:128':'legality/admission/known/estimate/dependency flags'},
 'ledger': {'shape':[112], '0:8':'cash observed,request spend estimate,request income estimate,scenario cash,queue slots,order slots,hires requested,land requested',
   '8:20':'observed shed', '20:32':'requested buys', '32:44':'requested sells', '44:56':'reserved job inputs',
   '56:61':'observed seeds','61:73':'market inventory','73:85':'current prices',
   '85:96':'projection/unknown flags and season/labor context', '96:101':'seed requests, distinct from product requests',
   '101:106':'synthetic seed requirements', '106:112':'reserved'},
 'quantity': {'shape':['Q',12], 'fields':['quantity','quantity_fraction','quoted_cost','quoted_value',
   'available','requested_fraction','cash_shortfall','input_shortfall','shed_room','price_known','fill_uncertain','positive']},
 'normalization': 'counts/currency signed log1p(abs(x))/12; time/719; day/30; coordinates/9; flags exact; remaining reserved columns zero',
 'missingness': 'Opponent private inventory/assignments unavailable. Unknown slots are zero with known flag false. No inferred hidden inventory.',
}


def clock(obs):
    return int(obs.get('step', obs['day'] * 24 + obs['hour']))


def lifecycle(tile, day, hour):
    """Earliest rule-based maturity, not a promise of future care or yield."""
    if not isinstance(tile, dict): return (0,0,0,False,None)
    crop, animal = tile.get('crop'), tile.get('animal')
    if crop in CROP_RULES:
        _, first, _, maximum, ongoing = CROP_RULES[crop]
        left = max(0, (tile['planted_day'] + first - day) * 24 - hour)
        return left, maximum, int(ongoing), True, crop
    if animal in ANIMAL_RULES:
        _, _, first, product = ANIMAL_RULES[animal]
        left = max(0, (tile['placed_day'] + first - day) * 24 - hour)
        return left, {'GOOSE':4,'COW':6,'SHEEP':6}[animal], 1, True, product
    return (0,0,0,False,None)


def completed(job, obs):
    t = obs['farms'][obs['player']]['tiles'][job['y']][job['x']]
    op = job['op']
    return (op == 'DIG' and t is None or isinstance(t, dict) and (
        op == 'PLANT' and t.get('crop') == job['item'] or
        op.startswith('BUILD_') and t.get('kind') == op[6:] or
        op == 'PLACE' and t.get('animal') == job['item']))


def prune(pending, obs):
    return [j for j in pending if j['deadline'] >= clock(obs) and not completed(j, obs)]


def admissibility(job, obs, pending):
    """Queue admission is broader than immediate execution, with one dependency.

    Supplies can arrive later. Occupied tiles require an earlier queued DIG;
    PLACE requires the correct existing or earlier queued structure. Never ROI.
    """
    if len(pending) >= 16: return False, False, -1
    if any(job_key(j) == job_key(job) for j in pending): return False, False, -1
    tile = obs['farms'][obs['player']]['tiles'][job['y']][job['x']]
    if tile == 'LOCKED': return False, False, -1
    dep = -1
    for i,j in enumerate(pending):
        if (j['x'],j['y']) == (job['x'],job['y']): dep = i
    prerequisite = pending[dep]['op'] if dep >= 0 else None
    op, item = job['op'], job['item']
    if op == 'DIG': ok = tile is not None and not (isinstance(tile,dict) and tile.get('animal')); queued = ok and dep < 0
    elif op in ('PLANT','BUILD_COOP','BUILD_PASTURE'):
        ok = tile is None; queued = (ok and dep < 0) or prerequisite == 'DIG'
    elif op == 'PLACE':
        structure = ANIMAL_RULES[item][1]
        ok = isinstance(tile,dict) and tile.get('kind') == structure and not tile.get('animal')
        queued = (ok and dep < 0) or prerequisite == 'BUILD_' + structure
    else: return False, False, -1
    farm=obs['farms'][obs['player']];positions=[farm['farmer']]+farm['hands']
    at=[i for i,pos in enumerate(positions) if tuple(pos)==(job['x'],job['y'])]
    ok=ok and bool(at)
    if op == 'PLANT': ok = ok and obs['private']['seeds'].get(item,0)>0
    if op == 'PLACE': ok = ok and any(obs['private']['inventories'][i].get(item,0)>0 for i in at)
    return bool(ok), bool(queued), dep


class PublicHistory:
    def __init__(self):
        self.rows = deque(maxlen=8); self.previous = None; self.requests = []
        self.ema = np.zeros(12); self.variance = np.zeros(12); self.count = 0

    def observe(self, obs):
        prices=np.array([obs['market']['prices'].get(item,0) for item in ITEMS])
        if self.count:
            delta=prices-self.ema;self.ema+=.125*delta
            self.variance=.875*(self.variance+.125*delta*delta)
        else:self.ema=prices.astype(float)
        self.count+=1
        if self.previous is not None:
            old = self.previous
            if clock(obs) != clock(old)+1: raise ValueError('History must advance one observed turn')
            row = np.zeros(48,np.float32)
            for i,item in enumerate(ITEMS):
                row[i] = scale(obs['market']['prices'].get(item,0)-old['market']['prices'].get(item,0))
                row[12+i] = scale(obs['market']['inventory'].get(item,0)-old['market']['inventory'].get(item,0))
            for order in self.requests:
                if (len(order)>2 and order[0] in ('SELL','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL')
                        and order[1] in ITEMS and type(order[2]) is int):
                    row[24+ITEMS.index(order[1])] += scale(abs(order[2])+1) * (-1 if order[0]=='SELL' else 1)
                    if order[0]=='BUY_SEED' and order[1]=='WHEAT':row[44]+=scale(abs(order[2])+1)
                    if order[0]=='BUY_PRODUCT' and order[1]=='WHEAT':row[45]+=scale(abs(order[2])+1)
                    if order[0]=='BUY_ANIMAL':row[46]+=scale(abs(order[2])+1)
                elif order and order[0] in ('HIRE','BUY_LAND'):row[47]+=.1
            p = obs['player']; row[36:40] = [clock(old)/719,1/719,
                scale(obs['farms'][p]['money']-old['farms'][p]['money']),
                scale(obs['farms'][1-p]['money']-old['farms'][1-p]['money'])]
            row[42:44] = [1,1]; self.rows.append(row)
        self.previous = obs; self.requests = []

    def arrays(self):
        a = np.zeros((8,48),np.float32); valid = np.zeros(8,bool)
        if self.rows: a[:len(self.rows)] = list(self.rows); valid[:len(self.rows)] = True
        return a,valid


class RequestLedger:
    def __init__(self, obs, pending):
        self.source_obs = obs; self.obs = obs; self.pending = list(pending); self.orders = []
        self.buys = np.zeros(12); self.sells = np.zeros(12)
        self.seed_buys = np.zeros(5)
        self.spend = 0.; self.income = 0.; self.hires = 0; self.lands = 0
        self.worker_projected = False
        self.projection_exact = False

    def reserved(self):
        v = np.zeros(12)
        for j in self.pending:
            if j['op'] in ('PLANT','PLACE'): v[ITEMS.index(j['item'])] += 1
        return v

    def vector(self):
        o = self.obs; f = o['farms'][o['player']]; pr=o['private']; out=np.zeros(112,np.float32)
        out[:8] = [scale(f['money']),scale(self.spend),scale(self.income),scale(f['money']-self.spend+self.income),
                   (16-len(self.pending))/16,(10-len(self.orders))/10,self.hires/20,self.lands/3]
        for i,item in enumerate(ITEMS):
            out[8+i]=scale(pr['shed'].get(item,0));out[20+i]=scale(self.buys[i]);out[32+i]=scale(self.sells[i])
            out[44+i]=scale(self.reserved()[i]);out[61+i]=scale(o['market']['inventory'].get(item,0))
            out[73+i]=scale(o['market']['prices'].get(item,0))
        out[56:61]=[scale(pr['seeds'].get(x,0)) for x in CROPS]
        out[85:92]=[self.worker_projected, bool(self.orders), 1, (719-clock(o))/719,
                   scale(1+len(f['hands'])), bool(self.orders), 1]
        out[94:96]=[self.worker_projected and not self.projection_exact,self.projection_exact]
        out[96:101]=scaled(self.seed_buys)
        out[101:106]=scaled(self.reserved()[:5])
        return out

    def cost(self, index):
        op,item=MARKET[index]; f=self.obs['farms'][self.obs['player']]
        if op=='BUY_SEED': return CROP_RULES[item][0]
        if op=='BUY_ANIMAL': return ANIMAL_RULES[item][0]
        if op=='HIRE':
            a,b=1,1
            for _ in range(f.get('hires_today',0)+self.hires): a,b=b,a+b
            return a
        if op=='BUY_LAND':
            i=len(f['unlocked_quadrants'])-1+self.lands
            return (1000,2000,4000)[i] if i<3 else 0
        return self.obs['market']['prices'].get(item,0)

    def add_job(self, job, inferred=False):
        if not admissibility(job,self.obs,self.pending)[1]: raise ValueError('Job not admissible')
        self.pending.append(dict(job,created=clock(self.obs),deadline=clock(self.obs)+24,inferred=inferred))

    def add_order(self,index,quantity):
        op,item=MARKET[index]
        if op=='STOP':return
        order=[] if op=='NOOP' else [op] if item is None else [op,item,int(quantity)]
        self.orders.append(order)
        if item:
            i=ITEMS.index(item)
            if op=='SELL': self.sells[i]+=quantity;self.income+=self.cost(index)*quantity
            else:
                if op=='BUY_SEED':self.seed_buys[CROPS.index(item)]+=quantity
                else:self.buys[i]+=quantity
                self.spend+=self.cost(index)*quantity
        elif op=='HIRE':self.spend+=self.cost(index);self.hires+=1
        elif op=='BUY_LAND':self.spend+=self.cost(index);self.lands+=1

    def project_workers(self,action,exact=True):
        u=UnitLedger(self.obs)
        supported={'PASS','NORTH','SOUTH','EAST','WEST','PLANT','WATER','HARVEST',
                   'FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','PLACE','FEED',
                   'CARE','COLLECT_FERTILIZER','DROP','PICKUP'}
        for i,a in enumerate([action['farmer']]+action['hands']):
            if not a or a[0] not in supported:
                exact=False
                continue
            if len(a)>2 and (type(a[2]) is not int or a[2]<=0):
                exact=False
                continue
            u.apply(i,Command(a[0],item=a[1] if len(a)>1 else None,quantity=a[2] if len(a)>2 else None))
        o=dict(self.obs);o['farms']=list(o['farms']);o['farms'][o['player']]=u.farm;o['private']=u.private
        self.obs=o;self.worker_projected=True;self.projection_exact=exact
        self.pending=prune(self.pending,self.obs)


def worker_rows(obs,pending):
    rows=[]; own_positions=[]
    for side,p in enumerate((obs['player'],1-obs['player'])):
        farm=obs['farms'][p];positions=[farm['farmer']]+farm['hands']
        if side==0:own_positions=positions
        for unit,pos in enumerate(positions):
            v=np.zeros(64,np.float32);v[:6]=[side,unit==0,scale(unit),pos[0]/9,pos[1]/9,side==0]
            if side==0:
                inv=obs['private']['inventories'][unit]
                v[8:20]=[scale(inv.get(x,0)) for x in ITEMS];v[20]=scale(sum(inv.values()))
            v[21]=distance(pos,nearest_shed(pos))/18;v[33]=1
            rows.append(v)
    return np.asarray(rows),own_positions


def encode(obs,pending,history):
    old=encode_v1(obs,pending); out={};b=np.zeros((2,36,10,10),np.float32);b[:,:24]=old['boards']
    workers,positions=worker_rows(obs,pending);farms=np.zeros((2,96),np.float32)
    turn=clock(obs);remaining=max(0,719-turn);pr=obs['private'];reserved=RequestLedger(obs,pending).reserved()
    market=np.zeros((12,64),np.float32);market[:,:16]=old['market']
    carry={item:sum(inv.get(item,0) for inv in pr['inventories']) for item in ITEMS}
    for side,p in enumerate((obs['player'],1-obs['player'])):
        farm=obs['farms'][p];f=farms[side]
        f[:8]=[scale(farm['money']),scale(len(farm['hands'])+1),scale(farm.get('hires_today',0)),
               len(farm['unlocked_quadrants'])/4,side==0,side==0,(16-len(pending))/16 if side==0 else 0,
               scale(max(0,100-sum(pr['shed'].values()))) if side==0 else 0]
        if side==0:
            f[8:20]=[scale(pr['shed'].get(x,0)) for x in ITEMS];f[20:32]=[scale(carry[x]) for x in ITEMS]
            f[32:37]=[scale(pr['seeds'].get(x,0)) for x in CROPS];f[77:89]=[scale(v) for v in reserved]
        for pos in [farm['farmer']]+farm['hands']:b[side,24,pos[1],pos[0]]+=.1
        for y,row in enumerate(farm['tiles']):
            for x,t in enumerate(row):
                left,maxyield,ongoing,known,product=lifecycle(t,obs['day'],obs['hour'])
                if not known:continue
                asset=t.get('crop',t.get('animal'));f[37+ITEMS.index(asset)]+=1
                ready=t.get('yield_units',0) if left==0 else 0
                b[side,30:36,y,x]=[left/719,ready>0, bool(t.get('crop') and not t.get('watered_today')),
                    bool(t.get('animal') and not t.get('fed_today')),scale(ready*obs['market']['prices'].get(product,0)),1]
                f[65+ITEMS.index(product)]+=ready
                for h,span in enumerate(HORIZONS):
                    horizon=min(span,remaining);f[49+h]+=int(left<=horizon)
                    f[53+h]+=horizon/24*(1+bool(t.get('animal')))
        for h,span in enumerate(HORIZONS): f[57+h]=(len(farm['hands'])+1)*min(span,remaining)
        f[37:77]=[scale(v) for v in f[37:77]];f[89:94]=[side==0,0,1,1,0]
    programs=np.zeros((16,96),np.float32);programs[:,:32]=old['programs'];pv=old['program_valid']
    assigned=set()
    for i,j in enumerate(pending):
        x,y=j['x'],j['y'];v=programs[i];near=sorted(range(len(positions)),key=lambda k:distance(positions[k],(x,y)))
        worker=next((k for k in near if k not in assigned),None)
        travel=distance(positions[worker],(x,y)) if worker is not None else 0
        now,admit,dep=admissibility(j,obs,pending[:i])
        item=j.get('item');material=np.zeros(12)
        if j['op'] in ('PLANT','PLACE'):material[ITEMS.index(item)]=1
        supply=pr['seeds'].get(item,0) if j['op']=='PLANT' else pr['shed'].get(item,0)+carry.get(item,0)
        short=max(0,material.sum()-supply)
        v[20:32]=[1,short>0,not now and not short,travel/18,scale(worker or 0),
            (turn-j.get('created',turn))/24,0, (dep+1)/16,0,0,1,0]
        v[32:44]=material;v[44:56]=material*short
        maturity=CROP_RULES[item][1] if item in CROPS else ANIMAL_RULES[item][2] if item in ANIMALS else 0
        v[56:64]=[maturity/30,scale(travel+1),item in CROPS,item in ANIMALS,remaining/719,0,1,1]
        # These are our queue's hypothetical material requirements and travel
        # estimates, never recovered teacher reservations or assignments.
        v[64:68]=[0,0,0,1]
        for hi,span in enumerate(HORIZONS):
            farms[0,61+hi]+=scale(min(travel+1,min(span,remaining)))
        b[0,26,y,x]+=.1;b[0,27,y,x]=max(b[0,27,y,x],max(0,j['deadline']-turn)/24)
        b[0,28,y,x]=v[26];b[0,29,y,x]=short>0 or not now
        if worker is not None:
            assigned.add(worker);b[0,25,y,x]+=.1
            workers[worker,7]=1;workers[worker,22:25]=[travel/18,x/9,y/9]
            workers[worker,25+JOB_OPS.index(j['op'])]=1;workers[worker,31:33]=1
    workers[:len(positions),30]=[i not in assigned for i in range(len(positions))]
    hist,hv=history.arrays()
    for i,item in enumerate(ITEMS):
        market[i,16:20]=[scale(carry[item]),item in obs['market']['prices'],item in obs['market']['inventory'],
            scale(CROP_RULES[item][0] if item in CROPS else ANIMAL_RULES[item][0] if item in ANIMALS else 0)]
        if item in CROPS:
            _,first,latest,maximum,ongoing=CROP_RULES[item]
            market[i,20:28]=[first/30,({'TOMATO':1,'STRAWBERRY':2}.get(item,0))/30,maximum/10,ongoing,0,1,0,1]
        if item in ANIMALS:
            market[i,20:28]=[ANIMAL_RULES[item][2]/30,{'GOOSE':1,'COW':2,'SHEEP':3}[item]/30,
                {'GOOSE':4,'COW':6,'SHEEP':6}[item]/10,1,1,0,1,1]
        if hv.any():
            latest=hist[np.where(hv)[0][-1]];market[i,28:32]=[latest[i],latest[12+i],latest[24+i],hist[hv,24+i].sum()]
            market[i,40:44]=[1,1,1,0]
            market[i,32:40]=[hist[hv,i].sum(),hist[hv,12+i].sum(),scale(history.ema[i]),
                scale(np.sqrt(history.variance[i])),min(history.count,719)/719,
                np.maximum(hist[hv,24+i],0).sum(),np.minimum(hist[hv,24+i],0).sum(),hv.sum()/8]
    g=np.zeros(64,np.float32);g[:9]=[turn/719,obs['day']/30,obs['hour']/24,remaining/719,
        int(turn>0)/719,int(turn>0)/719,turn==0,(16-len(pending))/16,2]
    g[9:17]=[obs.get('town',{}).get('unlocked_shops',[]).count(s)/10 for s in SHOPS]
    out.update(boards=b,global_features=g,farms=farms,workers=workers,worker_valid=np.ones(len(workers),bool),
               market=market,programs=programs,program_valid=pv,history=hist,history_valid=hv)
    return out


def candidates(ledger, encoded, phase):
    if phase==0:return job_candidates(ledger,encoded)
    obs=ledger.obs;pending=ledger.pending;farm=obs['farms'][obs['player']];pr=obs['private']
    positions=[farm['farmer']]+farm['hands'];n=len(JOBS) if phase==0 else len(MARKET)
    raw=np.zeros((n,128),np.float32);refs=np.full((n,5),-1,np.int64);relations=np.zeros((n,8),np.float32)
    legal=np.zeros(n,bool);admit=np.zeros(n,bool);legal[0]=admit[0]=True
    if phase==0:raw[:,:33]=BASE_JOB[:,:33]
    reserved=ledger.reserved()
    for i in range(1,n):
        item=None;cost=0;dep=-1;material=np.zeros(12);short=np.zeros(12);travel=0;first=0
        if phase==0:
            j=JOBS[i];op,item=j['op'],j['item'];x,y=j['x'],j['y']
            legal[i],admit[i],dep=admissibility(j,obs,pending)
            raw[i,33:69]=encoded['boards'][0,:,y,x];refs[i,0]=y*10+x
            near=sorted(range(len(positions)),key=lambda k:distance(positions[k],(x,y)))[:2]
            for k,w in enumerate(near):refs[i,2+k]=w;relations[i,k]=distance(positions[w],(x,y))/18
            travel=relations[i,0]*18
            if item:
                material[ITEMS.index(item)]=1
                supply=pr['seeds'].get(item,0) if op=='PLANT' else pr['shed'].get(item,0)+sum(v.get(item,0) for v in pr['inventories'])
                short[ITEMS.index(item)]=max(0,1+reserved[ITEMS.index(item)]-supply)
                cost=CROP_RULES[item][0] if item in CROPS else ANIMAL_RULES[item][0]
                first=CROP_RULES[item][1] if item in CROPS else ANIMAL_RULES[item][2]
                for k,w in enumerate(near):relations[i,2+k]=scale(pr['inventories'][w].get(item,0))
            if dep>=0:
                prior=pending[dep]
                for pi,pv in enumerate(encoded['programs']):
                    if encoded['program_valid'][pi] and pv[JOB_OPS.index(prior['op'])] and np.allclose(pv[17:19],[prior['x']/9,prior['y']/9]):
                        refs[i,4]=pi;break
            relations[i,4:8]=[len(positions)/20,dep>=0,1,1]
        else:
            op,item=MARKET[i];raw[i,('STOP','NOOP','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL').index(op)]=1
            if item:raw[i,8+ITEMS.index(item)]=1
            raw[i,32]=0;raw[i,63]=1;cost=ledger.cost(i)
            # A syntactically valid request can fill conditionally or partially.
            # No hard affordability/stock cap is imposed on valid requests.
            legal[i]=admit[i]=len(ledger.orders)<10
            # Engine parsing admits these requests even when they fill zero.
            # Execution feasibility remains a feature, never a grammar mask.
        if item:
            ii=ITEMS.index(item);refs[i,1]=ii
            raw[i,105:113]=[scale(pr['shed'].get(item,0)),scale(pr['seeds'].get(item,0)),scale(reserved[ii]),
                scale(ledger.seed_buys[CROPS.index(item)] if op=='BUY_SEED' else ledger.buys[ii]),scale(ledger.sells[ii]),scale(obs['market']['inventory'].get(item,0)),
                scale(obs['market']['prices'].get(item,0)),item in obs['market']['prices']]
        raw[i,69:81]=[scale(cost),scale(travel+1),first/30,(719-clock(obs))/719,
            scale(farm['money']),scale(max(0,cost-farm['money'])),scale(short.sum()),(16-len(pending))/16,
            len(ledger.orders)/10,phase,scale(ledger.spend),scale(ledger.income)]
        raw[i,81:93]=material;raw[i,93:105]=short
        raw[i,113:117]=[scale(farm['money']-ledger.spend+ledger.income),dep>=0,phase==1,1]
        raw[i,117:128]=[legal[i],admit[i],1,phase==1,1,dep>=0,(dep+1)/16,1,phase==0,
                        ledger.worker_projected,ledger.projection_exact]
    return dict(raw=raw,refs=refs,relations=relations,legal_now=legal,admissible=admit)


def job_candidates(ledger,encoded):
    """Vectorized over 1,100 targets; only the small pending queue uses a loop."""
    obs=ledger.obs;farm=obs['farms'][obs['player']];pr=obs['private'];pending=ledger.pending
    positions=np.asarray([farm['farmer']]+farm['hands']);xy=JOB_XY;x,y=xy.T
    raw=np.zeros((1101,128),np.float32);raw[:,:33]=BASE_JOB[:,:33]
    refs=np.full((1101,5),-1,np.int64);rel=np.zeros((1101,8),np.float32)
    r=raw[1:];rf=refs[1:];rr=rel[1:];board=encoded['boards'][0]
    r[:,33:69]=board[:,y,x].T;rf[:,0]=y*10+x;rf[:,1]=JOB_ITEMS
    d=np.abs(xy[:,None]-positions[None]).sum(-1)
    has=JOB_ITEMS>=0;ii=JOB_ITEMS.clip(0);supply=np.zeros(12)
    carry=np.array([[v.get(item,0) for item in ITEMS] for v in pr['inventories']])
    work_distance=d.copy()
    sheds=np.asarray([nearest_shed(pos) for pos in positions])
    via_shed=np.abs(positions-sheds).sum(-1)[None]+np.abs(xy[:,None]-sheds[None]).sum(-1)+1
    needs_pickup=np.isin(JOB_TYPES,(7,8,9))[:,None]&(carry[:,ii].T<=0)
    work_distance=np.where(needs_pickup,via_shed,work_distance)
    nearest=np.argsort(work_distance,axis=1,kind='stable')[:,:min(2,len(positions))]
    rf[:,2:2+nearest.shape[1]]=nearest
    rr[:,:nearest.shape[1]]=np.take_along_axis(work_distance,nearest,axis=1)/18
    for item_i,item in enumerate(ITEMS):supply[item_i]=pr['seeds'].get(item,0) if item in CROPS else pr['shed'].get(item,0)+carry[:,item_i].sum()
    for k in range(nearest.shape[1]):rr[:,2+k]=np.where(has,scaled(carry[nearest[:,k],ii]),0)
    material=np.eye(12)[ii]*has[:,None];reserved=ledger.reserved()
    short=np.maximum(0,1+reserved[ii]-supply[ii])*has
    r[:,81:93]=material;r[:,93:105]=material*short[:,None]
    empty=board[1,y,x]>0;locked=board[0,y,x]>0;animal=board[10:13,y,x].sum(0)>0
    legal=np.zeros(1100,bool)
    legal[JOB_TYPES<7]=empty[JOB_TYPES<7]
    legal[JOB_TYPES==7]=(board[9,y,x]>0)[JOB_TYPES==7]&~animal[JOB_TYPES==7]
    for typ in (8,9):legal[JOB_TYPES==typ]=(board[8,y,x]>0)[JOB_TYPES==typ]&~animal[JOB_TYPES==typ]
    legal[JOB_TYPES==10]=~empty[JOB_TYPES==10]&~animal[JOB_TYPES==10]&~locked[JOB_TYPES==10]
    admit=legal.copy();dep=np.full(1100,-1)
    # Last queued action at a tile defines the next allowed dependent operation.
    last={}
    for pi,p in enumerate(pending):last[(p['x'],p['y'])]=(pi,p)
    for (px,py),(pi,p) in last.items():
        at=(x==px)&(y==py);admit[at]=False;dep[at]=pi
        if p['op']=='DIG':admit[at&(JOB_TYPES<7)]=True
        if p['op']=='BUILD_COOP':admit[at&(JOB_TYPES==7)]=True
        if p['op']=='BUILD_PASTURE':admit[at&np.isin(JOB_TYPES,(8,9))]=True
        for old_i,pv in enumerate(encoded['programs']):
            if encoded['program_valid'][old_i] and pv[JOB_OPS.index(p['op'])] and np.allclose(pv[17:19],[px/9,py/9]):
                rf[at,4]=old_i;break
    for p in pending:admit[JOB_INDEX[job_key(p)]-1]=False;legal[JOB_INDEX[job_key(p)]-1]=False
    if len(pending)>=16:admit[:]=False;legal[:]=False
    admit &= ~locked
    seed_available=np.array([pr['seeds'].get(item,0) for item in ITEMS])
    legal &= np.where(JOB_TYPES<5,seed_available[ii]>0,True)
    legal &= (d==0).any(1)
    legal &= np.where(np.isin(JOB_TYPES,(7,8,9)),((d==0)&(carry[:,ii].T>0)).any(1),True)
    costs=JOB_COSTS;first=JOB_MATURITY
    r[:,69]=0;r[:,70]=scaled(work_distance.min(1)+1);r[:,71]=first/30;r[:,72]=(719-clock(obs))/719
    r[:,73]=scale(farm['money']);r[:,74]=scaled(np.maximum(0,costs-farm['money']));r[:,75]=scaled(short)
    r[:,76]=(16-len(pending))/16;r[:,77]=len(ledger.orders)/10;r[:,79]=scale(ledger.spend);r[:,80]=scale(ledger.income)
    item_context=np.zeros((12,8))
    for i,item in enumerate(ITEMS):item_context[i]=[scale(pr['shed'].get(item,0)),scale(pr['seeds'].get(item,0)),
        scale(reserved[i]),scale(ledger.buys[i]),scale(ledger.sells[i]),scale(obs['market']['inventory'].get(item,0)),
        scale(obs['market']['prices'].get(item,0)),item in obs['market']['prices']]
    r[:,105:113]=item_context[ii]*has[:,None];r[:,113]=scale(farm['money']-ledger.spend+ledger.income)
    seed_jobs=JOB_TYPES<5;r[seed_jobs,108]=scaled(ledger.seed_buys[ii[seed_jobs]])
    r[:,114]=dep>=0;r[:,116]=scaled(costs)
    for typ,item in enumerate(CROPS):
        at=JOB_TYPES==typ;r[at,115]=scale(CROP_RULES[item][3]*obs['market']['prices'].get(item,0))
    for typ,item in zip((7,8,9),ANIMALS):
        at=JOB_TYPES==typ;product=ANIMAL_RULES[item][3]
        r[at,115]=scale({'GOOSE':4,'COW':6,'SHEEP':6}[item]*obs['market']['prices'].get(product,0))
    r[:,117:128]=np.stack([legal,admit,np.ones(1100),np.zeros(1100),np.ones(1100),dep>=0,
                          (dep+1)/16,np.ones(1100),np.ones(1100),np.ones(1100),np.ones(1100)],-1)
    rr[:,4:8]=np.stack([np.full(1100,len(positions)/20),dep>=0,np.ones(1100),np.ones(1100)],-1)
    return dict(raw=raw,refs=refs,relations=rel,legal_now=np.r_[True,legal],admissible=np.r_[True,admit])


def quantity_features(ledger,index,quantities):
    op,item=MARKET[index];q=np.asarray(quantities);pr=ledger.obs['private'];cash=ledger.obs['farms'][ledger.obs['player']]['money']
    out=np.zeros((len(q),12),np.float32);price=ledger.cost(index)
    available=(pr['seeds'] if op=='BUY_SEED' else pr['shed']).get(item,0)
    out[:,0]=scaled(q);out[:,1]=q/max(1,q.max())
    out[:,2 if op!='SELL' else 3]=scaled(q*price)
    out[:,4]=scale(available);out[:,5]=scaled(q/max(1,available))
    out[:,6]=scaled(np.maximum(0,q*price-cash)) if op!='SELL' else 0
    out[:,7]=scaled(np.maximum(0,q-available))
    out[:,8]=scale(max(0,100-sum(pr['shed'].values())));out[:,9]=price>0;out[:,10]=1;out[:,11]=q>0
    return out,q>0


def batch_arrays(rows,device):
    import torch
    out={}
    for key in rows[0]:
        vals=[r[key] for r in rows]
        if key in ('workers','worker_valid'):
            size=max(len(v) for v in vals)
            vals=[np.pad(v,[(0,size-len(v))]+[(0,0)]*(v.ndim-1)) for v in vals]
        out[key]=torch.as_tensor(np.stack(vals),device=device)
    return out
