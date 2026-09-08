"""Horizon Ledger DP v12: an independently implemented observation-only farming agent.

No replay, opponent program, episode seed, trained model, or official simulator
is loaded by this file. Constants below describe the provided game rules.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np

ITEMS = ('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
IX = {x:i for i,x in enumerate(ITEMS)}
CROPS = {'WHEAT':(10,2,4,0,6), 'CARROT':(20,2,3,0,4),
         'TOMATO':(50,8,8,1,4), 'STRAWBERRY':(100,10,10,2,4), 'MELON':(80,10,12,0,6)}
ANIMALS = {'GOOSE':(300,4,1,4,'EGG'), 'COW':(400,8,2,6,'MILK'), 'SHEEP':(500,6,3,6,'WOOL')}
PARAM = {'WHEAT':(25,400,'sqrt',.8,'log',.2), 'CARROT':(35,450,'hinge',1.,'sqrt',.7),
         'TOMATO':(60,200,'hinge',.4,'sqrt',.6), 'STRAWBERRY':(120,100,'sqrt',.7,'linear',1.6),
         'MELON':(250,300,'log',.2,'sq',3.6), 'EGG':(50,332,'hinge',.4,'log',.2),
         'MILK':(160,122,'sqrt',.6,'linear',1.6), 'WOOL':(200,105,'log',.2,'sq',3.2),
         'FERTILIZER':(100,200,'linear',.4,'linear',.4)}
SHOPS = {'BAKERY':('EGG','WHEAT'),'PIZZA_SHOP':('MILK','TOMATO','WHEAT'),
         'BRUNCH_SPOT':('EGG','WHEAT','STRAWBERRY'),'YARN_STORE':('WOOL',),
         'ICE_CREAM_SHOP':('STRAWBERRY','MILK','WHEAT'),'PET_CAFE':('CARROT',),
         'SMOOTHIE_SHOP':('STRAWBERRY','MILK'),'FARMERS_MARKET':('WHEAT','CARROT','TOMATO','STRAWBERRY')}
FIB = [1,1]
for _ in range(20): FIB.append(sum(FIB[-2:]))

# Economic/search settings, not a route or calendar of purchases.
SETTINGS = dict(labor=3., discount=.04, reserve=220., max_hands=15,
                future_shop_weight=1., investment_fraction=.97, max_new=40,
                budget_unit=10, land_value_haircut=.85, admission_capacity=125, terminal_extra=2, portfolio_rounds=3, price_relax=1., labor_visits=1.85, labor_capacity=18., wage_blend=1., liquidity_discount=.2)


def dist(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def shape(f,x,T):
    if f=='linear': return x
    if f=='sqrt': return np.sqrt(x)
    if f=='log': return np.log1p(x)
    if f=='sq': return x*x
    u=x/T
    return u+8*np.maximum(0,u-1)**2

def prices(item, inventory):
    b,T,lo,lt,hi,ht=PARAM[item]
    x=np.asarray(inventory,dtype=float)-10000
    v=np.where(x<0, b+b*lt*shape(lo,np.maximum(0,-x),T)/shape(lo,float(T),T),
                    b-b*ht*shape(hi,np.maximum(0,x),T)/shape(hi,float(T),T))
    return np.maximum(1,np.rint(v))


def crop_schedule(crop, start, last=29, fertilizer=True):
    """Single lifecycle; daily physical outputs, inputs and service operations."""
    seed,first,peak,interval,cap=CROPS[crop]
    out=np.zeros((30,9)); fert=np.zeros(30); work=np.zeros(30)
    if start+first>last: return None
    water=set(); fs=set(); harvest={}
    if not interval:
        age={'WHEAT':4,'CARROT':3,'MELON':10}[crop]
        end=min(last,start+age)
        # Fertilization is optional; its opportunity cost is charged separately.
        if fertilizer and crop in ('WHEAT','CARROT'):
            fs.add(start+2)
        y=1; missed=1
        for d in range(start,end+1):
            a=d-start
            needed=missed>=1 or ((peak+1)//2<=a<=peak)
            if needed:
                water.add(d); missed=0
                if (peak+1)//2<=a<=peak:
                    y=min(cap,y+(2 if any(f<=d<=f+2 for f in fs) else 1))
            else: missed+=1
        harvest[end]=y
    else:
        ts=[start+first+k*interval for k in range(cap) if start+first+k*interval<=last]
        if fertilizer:
            for t in ts:
                if not any(f<=t-1<=f+2 for f in fs): fs.add(t-1)
        end=ts[-1]; held=0; missed=1
        for d in range(start,end+1):
            needed=(missed>=1 or (d+1 in ts and any(f<=d<=f+2 for f in fs))) and d<end
            if needed: water.add(d); missed=0
            else: missed+=1
            if d in ts:
                held+=2 if any(f<=d-1<=f+2 for f in fs) else 1
                if held>=3 or d==end:
                    harvest[d]=held;held=0
    for d,n in harvest.items(): out[min(last,d+1),IX[crop]]+=n;work[d]+=1
    for d in water: work[d]+=1
    for d in fs:
        if d<=end: fert[d]+=1;work[d]+=1
    work[start]+=1
    return dict(out=out,fert=fert,work=work,water=water,fs=fs,harvest=harvest,end=end)


class Policy:
    def __init__(self, settings=None):
        self.s=dict(SETTINGS)
        if settings:self.s.update(settings)
        self.day=-1;self.targets={};self.routes={};self.wanted_hands=0
        self.events=[];self.last_step=-1;self.all_diag=[]

    def demands(self,obs):
        dm=np.zeros((30,9));dm[:,:8]=1
        for sn in obs['town']['unlocked_shops']:
            ps=SHOPS[sn]
            for p in ps:dm[:,IX[p]]+=6*(2 if len(ps)==1 else 1)
        expected=np.zeros(9)
        for ps in SHOPS.values():
            for p in ps:expected[IX[p]]+=6*(2 if len(ps)==1 else 1)/8
        now=obs['day']
        for d in range(now,30):
            future=max(0,min(8,d//3)-len(obs['town']['unlocked_shops']))
            dm[d]+=future*expected*self.s['future_shop_weight']
        return dm

    def animal_stream(self,kind,start,now,existing=None):
        out=np.zeros((30,9));feed=np.zeros(30);work=np.zeros(30)
        cost,first,interval,cap,prod=ANIMALS[kind]
        bank=existing.get('pending_care_bonus',0) if existing else 0
        held=existing.get('yield_units',0) if existing else 0
        for d in range(now,30):
            if d>start:
                out[min(29,d+1),8]+=1;work[d]+=1
            if held:
                out[min(29,d+1),IX[prod]]+=held;held=0;work[d]+=1
            feed[d]=int(d<29 and (existing is not None or d>start))
            work[d]+=feed[d]
            ts=[t for t in range(max(d+2,start+first),30) if (t-start-first)%interval==0]
            care=bool(ts and bank<cap-1)
            # At end of d, scheduled production consumes the old bank before care.
            if d+1>=start+first and (d+1-start-first)%interval==0:
                held=min(cap,1+bank) if feed[d] else 1;bank=0
            if care and feed[d]:bank+=1;work[d]+=1
        if not existing:work[start]+=3
        return out,feed,np.zeros(30),work,(work>0).astype(float)

    def crop_stream(self,kind,start,now,existing=None,fertilizer=False):
        out=np.zeros((30,9));feed=np.zeros(30);fert=np.zeros(30);work=np.zeros(30)
        end=start
        sc=crop_schedule(kind,start,fertilizer=fertilizer)
        if sc is None:return out,feed,fert,work,(work>0).astype(float)
        out+=sc['out'];fert+=sc['fert'];work+=sc['work'];end=sc['end']
        if existing:
            # Correct current held inventory and fertilizer already paid for.
            out[:now]=0;fert[:now]=0;work[:now]=0
            if existing.get('fertilized_until_day',-1)>=now:
                for d in range(now,min(30,existing['fertilized_until_day']+1)):fert[d]=0
        # Same-crop renewal is a forecast, never a binding action chain.
        while end+1+CROPS[kind][1]<=29:
            sc=crop_schedule(kind,end+1,fertilizer=fertilizer)
            out+=sc['out'];fert+=sc['fert'];work+=sc['work'];end=sc['end']
        return out,feed,fert,work,(work>0).astype(float)

    def streams(self,kind,start,now,existing=None,fertilizer=False):
        if kind in ANIMALS:return self.animal_stream(kind,start,now,existing)
        return self.crop_stream(kind,start,now,existing,fertilizer)

    def forecast_base(self,obs):
        d=obs['day'];f=obs['farms'][obs['player']]
        out=np.zeros((30,9));feed=np.zeros(30);fert=np.zeros(30);work=np.zeros(30);visits=np.zeros(30)
        for row in f['tiles']:
            for t in row:
                if not isinstance(t,dict):continue
                if 'animal' in t:
                    a,b,c,w,vis=self.animal_stream(t['animal'],t['placed_day'],d,t)
                elif t.get('kind')=='PLANT':
                    a,b,c,w,vis=self.crop_stream(t['crop'],t['planted_day'],d,t,self.fert_mode.get(t['crop'],False))
                else:continue
                out+=a;feed+=b;fert+=c;work+=w;visits+=vis
        for k,n in obs['private']['shed'].items():
            if k in IX:out[d,IX[k]]+=n
        for inv in obs['private']['inventories']:
            for k,n in inv.items():
                if k in IX:out[d,IX[k]]+=n
        out[:d]=0;feed[:d]=0;fert[:d]=0;work[:d]=0;visits[:d]=0
        return out,feed,fert,work,visits

    def economic_value(self,obs,streams,dm):
        out,feed,fert,work,visits=streams
        d=obs['day'];q=out.copy();q[:,0]-=feed;q[:,8]-=fert
        initial=np.array([obs['market']['inventory'][x] for x in ITEMS],float)
        delta=q[d:]
        before=initial+np.cumsum(delta-dm[d:],axis=0)-delta
        discounted=np.exp(-self.current_discount*np.arange(30-d))
        # Fibonacci hiring costs make workload coupling nonseparable.
        need=np.maximum(0,(work[d:]+self.s['labor_visits']*visits[d:])/self.s['labor_capacity']-1)
        need=np.minimum(19,need)
        lo=np.floor(need).astype(int);frac=need-lo
        cumulative=np.array([0]+[sum(FIB[:n]) for n in range(1,21)],float)
        wage=cumulative[lo]+frac*np.array(FIB)[lo]
        daily=-(self.s['wage_blend']*wage+(1-self.s['wage_blend'])*work[d:]*self.s['labor'])
        for j,k in enumerate(ITEMS):
            mids=before[:,j,None]+delta[:,j,None]*np.array([.125,.375,.625,.875])
            daily+=delta[:,j]*prices(k,mids).mean(axis=1)
        return float(daily@discounted)

    def choose_portfolio(self,obs):
        d=obs['day'];f=obs['farms'][obs['player']];money=f['money'];p=obs['private']
        occupied=sum(isinstance(t,dict) and (t.get('kind')=='PLANT' or 'animal' in t) for row in f['tiles'] for t in row)
        liquid=sum(n*obs['market']['prices'].get(k,0) for k,n in p['shed'].items() if k not in ANIMALS)
        # Cash paid before expansion is more valuable than distant liquidation.
        # This state-derived shadow price vanishes as cash/land cease to bind.
        growth_room=max(0,1-occupied/100)
        cash_pressure=max(0,1-(money+liquid)/10000)
        self.current_discount=self.s['discount']+self.s['liquidity_discount']*growth_room*cash_pressure
        self.fert_mode={k:False for k in CROPS}
        # Fertilizer has a positive input opportunity cost, not production revenue.
        fp=obs['market']['prices']['FERTILIZER']
        for k in ('WHEAT','CARROT','TOMATO','STRAWBERRY'):
            extra={'WHEAT':2,'CARROT':1,'TOMATO':2,'STRAWBERRY':2}[k]
            self.fert_mode[k]=obs['market']['prices'][k]*extra>fp+8
        dm=self.demands(obs);base=self.forecast_base(obs);baseval=self.economic_value(obs,base,dm)
        self.base=base;self.dm=dm
        vacant=[]
        for y,row in enumerate(f['tiles']):
            for x,t in enumerate(row):
                if t is None or (isinstance(t,dict) and t.get('kind') in ('WEED','COOP','PASTURE') and 'animal' not in t):vacant.append((x,y))
        self.vacant=vacant
        n_anim=sum('animal' in t for row in f['tiles'] for t in row if isinstance(t,dict))
        reserve=max(self.s['reserve'],n_anim*obs['market']['prices']['WHEAT']*1.4+sum(FIB[:min(10,self.s['max_hands'])]))
        # Shed assets can finance commitments, but no unopened shop/hidden money.
        liquid=sum(n*obs['market']['prices'].get(k,0) for k,n in p['shed'].items() if k not in ANIMALS)
        available=max(0,money+liquid-reserve)
        if d>=28: available=min(available,300)
        budget=min(14000,int(available*self.s['investment_fraction']))
        slots=min(75,100-sum(1 for row in f['tiles'] for t in row if isinstance(t,dict) and (t.get('kind')=='PLANT' or 'animal' in t)),self.s['max_new'])
        if budget<10 or slots<=0:return {}
        B=budget//self.s['budget_unit'];T=slots
        # Candidate DP with iterative cross-industry resource repricing.
        # Each solve is exact on its discretized conditional value tables;
        # selecting among jointly repriced candidates is approximate globally.
        groups=[]
        for k in list(ANIMALS)+list(CROPS):
            cost=ANIMALS[k][0] if k in ANIMALS else CROPS[k][0]
            first=ANIMALS[k][1] if k in ANIMALS else CROPS[k][1]
            if d+first>29:continue
            c=math.ceil(cost/self.s['budget_unit'])
            m=min(T,B//c,24 if k in ANIMALS else T)
            if not m:continue
            stream=self.streams(k,d,d,fertilizer=self.fert_mode.get(k,False))
            renewal=0
            if k in CROPS:
                duration={'WHEAT':5,'CARROT':4,'TOMATO':12,'STRAWBERRY':17,'MELON':11}[k]
                cycles=max(0,(29-d-first)//duration)
                renewal=cost*sum(math.exp(-self.current_discount*duration*i) for i in range(1,cycles+1))
            groups.append((k,c,m,cost+renewal,stream))
        unlocked=len(f['unlocked_quadrants']);landprices=[1000,2000,4000]
        def joint(chosen,extra):
            ss=tuple(z.copy() for z in base);capital=0
            for k,c,m,cost,st in groups:
                n=chosen.get(k,0);capital+=n*cost
                for j in range(5):ss[j][:]+=n*st[j]
            lc=sum(landprices[unlocked-1:unlocked-1+extra])
            val=self.economic_value(obs,ss,dm)-baseval-capital-lc
            if extra:val*=self.s['land_value_haircut']
            return val,ss
        previous={};aggregate=base;best=(0.,{},0)
        for iteration in range(self.s['portfolio_rounds']):
            dp=np.full((T+1,B+1),-1e30);dp[0,:]=0;records=[]
            for k,c,m,cost,stream in groups:
                scale=self.s['price_relax'] if iteration else 0.
                others=tuple(base[j]+scale*(aggregate[j]-base[j]-previous.get(k,0)*stream[j]) for j in range(5))
                base_cond=self.economic_value(obs,others,dm)
                vals=[0.]
                for n in range(1,m+1):
                    ss=tuple(others[j]+n*stream[j] for j in range(5))
                    vals.append(self.economic_value(obs,ss,dm)-base_cond-n*cost)
                nxt=dp.copy();choice=np.zeros_like(dp,dtype=np.int8)
                for n in range(1,m+1):
                    if vals[n]<=0:continue
                    z=dp[:-n,:-n*c]+vals[n]
                    region=nxt[n:,n*c:];better=z>region
                    region[better]=z[better];choice[n:,n*c:][better]=n
                dp=nxt;records.append(choice)
            candidates=[]
            for extra in range(4-unlocked+1):
                lc=sum(landprices[unlocked-1:unlocked-1+extra])
                b=(budget-lc)//self.s['budget_unit']
                if b<0:continue
                cap=min(T,len(vacant)+25*extra)
                t=int(np.argmax(dp[:cap+1,b]));chosen={}
                for rec,(k,c,m,cost,st) in zip(reversed(records),reversed(groups)):
                    n=int(rec[t,b]);chosen[k]=n;t-=n;b-=n*c
                val,ss=joint(chosen,extra)
                candidates.append((val,chosen,extra,ss))
                if val>best[0]:best=(val,chosen,extra)
            if not candidates:break
            val,chosen,extra,ss=max(candidates,key=lambda z:z[0])
            if chosen==previous:break
            previous=chosen;aggregate=ss
        self.buy_land=best[2]
        return {k:n for k,n in best[1].items() if n}

    def distance_shed(self,pos):return min(dist(pos,s) for s in self.shedtiles)

    def new_day(self,obs):
        self.day=obs['day'];self.routes={};self.refill={};self.targets={};self.buy_land=0;self.deferred_fert=set()
        n=len(obs['farms'][obs['player']]['tiles']);h=n//2
        self.shedtiles=((h-1,h-1),(h,h-1),(h-1,h),(h,h))
        chosen=self.choose_portfolio(obs)
        for k in ANIMALS:
            held=obs['private']['shed'].get(k,0)+sum(v.get(k,0) for v in obs['private']['inventories'])
            chosen[k]=max(chosen.get(k,0),held)
        chosen={k:n for k,n in chosen.items() if n}
        self.events.append(dict(day=self.day,chosen=chosen,land=self.buy_land,fert_mode=self.fert_mode.copy(),capital_discount=self.current_discount))
        # New sites are selected by service frequency, not a pre-recorded map.
        sites=list(self.vacant)
        f=obs['farms'][obs['player']]
        for extra in range(self.buy_land):
            q=('NE','SW','SE')[len(f['unlocked_quadrants'])-1+extra]
            for y in range(n):
                for x in range(n):
                    if (('N' if y<h else 'S')+('W' if x<h else 'E'))==q:sites.append((x,y))
        sites.sort(key=lambda p:(self.distance_shed(p),p[1],p[0]))
        # Frequent animal visits stay closest to the shed. Crop placement is
        # merely a spatial cost decision, independent of worker identity.
        for k in sorted(chosen,key=lambda k:0 if k in ANIMALS else 1):
            for _ in range(chosen[k]):
                if not sites:break
                self.targets[sites.pop(0)]=k
        self.wanted_hands=0;self.compiled=False

    def tile_tasks(self,obs):
        """Return independent task flags. Missing input removes only that action."""
        d=obs['day'];f=obs['farms'][obs['player']];p=obs['private'];tasks={}
        for y,row in enumerate(f['tiles']):
            for x,t in enumerate(row):
                pos=(x,y);ops=[];value=0.;need={}
                if isinstance(t,dict) and 'animal' in t:
                    a=t['animal'];cost,first,interval,cap,prod=ANIMALS[a]
                    if t.get('yield_units',0):ops.append(['HARVEST']);value+=t['yield_units']*obs['market']['prices'][prod]
                    if t.get('fertilizer_available'):ops.append(['COLLECT_FERTILIZER']);value+=obs['market']['prices']['FERTILIZER']
                    ts=[z for z in range(max(d+2,t['placed_day']+first),30) if (z-t['placed_day']-first)%interval==0]
                    care=bool(d>t['placed_day'] and ts and t.get('pending_care_bonus',0)<cap-1)
                    nextprod=d+1>=t['placed_day']+first and (d+1-t['placed_day']-first)%interval==0
                    feed=t['placed_day']<d<29 and (t['consecutive_unfed']>=1 or care or nextprod)
                    if feed and not t['fed_today']:
                        ops.append(['FEED']);need['WHEAT']=1;value+=1500 if t['consecutive_unfed'] else 120
                    if care and not t['cared_today']:
                        ops.append(['CARE']);value+=obs['market']['prices'][prod]
                elif isinstance(t,dict) and t.get('kind')=='PLANT':
                    k=t['crop'];seed,first,peak,interval,cap=CROPS[k];age=d-t['planted_day']
                    if not interval:
                        ideal={'WHEAT':4,'CARROT':3,'MELON':10}[k]
                        harvest=age>=ideal or d==29
                        dofert=self.fert_mode.get(k,False) and 2<=age<=3 and t['fertilized_until_day']<d and k in ('WHEAT','CARROT') and not t['watered_today']
                        water=not t['watered_today'] and d<29 and (t['consecutive_unwatered']>=1 or (peak+1)//2<=age<=peak)
                        if d==29 and age>=first and (peak+1)//2<=age<=peak and not t['watered_today']:water=True
                        if dofert and pos not in self.deferred_fert:ops.append(['FERTILIZE']);need['FERTILIZER']=1;value+=40
                        if water:ops.append(['WATER']);value+=800 if t['consecutive_unwatered'] else 80
                        if harvest and age>=first and t['yield_units']>0:
                            ops.append(['HARVEST']);value+=t['yield_units']*obs['market']['prices'][k]
                    else:
                        prod=[t['planted_day']+first+i*interval for i in range(cap)]
                        last=min(29,prod[-1]);lastprod=max((z for z in prod if z<=29),default=30)
                        yf=t.get('yield_units',0)
                        harvest=yf>=3 or (d>=lastprod and yf>0) or (d==29 and yf>0) or (t.get('max_lifespan_step',-1)>=0 and t['max_lifespan_step']<=obs['step']+24 and yf>0)
                        if harvest:ops.append(['HARVEST']);value+=yf*obs['market']['prices'][k]+(1000 if d>=lastprod else 0)
                        nextp=d+1 in prod and d+1<=29
                        dofert=nextp and self.fert_mode.get(k,False) and t['fertilized_until_day']<d
                        if dofert and pos not in self.deferred_fert:ops.append(['FERTILIZE']);need['FERTILIZER']=1;value+=100
                        water=not t['watered_today'] and d<lastprod and (t['consecutive_unwatered']>=1 or (nextp and (t['fertilized_until_day']>=d or dofert)))
                        if water:ops.append(['WATER']);value+=1000 if t['consecutive_unwatered'] else 80
                        # Freed mature perennial slots are replanned next dawn.
                        # Do not append an unbudgeted DIG after the final HARVEST.
                        # Natural clearing / next-day investment decides reuse.
                if pos in self.targets:
                    k=self.targets[pos]
                    if t=='LOCKED':continue
                    clear = isinstance(t,dict) and (t.get('kind')=='WEED' or ('animal' not in t and t.get('kind') in ('COOP','PASTURE') and (k in CROPS or t['kind']!=('COOP' if k=='GOOSE' else 'PASTURE'))))
                    if clear:ops.append(['DIG']);value+=20
                    if k in CROPS and (t is None or clear):
                        ops.extend([['PLANT',k],['WATER']]);value+=500
                    elif k in ANIMALS and not (isinstance(t,dict) and 'animal' in t):
                        if t is None or clear:
                            ops.append(['BUILD_COOP' if k=='GOOSE' else 'BUILD_PASTURE'])
                        ops.append(['PLACE',k]);need[k]=1;value+=1500
                if ops:tasks[pos]=dict(ops=ops,value=value,need=need)
        return tasks

    def admit_outputs(self,obs,tasks):
        if obs['day']==29:return tasks
        if not hasattr(self,'admitted') or self.admitted_day!=obs['day']:
            f=obs['farms'][obs['player']];d=obs['day']
            animals=sum('animal' in t for row in f['tiles'] for t in row if isinstance(t,dict))
            capacity=max(30,self.s['admission_capacity']-animals)
            options=[]
            for pos,job in tasks.items():
                tile=f['tiles'][pos[1]][pos[0]]
                for op in job['ops']:
                    if op[0]=='HARVEST':
                        item=tile.get('crop') or ANIMALS[tile['animal']][4]
                        q=tile.get('yield_units',0)
                        if item in ('WHEAT','CARROT','MELON') and not tile.get('watered_today'):
                            q=min(CROPS[item][4],q+1)
                        price=obs['market']['prices'][item]
                        # Cost of withholding an expiring or blocked production.
                        urgent=0
                        if 'animal' in tile:
                            _,first,interval,cap,prod=ANIMALS[tile['animal']]
                            nextprod=(d+1-tile['placed_day']>=first and (d+1-tile['placed_day']-first)%interval==0)
                            if nextprod:urgent=max(0,q+1+tile.get('pending_care_bonus',0)-cap)*price
                        else:
                            expiry=tile.get('max_lifespan_step',-1)
                            if 0<=expiry<=(d+1)*24:urgent=q*price
                            if item in ('TOMATO','STRAWBERRY'):
                                first=CROPS[item][1];interval=CROPS[item][3]
                                if d+1-tile['planted_day']>=first and (d+1-tile['planted_day']-first)%interval==0:urgent+=max(0,q+2-4)*price
                        options.append((pos,op[0],q,q*price+2*urgent))
                    elif op[0]=='COLLECT_FERTILIZER':
                        options.append((pos,op[0],1,obs['market']['prices']['FERTILIZER']*1.25))
            dp=[0.]*(capacity+1);sets=[frozenset() for _ in dp]
            for pos,op,q,v in options:
                for c in range(capacity,q-1,-1):
                    vv=dp[c-q]+v
                    if vv>dp[c]:dp[c]=vv;sets[c]=sets[c-q]|{(pos,op)}
            self.admitted=set(sets[capacity]);self.admitted_day=d
        filtered={}
        for pos,job in tasks.items():
            ops=[a for a in job['ops'] if a[0] not in ('HARVEST','COLLECT_FERTILIZER') or (pos,a[0]) in self.admitted]
            if ops:
                filtered[pos]=dict(job,ops=ops)
        return filtered

    def plan_routes(self,obs,tasks):
        f=obs['farms'][obs['player']];units=[tuple(f['farmer'])]+[tuple(q) for q in f['hands']]
        remaining=23-obs['hour'] if self.day<29 else 23-obs['hour']
        # Greedy regret insertion assigns unbound tile services to daily routes.
        # A service is re-read from the actual board before every action.
        routes=[[] for _ in units];load=[0 for _ in units]
        if self.compiled and obs['hour']>=2:
            for u,old in self.routes.items():
                if u>=len(units):continue
                routes[u]=[q for q in old if q in tasks]
                prev=units[u];req=Counter()
                for q in routes[u]:
                    load[u]+=dist(prev,q)+len(tasks[q]['ops']);req.update(tasks[q]['need']);prev=q
                inv=obs['private']['inventories'][u]
                load[u]+=sum(inv.get(k,0)<n for k,n in req.items())
                if any(inv.get(k,0)<n for k,n in req.items()) and self.distance_shed(units[u])>0 and routes[u]:
                    load[u]+=min(dist(units[u],z)+dist(z,routes[u][0]) for z in self.shedtiles)-dist(units[u],routes[u][0])
        existing={q for r in routes for q in r}
        order=sorted(tasks,key=lambda p:(-int(tasks[p]['value']>=800),-self.distance_shed(p),-tasks[p]['value']))
        dropped=[]
        for pos in order:
            if pos in existing:continue
            job=tasks[pos];best=None
            for u,start in enumerate(units):
                route=routes[u]
                for j in range(len(route)+1):
                    prev=start if j==0 else route[j-1]
                    nxt=route[j] if j<len(route) else None
                    extra=dist(prev,pos)+(dist(pos,nxt)-dist(prev,nxt) if nxt else 0)+len(job['ops'])
                    # Account for wheat/fertilizer/animal pickup at the shed.
                    needs={k for q in route for k in tasks[q]['need']}
                    pickup=len(set(job['need'])-needs)
                    cost=extra+pickup
                    if not route and pickup and self.distance_shed(start)>0:
                        inv=obs['private']['inventories'][u]
                        if any(inv.get(k,0)<n for k,n in job['need'].items()):
                            cost+=min(dist(start,q)+dist(q,pos) for q in self.shedtiles)-dist(start,pos)
                    final=pos if j==len(route) else route[-1]
                    ret=self.distance_shed(final)+1 if self.day==29 else 0
                    total=load[u]+cost
                    if total+ret>remaining:continue
                    metric=cost+.025*total
                    if best is None or metric<best[0]:best=(metric,u,j,cost)
            if best:
                _,u,j,cost=best;routes[u].insert(j,pos);load[u]+=cost
            else:dropped.append(pos)
        self.routes={u:r for u,r in enumerate(routes)}
        self.compiled=True;self.dropped=dropped
        self.all_diag.append(dict(day=self.day,hour=obs['hour'],tasks=len(tasks),workers=len(units),dropped=len(dropped),loads=load))

    def move(self,pos,target):
        if pos[0]<target[0]:return ['EAST']
        if pos[0]>target[0]:return ['WEST']
        if pos[1]<target[1]:return ['SOUTH']
        if pos[1]>target[1]:return ['NORTH']
        return ['PASS']

    def call(self,obs,configuration=None):
        if obs['day']!=self.day:self.new_day(obs)
        f=obs['farms'][obs['player']];p=obs['private'];d=obs['day'];h=obs['hour']
        tasks=self.admit_outputs(obs,self.tile_tasks(obs))
        # Workload with spatial lower bound; hire to reduce missed profitable work.
        if h<=3:
            work=sum(len(v['ops']) for v in tasks.values())+len(tasks)*1.85
            work+=sum((3 if k in ANIMALS else 2)+1.85 for pos,k in self.targets.items() if f['tiles'][pos[1]][pos[0]]=='LOCKED')
            if d==29:work+=len(tasks)*1.3
            required=math.ceil(work/18.)-1
            cap=self.s['max_hands']+(self.s['terminal_extra'] if d==29 else 0)
            self.wanted_hands=min(cap,max(5,required)) if tasks else 0
        market=[];cash=f['money'];shed=dict(p['shed']);seeds=dict(p['seeds'])
        animal_need=sum(1 for row in f['tiles'] for t in row if isinstance(t,dict) and 'animal' in t)
        feed_keep=animal_need if d<29 else 0
        fert_keep=sum(v['need'].get('FERTILIZER',0) for v in tasks.values()) if d<29 else 0
        # Sell available surpluses. Resource protection is independent of hiring
        # and production, so ordinary sale cannot consume today's committed feed.
        for k in ITEMS:
            keep=feed_keep if k=='WHEAT' else fert_keep if k=='FERTILIZER' else 0
            qty=max(0,shed.get(k,0)-keep)
            if qty:
                market.append(['SELL',k,qty]);cash+=qty*obs['market']['prices'][k];shed[k]-=qty
        mandatory=sum(v['need'].get('WHEAT',0) for v in tasks.values())
        have=p['shed'].get('WHEAT',0)+sum(v.get('WHEAT',0) for v in p['inventories'])
        topup=min(max(0,mandatory-have),max(0,int((cash-10)//(obs['market']['prices']['WHEAT']+2))))
        if topup and len(market)<10:
            market.append(['BUY_PRODUCT','WHEAT',topup]);cash-=topup*(obs['market']['prices']['WHEAT']+2)
        while len(f['hands'])+sum(x[0]=='HIRE' for x in market)<self.wanted_hands and len(market)<10:
            n=len(f['hands'])+sum(x[0]=='HIRE' for x in market);cost=FIB[n]
            if cash<cost+20:break
            market.append(['HIRE']);cash-=cost
        if h<6:
            while self.buy_land and len(market)<10:
                cost=[1000,2000,4000][len(f['unlocked_quadrants'])-1+sum(x[0]=='BUY_LAND' for x in market)]
                if cash<cost+50:break
                market.append(['BUY_LAND']);cash-=cost;self.buy_land-=1
            # Only resources needed by real, still-unfinished tasks are purchased.
            need=Counter()
            for pos,k in self.targets.items():
                t=f['tiles'][pos[1]][pos[0]]
                if k in CROPS:
                    if not (isinstance(t,dict) and t.get('crop')==k):need[k]+=1
                elif not (isinstance(t,dict) and t.get('animal')==k):need[k]+=1
            for k,n in need.items():
                stock=seeds.get(k,0) if k in CROPS else shed.get(k,0)+sum(v.get(k,0) for v in p['inventories'])
                q=max(0,n-stock);cost=CROPS[k][0] if k in CROPS else ANIMALS[k][0]
                q=min(q,max(0,int((cash-40)//cost)))
                if q and len(market)<10:
                    market.append(['BUY_SEED' if k in CROPS else 'BUY_ANIMAL',k,q]);cash-=cost*q
                    if k in CROPS:seeds[k]=stock+q
                    else:shed[k]=shed.get(k,0)+q
        wheat_needed=sum(v['need'].get('WHEAT',0) for v in tasks.values())
        wheat_have=p['shed'].get('WHEAT',0)+sum(v.get('WHEAT',0) for v in p['inventories'])
        q=max(0,wheat_needed-wheat_have-sum(z[2] for z in market if z[0]=='BUY_PRODUCT' and z[1]=='WHEAT'))
        q=min(q,max(0,int((cash-10)//(obs['market']['prices']['WHEAT']+2))))
        if q and len(market)<10:market.append(['BUY_PRODUCT','WHEAT',q]);cash-=q*(obs['market']['prices']['WHEAT']+2)
        # Fertilizer purchases are deliberate, not an accidental sign inversion.
        fert_have=p['shed'].get('FERTILIZER',0)+sum(v.get('FERTILIZER',0) for v in p['inventories'])
        q=max(0,fert_keep-fert_have)
        q=min(q,max(0,int((cash-80)//(obs['market']['prices']['FERTILIZER']+2))))
        if q and len(market)<10:market.append(['BUY_PRODUCT','FERTILIZER',q])
        units=[tuple(f['farmer'])]+[tuple(z) for z in f['hands']]
        if not self.compiled or len(self.routes)!=len(units):self.plan_routes(obs,tasks)
        actions=[];reserved_shed=Counter(p['shed']);reserved_seeds=Counter(p['seeds']);claimed=set()
        virtual_shed=Counter(p['shed'])
        for u,pos in enumerate(units):
            inv=p['inventories'][u];route=self.routes.get(u,[])
            # Remove satisfied goals; no worker owns a crop or animal.
            route[:]=[q for q in route if q in tasks]
            a=['PASS'] if h==0 else None
            if u in self.refill and a is None:
                if self.distance_shed(pos)>0:a=self.move(pos,self.refill[u])
                else:self.refill.pop(u,None)
            if self.distance_shed(pos)==0 and a is None:
                # Deposit fertilizer/output even during normal days. KEEP wheat
                # and animals unless clearing the final day.
                deposit=next((k for k,n in inv.items() if n and k in IX and k not in ('WHEAT','FERTILIZER')),None)
                if d==29 and any(inv.values()):a=['DROP']
                elif deposit and h>8:a=['PLACE',deposit,inv[deposit]]
                if a is None and route:
                    req=Counter()
                    for qpos in route:
                        req.update(tasks[qpos]['need'])
                    for k,n in req.items():
                        missing=max(0,n-inv.get(k,0));take=min(missing,reserved_shed[k])
                        if take:
                            a=['PICKUP',k,take];reserved_shed[k]-=take;break
                        if missing and (any(z[0] in ('BUY_PRODUCT','BUY_ANIMAL') and z[1]==k for z in market) or (k in ANIMALS and h<6)):
                            a=['PASS'];break
            # Final-day routes need time to deposit before the last action.
            if a is None and d==29 and inv and h>=21-self.distance_shed(pos):
                target=min(self.shedtiles,key=lambda q:dist(pos,q));a=self.move(pos,target)
            if a is None and route:
                target=route[0]
                if pos!=target:a=self.move(pos,target)
                else:
                    for op in tasks[target]['ops']:
                        key=(target,op[0])
                        if key in claimed or any(z[0]==target for z in claimed):continue
                        name=op[0]
                        # Evaluate real action preconditions, not the stale route.
                        t=f['tiles'][pos[1]][pos[0]]
                        if name=='WATER' and not (isinstance(t,dict) and t.get('kind')=='PLANT' and not t['watered_today']):continue
                        if name=='FERTILIZE' and inv.get('FERTILIZER',0)<1:
                            # Optional yield enhancement must not turn the route
                            # into an unbudgeted depot round trip and kill crops.
                            self.deferred_fert.add(target)
                            tasks[target]['need'].pop('FERTILIZER',None)
                            continue
                        if name=='FEED' and inv.get('WHEAT',0)<1:continue
                        if name=='CARE' and isinstance(t,dict) and not t.get('fed_today') and not inv.get('WHEAT',0):continue
                        if name=='PLANT':
                            if t is not None or not reserved_seeds[op[1]]:continue
                            reserved_seeds[op[1]]-=1
                        if name.startswith('BUILD') and t is not None:continue
                        if name=='PLACE' and (not inv.get(op[1],0) or not isinstance(t,dict) or t.get('animal') or t.get('kind')!=('COOP' if op[1]=='GOOSE' else 'PASTURE')):continue
                        a=op;claimed.add(key);break
                    if a is None:
                        # Independent harvest/collect above still execute before
                        # a resource detour for a blocked feed or fertilize.
                        missing=any(inv.get(k,0)<n for k,n in tasks[target]['need'].items())
                        if missing and self.distance_shed(pos)>0:
                            target=min(self.shedtiles,key=lambda q:dist(pos,q));self.refill[u]=target;a=self.move(pos,target)
                        else:
                            route.pop(0)
            if a is None and not route:
                # Reassign unfinished/dropped service to idle workers.
                goals=[q for q in tasks if not any(z[0]==q for z in claimed) and q not in {v[0] for j,v in self.routes.items() if j!=u and v}]
                goals=[q for q in goals if dist(pos,q)+len(tasks[q]['ops'])+(self.distance_shed(q)+1 if d==29 else 0)<=22-h]
                if goals:
                    q=max(goals,key=lambda q:tasks[q]['value']/(dist(pos,q)+len(tasks[q]['ops'])+1))
                    route.append(q);a=self.move(pos,q) if pos!=q else ['PASS']
                elif inv and self.distance_shed(pos)>0:
                    a=self.move(pos,min(self.shedtiles,key=lambda q:dist(pos,q)))
                elif inv and self.distance_shed(pos)==0:a=['DROP']
            a=a or ['PASS']
            if a[0]=='DROP':
                room=max(0,100-sum(virtual_shed.values()))
                if sum(inv.values())>room:
                    available=[k for k,n in inv.items() if n>0 and k in IX]
                    k=max(available,key=lambda k:obs['market']['prices'][k]) if available else None
                    a=['PLACE',k,min(inv[k],room)] if k and room>0 else ['PASS']
                else:
                    virtual_shed.update(inv)
            if a[0]=='PLACE' and a[1] in IX and self.distance_shed(pos)==0:
                take=min(a[2] if len(a)>2 else 1,inv.get(a[1],0),max(0,100-sum(virtual_shed.values())))
                virtual_shed[a[1]]+=take
            if a[0]=='PICKUP':virtual_shed[a[1]]-=min(a[2],virtual_shed[a[1]])
            actions.append(a)
        if d<29 and len(market)<10:
            deficits=0
            for u,route in self.routes.items():
                req=sum(tasks[q]['need'].get('WHEAT',0) for q in route if q in tasks)
                deficits+=max(0,req-p['inventories'][u].get('WHEAT',0))
            incoming=sum(z[2] for z in market if z[0]=='BUY_PRODUCT' and z[1]=='WHEAT')
            shortage=max(0,deficits-p['shed'].get('WHEAT',0)-incoming)
            q=min(shortage,max(0,int((cash-20)//(obs['market']['prices']['WHEAT']+2))))
            if q:market.append(['BUY_PRODUCT','WHEAT',q])
        # Unit deposits happen before the market, so all actual same-turn
        # incoming goods are eligible for sale, including the last legal hour.
        protected={'WHEAT':feed_keep,'FERTILIZER':fert_keep} if d<29 else {}
        sales=[['SELL',k,int(max(0,virtual_shed.get(k,0)-protected.get(k,0)))] for k in ITEMS]
        sales=[x for x in sales if x[2]>0]
        market=sales+[x for x in market if x[0]!='SELL']
        self.last_step=obs['step']
        return {'farmer':actions[0], 'hands':actions[1:], 'market':market[:10]}

_POLICY=None

def agent(observation,configuration=None):
    global _POLICY
    if _POLICY is None or observation.get('step',0)==0 or observation.get('step',0)<_POLICY.last_step:
        _POLICY=Policy()
    return _POLICY.call(observation,configuration)
