"""Independent daily resource-flow DP agent. No replay or external policy imports.
The constants below are the game's rules, not a learned route.
"""
from __future__ import annotations
import math
from functools import lru_cache

PRODUCTS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
IDX={k:i for i,k in enumerate(PRODUCTS)}
COST={'WHEAT':10,'CARROT':20,'TOMATO':50,'STRAWBERRY':100,'MELON':80,'GOOSE':300,'COW':400,'SHEEP':500}
ANIMALS={'GOOSE':('EGG',4,1,4,'COOP'),'COW':('MILK',8,2,6,'PASTURE'),'SHEEP':('WOOL',6,3,6,'PASTURE')}
CROPS=tuple(COST)[:5]
BASE=(25,35,60,120,250,50,160,200,100)
THR=(400,450,200,100,300,332,122,105,200)
LO=('sqrt','hinge','hinge','sqrt','log','hinge','sqrt','log','linear')
HI=('log','sqrt','sqrt','linear','sq','log','linear','sq','linear')
LT=(.8,1,.4,.7,.2,.4,.6,.2,.4)
HT=(.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4)
SHOPS={'BAKERY':('EGG','WHEAT'),'PIZZA_SHOP':('MILK','TOMATO','WHEAT'),'BRUNCH_SPOT':('EGG','WHEAT','STRAWBERRY'),'YARN_STORE':('WOOL','WOOL'),'ICE_CREAM_SHOP':('STRAWBERRY','MILK','WHEAT'),'PET_CAFE':('CARROT','CARROT'),'SMOOTHIE_SHOP':('STRAWBERRY','MILK'),'FARMERS_MARKET':('WHEAT','CARROT','TOMATO','STRAWBERRY')}
FIB=(1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987)
SETTINGS={'capital_power':.6,'labor_price':3.0,'land_rent':5.0,'future_shop_weight':1.0,'max_hands':15,'crop_repeat':True,'fertilize':True,'animal_care':True,'opening_reserve':80.0,'labor_capacity':True,'advance_revenue':.7,'crop_work_mult':1.0,'animal_work_mult':1.0,'labor_hours':17.0,'delivery_planning':True}
MEM={}

def shape(fn,x,t):
    if fn=='sqrt':return math.sqrt(x)
    if fn=='sq':return x*x
    if fn=='log':return math.log1p(x)
    if fn=='log10':return math.log10(1+x)
    if fn=='hinge':
        u=x/t;return u+8*max(0,u-1)**2
    return x

def price(i,inv,params=None):
    if params and PRODUCTS[i] in params:
        p=params[PRODUCTS[i]]; b=p.get('base',BASE[i]);t=p.get('T',THR[i]);z=p.get('I0',10000)
        below=inv<z;fn=p.get('below_func' if below else 'above_func',LO[i] if below else HI[i]);fac=p.get('below_target' if below else 'above_target',LT[i] if below else HT[i])
    else:
        b=BASE[i];t=THR[i];z=10000;below=inv<z;fn=LO[i] if below else HI[i];fac=LT[i] if below else HT[i]
    return max(1,round(b+(1 if below else -1)*b*fac*shape(fn,abs(inv-z),t)/shape(fn,t,t)))

def dist(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
def access(p,n=10):
    h=n//2;return (min(max(p[0],h-1),h),min(max(p[1],h-1),h))
def move(p,q):
    if p[0]<q[0]:return ['EAST']
    if p[0]>q[0]:return ['WEST']
    if p[1]<q[1]:return ['SOUTH']
    if p[1]>q[1]:return ['NORTH']
    return ['PASS']

def flows0():return [[0.0]*9 for _ in range(30)]
def addflow(a,b,scale=1):
    for d in range(30):
        for i in range(9):a[d][i]+=scale*b[d][i]

def demand_path(obs,cfg):
    day=obs['day'];shops=obs['town']['unlocked_shops'];known=[0.0]*9;avg=[0.0]*9
    daily=cfg.get('turnsPerDay',24)/cfg.get('townShopSellInterval',4)
    for s in shops:
        for p in SHOPS.get(s,()):known[IDX[p]]+=daily
    for names in SHOPS.values():
        for p in names:avg[IDX[p]]+=daily/len(SHOPS)
    out=[]
    for d in range(30):
        new=max(0,min(8,d//cfg.get('townShopUnlockInterval',3))-len(shops))
        out.append([(1 if i<8 else 0)+known[i]+new*avg[i]*SETTINGS['future_shop_weight'] for i in range(9)])
    return out

# A cycle is (duration, [(day offset, resource, quantity), ...], action demand).
@lru_cache(None)
def cycles(crop,fertilize=True):
    variants=[]
    if crop in ('WHEAT','CARROT'):
        for length in ((2,3,4) if crop=='WHEAT' else (2,3)):
            cap=6 if crop=='WHEAT' else 4
            for fert in ((False,True) if fertilize else (False,)):
                q=min(cap,1+(length-1)*(2 if fert else 1))
                events=[(length,IDX[crop],q)]
                if fert:events.append((2,8,-1))
                variants.append((length,tuple(events),2+length+int(fert),int(fert)))
    elif crop=='MELON':
        variants=[(10,((10,4,6),),10,0)]
    else:
        ages=(8,9,10,11) if crop=='TOMATO' else (10,12,14,16)
        schedules=[()] if not fertilize else [(),(ages[0]-1,),((ages[0]-1),(ages[-1]-1)) if crop=='TOMATO' else (9,13)]
        for fs in schedules:
            events=[(f,8,-1) for f in fs]
            for a in ages:events.append((a,IDX[crop],1+int(any(f<=a-1<=f+2 for f in fs))))
            variants.append((ages[-1],tuple(events),2+ages[-1]*.6+len(fs)+4,len(fs)))
    return tuple(variants)

def animal_flow(animal,placed,day,tile=None,care=True):
    out=flows0();prod,first,interval,cap,_=ANIMALS[animal];i=IDX[prod]
    pending=tile.get('pending_care_bonus',0) if tile else 0
    if tile:
        out[day][i]+=tile.get('yield_units',0)
        out[day][8]+=int(tile.get('fertilizer_available',False))
    for d in range(day,29):
        if d>=placed:
            out[d][0]-=1
            if d>day or not tile:out[d][8]+=int(d>placed)
            age=d+1-placed
            if age>=first and (age-first)%interval==0:
                out[d+1][i]+=min(cap,1+pending);pending=0
            if care:pending+=1
    # Last morning manure, no need to feed or bank bonus beyond the season.
    if 29>max(day,placed):out[29][8]+=1
    return out

def crop_flow(crop,placed,day,variant,tile=None):
    out=flows0();duration,events,work,nfert=variant
    if tile:
        y=tile.get('yield_units',0)
        if crop in ('TOMATO','STRAWBERRY'):out[day][IDX[crop]]+=y
        for a,i,q in events:
            d=placed+a
            if d<day or d>29:continue
            if q<0 and tile.get('fertilized_until_day',-1)>=d:continue
            if i==IDX[crop] and crop in ('WHEAT','CARROT','MELON'):
                age=day-placed;target=a
                if age>target:continue
                # Real one-time crop yields take precedence over nominal history.
                q=y
                for dd in range(day,d+1):
                    aa=dd-placed
                    if (crop=='WHEAT' and 2<=aa<=4) or (crop=='CARROT' and 2<=aa<=3) or (crop=='MELON' and 6<=aa<=12):
                        if dd==day and tile.get('watered_today'):continue
                        fert=tile.get('fertilized_until_day',-1)>=dd or any(j==8 and k<0 and f<=aa<=f+2 for f,j,k in events)
                        q=min(6 if crop!='CARROT' else 4,q+1+int(fert))
            if i==IDX[crop] and crop in ('TOMATO','STRAWBERRY') and d==day:continue
            out[d][i]+=q
    else:
        for a,i,q in events:
            d=placed+a
            if day<=d<=29:out[d][i]+=q
    return out

class Economy:
    def __init__(self,obs,cfg):
        self.obs=obs;self.cfg=cfg;self.day=obs['day'];self.dem=demand_path(obs,cfg)
        self.inv=[obs['market']['inventory'][p] for p in PRODUCTS];self.params=obs['market'].get('params')
        self.stock=[sum(inv.get(p,0) for inv in [obs['private']['shed']]+obs['private']['inventories']) for p in PRODUCTS]
    def value(self,flow):
        inv=list(self.inv);cash=0;prices=[[0.0]*9 for _ in range(30)]
        for d in range(self.day,30):
            for i in range(9):
                # Daily forecast nets own input use and own production, including initial inventory.
                q=flow[d][i]+(self.stock[i] if d==self.day else 0)
                inv[i]-=self.dem[d][i]*.5
                mid=inv[i]+q*.5
                px=price(i,mid,self.params)
                # Quadrature handles nonlinear gluts without making evaluator unit-linear.
                if abs(q)>2:
                    px=(price(i,inv[i]+q*.1127017,self.params)*5+price(i,mid,self.params)*8+price(i,inv[i]+q*.8872983,self.params)*5)/18
                cash+=q*px
                inv[i]+=q
                if q>0 and px<=1.001:
                    # Official $1 floor sales no longer grow market inventory.
                    inv[i]-=q
                inv[i]-=self.dem[d][i]*.5
                prices[d][i]=price(i,inv[i],self.params)
        return cash,prices

    def new_crop_dp(self,crop,prices,start=None):
        """Bellman recursion over planting/harvest dates. Future replant is optional.
        F[t] = max_cycle (revenue - seed - labor + F[t+duration+1]).
        The selected first cycle is executed; no fixed action sequence is stored.
        """
        start=self.day if start is None else start
        values=[0.0]*32;choices=[None]*30
        for d in range(29,start-1,-1):
            best=0.0
            for variant in cycles(crop,SETTINGS['fertilize']):
                length,events,work,nfert=variant
                sell=[(a,i,q) for a,i,q in events if q>0 and d+a<=29]
                if not sell:continue
                end=min(29,d+length)
                val=-COST[crop]-SETTINGS['labor_price']*work-SETTINGS['land_rent']*(end-d)
                for a,i,q in events:
                    if d+a<=29:val+=q*prices[d+a][i]
                nxt=end+1
                if SETTINGS['crop_repeat']:val+=values[nxt]
                if val>best:best=val;choices[d]=variant
            values[d]=best
        variant=choices[start]
        if variant is None:return None
        out=crop_flow(crop,start,self.day,variant);cost=COST[crop];work=variant[2]
        if SETTINGS['crop_repeat']:
            d=start+variant[0]+1
            while d<=29 and choices[d]:
                v=choices[d];addflow(out,crop_flow(crop,d,self.day,v));cost+=COST[crop];work+=v[2];d+=v[0]+1
        return out,cost,work,variant


def asset_work(item,xy,variant=None):
    distance=dist(xy,access(xy))
    if item in ANIMALS:return (4.7+.35*distance)*SETTINGS['animal_work_mult']
    return (variant[2]/(variant[0]+1)+.60+.04*distance)*SETTINGS['crop_work_mult']

def daily_plan(obs,cfg):
    farm=obs['farms'][obs['player']];day=obs['day'];money=farm['money'];eco=Economy(obs,cfg)
    flow=flows0();assets=[];free=[];new={};policy={};nanim=0
    # Existing assets are forecast from actual ages and inventories, never from a route calendar.
    roughpx=eco.value(flow)[1]
    for y,row in enumerate(farm['tiles']):
        for x,t in enumerate(row):
            xy=(x,y)
            if t=='LOCKED':continue
            if isinstance(t,dict) and t.get('animal'):
                nanim+=1;f=animal_flow(t['animal'],t['placed_day'],day,t,SETTINGS['animal_care']);addflow(flow,f);assets.append((xy,t))
            elif isinstance(t,dict) and t.get('kind')=='PLANT':
                crop=t['crop'];age=day-t['planted_day'];variants=cycles(crop,SETTINGS['fertilize'])
                legal=[v for v in variants if v[0]>=age] or [variants[-1]]
                def quality(v):
                    f=crop_flow(crop,t['planted_day'],day,v,t)
                    return sum(f[d][i]*roughpx[d][i] for d in range(day,30) for i in range(9))-SETTINGS['labor_price']*v[2]
                variant=max(legal,key=quality)
                policy[xy]=variant;addflow(flow,crop_flow(crop,t['planted_day'],day,variant,t));assets.append((xy,t))
                nxt=t['planted_day']+variant[0]+1
                if SETTINGS['crop_repeat'] and day<nxt<29:
                    continuation=eco.new_crop_dp(crop,roughpx,nxt)
                    if continuation:addflow(flow,continuation[0])
            else:free.append(xy)
    # Near shed real estate is more valuable to daily animal service and deliveries.
    free.sort(key=lambda p:(dist(p,access(p)),p[1],p[0]))
    baseline,px=eco.value(flow)
    workload=[0.0]*30
    for xy,t in assets:
        if t.get('animal'):
            for d in range(day,30):workload[d]+=asset_work(t['animal'],xy)
        else:
            end=29 if SETTINGS['crop_repeat'] else min(29,t['planted_day']+policy[xy][0])
            for d in range(day,end+1):workload[d]+=asset_work(t['crop'],xy,policy[xy])
    reserve=SETTINGS['opening_reserve']+nanim*max(25,obs['market']['prices']['WHEAT'])*1.5
    liquid=0
    for p,q in obs['private']['shed'].items():
        if p in PRODUCTS:liquid+=q*obs['market']['prices'][p]
    liquid+=sum(t.get('fertilizer_available',False) for xy,t in assets)*obs['market']['prices']['FERTILIZER']
    budget=max(0,money+SETTINGS['advance_revenue']*liquid-reserve)
    # Honor already purchased, unfinished projects across day boundaries.
    pending=MEM.get('new',{})
    stockanimals={a:obs['private']['shed'].get(a,0)+sum(v.get(a,0) for v in obs['private']['inventories']) for a in ANIMALS}
    for xy,item in pending.items():
        if xy not in free:continue
        if item in ANIMALS and stockanimals[item]>0:
            f=animal_flow(item,day,day,care=SETTINGS['animal_care']);new[xy]=item;free.remove(xy);addflow(flow,f);stockanimals[item]-=1;nanim+=1
    # Any orphaned livestock are sunk assets to place, never a reason to buy more.
    for item,count in stockanimals.items():
        for _ in range(count):
            if not free:break
            xy=free.pop(0);new[xy]=item;addflow(flow,animal_flow(item,day,day,care=SETTINGS['animal_care']));nanim+=1
    baseline,px=eco.value(flow)
    limit=min(len(free),20)
    for it in range(limit):
        best=None
        for item in COST:
            if COST[item]>budget:continue
            if item in ANIMALS:
                first=ANIMALS[item][1]
                if day+first>26:continue
                f=animal_flow(item,day,day,care=SETTINGS['animal_care']);cost=COST[item]
                work=(29-day)*3.5+5;variant=None
            else:
                info=eco.new_crop_dp(item,px)
                if info is None:continue
                f,cost,work,variant=info
            addflow(flow,f)
            val,nextpx=eco.value(flow)
            addflow(flow,f,-1)
            if item in ANIMALS:
                dailywork=[asset_work(item,free[0]) if d>=day else 0 for d in range(30)]
            else:
                dailywork=[asset_work(item,free[0],variant) if day<=d<=(29 if SETTINGS['crop_repeat'] else min(29,day+variant[0])) else 0 for d in range(30)]
            def wages(w):
                h=max(0,w/SETTINGS['labor_hours']-1)
                return max(0,(1.618034**(h+2)-1)/2.236068-1)
            wcost=sum(wages(workload[d]+dailywork[d])-wages(workload[d]) for d in range(day,30))
            marginal=val-baseline-cost-(wcost if SETTINGS['labor_capacity'] else SETTINGS['labor_price']*work)
            duration=(29-day) if item in ANIMALS else min(29-day,variant[0])
            marginal-=SETTINGS['land_rent']*duration
            if marginal<=0:continue
            # Early liquidity matters: a positive NPV must still finance its inputs.
            financing=COST[item]+(100 if item in ANIMALS else 0)
            exponent=SETTINGS['capital_power']*max(0,1-money/16000)
            score=marginal/(financing**exponent)
            if best is None or score>best[0]:best=(score,item,f,cost,work,variant,val,nextpx,marginal,dailywork)
        if best is None:break
        _,item,f,cost,work,variant,baseline,px,marginal,dailywork=best
        for d in range(day,30):workload[d]+=dailywork[d]
        xy=free.pop(0);new[xy]=item
        if variant:policy[xy]=variant
        addflow(flow,f);budget-=COST[item]
        if item in ANIMALS:
            budget-=obs['market']['prices']['WHEAT']*1.5;nanim+=1
    # Resource-affine placement: recurrent input consumers use proximal plots;
    # low-frequency crops occupy distant plots. No map coordinates are scripted.
    coords=sorted(new,key=lambda p:(dist(p,access(p)),p[1],p[0]))
    projects=sorted(new.items(),key=lambda z:(0 if z[1] in ANIMALS else 1,0 if z[1]=='WHEAT' else 1))
    replacement={};relocated_policy=dict(policy)
    for xy,(oldxy,item) in zip(coords,projects):
        replacement[xy]=item
        if item in CROPS:relocated_policy[xy]=policy[oldxy]
    new=replacement;policy=relocated_policy
    # Expansion is conditional on current usable free space and surviving investment horizon.
    land=False
    if not free and len(farm['unlocked_quadrants'])<4 and day<20:
        landprice=(1000,2000,4000)[len(farm['unlocked_quadrants'])-1]
        if budget>landprice+700:land=True
    MEM['new']=new;MEM['policy']=policy;MEM['forecast']=flow;MEM['prices']=px;MEM['land']=land;MEM['routes']=None;MEM['route_count']=-1
    MEM['targets']={};MEM['day']=day
    MEM['daily_records'].append({'day':day,'money':money,'new':[[*xy,i] for xy,i in new.items()],'animals':nanim,'land':land})


def needs(obs,xy,t):
    """Independent operations; an unavailable input never hides harvest or manure."""
    day=obs['day'];h=obs['hour'];terminal=day==29
    out=[]
    new=MEM['new'].get(xy)
    px=obs['market']['prices']
    if isinstance(t,dict) and t.get('animal'):
        a=t['animal'];prod,first,interval,cap,_=ANIMALS[a];age=day-t['placed_day'];q=t.get('yield_units',0)
        nexttick=age+1>=first and (age+1-first)%interval==0
        if q and (q>=cap-1 or terminal or (nexttick and q+1+t.get('pending_care_bonus',0)>cap)):
            out.append(('HARVEST',None,300+q*px[prod],True))
        if t.get('fertilizer_available'):
            out.append(('COLLECT_FERTILIZER',None,max(15,px['FERTILIZER']),True))
        if not terminal and not t.get('fed_today'):
            out.append(('FEED','WHEAT',350+100*t.get('consecutive_unfed',0),True))
        # Can bank only into a production event that occurs before terminal morning.
        nextage=first if age<first else first+((age-first)//interval+1)*interval
        if SETTINGS['animal_care'] and not t.get('cared_today') and t['placed_day']+nextage<=29:
            out.append(('CARE',None,max(20,px[prod]),False))
        return sorted(out,key=lambda z:0 if z[0]=='FEED' and t.get('consecutive_unfed',0)>=1 else 1)
    if isinstance(t,dict) and t.get('kind')=='PLANT':
        crop=t['crop'];age=day-t['planted_day'];q=t.get('yield_units',0);v=MEM['policy'].get(xy,cycles(crop)[-1])
        length,events,work,nfert=v
        if crop in ('TOMATO','STRAWBERRY'):
            first=8 if crop=='TOMATO' else 10;interval=1 if crop=='TOMATO' else 2
            tick=age+1>=first and (age+1-first)%interval==0 and age+1<=first+3*interval
            deadline=t.get('max_lifespan_step',-1)
            ready=q>0 and (q>=3 or terminal or deadline>=0 or (tick and q>=2))
            water=not terminal and (t.get('consecutive_unwatered',0)>=1 or age==0 or (tick and (t.get('fertilized_until_day',-1)>=day or any(a==age and i==8 for a,i,q0 in events))))
        else:
            cap=4 if crop=='CARROT' else 6
            window=(2<=age<=4) if crop=='WHEAT' else ((2<=age<=3) if crop=='CARROT' else (6<=age<=12))
            first=10 if crop=='MELON' else 2
            ready=q>0 and age>=first and (age>=length or terminal or q>=cap)
            water=(t.get('consecutive_unwatered',0)>=1 or age==0 or (window and q<cap)) and not (ready and not window) and not (terminal and not ready)
        fert=SETTINGS['fertilize'] and not terminal and any(a==age and i==8 and q0<0 for a,i,q0 in events) and t.get('fertilized_until_day',-1)<day
        if fert:out.append(('FERTILIZE','FERTILIZER',max(80,px[crop]*1.2),False))
        if water and not t.get('watered_today'):
            out.append(('WATER',None,450 if age==0 or t.get('consecutive_unwatered',0)>=1 else 110,True))
        if ready:out.append(('HARVEST',None,300+q*px[crop],True))
        return out
    if new:
        if t is not None and not (isinstance(t,dict) and t.get('kind') in ('COOP','PASTURE') and new in ANIMALS and t['kind']==ANIMALS[new][4]):
            out.append(('DIG',None,150,False))
        elif new in ANIMALS:
            if t is None:out.append(('BUILD_'+ANIMALS[new][4],None,180,False))
            else:out.append(('PLACE',new,250,False))
        else:out.append(('PLANT',new,160,False))
    return out


def market_orders(obs,cfg,added_stock=None):
    farm=obs['farms'][obs['player']];shed=dict(obs['private']['shed']);day=obs['day'];hour=obs['hour'];orders=[]
    if added_stock:
        for p,n in added_stock.items():shed[p]=shed.get(p,0)+n
    cash=farm['money'];tiles=farm['tiles'];nanim=sum(isinstance(t,dict) and bool(t.get('animal')) for row in tiles for t in row)
    ntarget=sum(i in ANIMALS for xy,i in MEM['new'].items() if not (isinstance(tiles[xy[1]][xy[0]],dict) and tiles[xy[1]][xy[0]].get('animal')))
    needed_fert=0;needed_feed=0
    for y,row in enumerate(tiles):
        for x,t in enumerate(row):
            for op,res,val,mandatory in needs(obs,(x,y),t):
                if op=='FERTILIZE':needed_fert+=1
                if op=='FEED':needed_feed+=1
    carried={p:sum(v.get(p,0) for v in obs['private']['inventories']) for p in ('WHEAT','FERTILIZER')}
    # Inputs reserved independently from saleable output. Stocks in workers count once.
    keepfeed=max(0,needed_feed+ntarget-carried['WHEAT']) if day<29 else 0
    keepfert=max(0,needed_fert-carried['FERTILIZER'])
    if hour>0 and MEM.get('routes'):
        keepfeed=0;keepfert=0
        for u,route in enumerate(MEM['routes']):
            req=route_inputs(obs,route);inv=obs['private']['inventories'][u] if u<len(obs['private']['inventories']) else {}
            keepfeed+=max(0,req.get('WHEAT',0)-inv.get('WHEAT',0))
            keepfert+=max(0,req.get('FERTILIZER',0)-inv.get('FERTILIZER',0))
    # Preserve tomorrow's available fertilizer only when its use is already scheduled.
    if hour>=18 and day<29:
        tomorrow=0
        for (x,y),v in MEM['policy'].items():
            t=tiles[y][x]
            if isinstance(t,dict) and t.get('kind')=='PLANT':
                age=day+1-t['planted_day']
                tomorrow+=sum(1 for a,i,q in v[1] if a==age and i==8 and q<0)
        keepfert=max(keepfert,tomorrow)
    for p in PRODUCTS:
        keep=keepfeed if p=='WHEAT' else keepfert if p=='FERTILIZER' else 0
        q=max(0,shed.get(p,0)-keep)
        if q:
            orders.append(['SELL',p,q]);cash+=sum(price(IDX[p],obs['market']['inventory'][p]+k,obs['market'].get('params')) for k in range(q));shed[p]-=q
    # Workers are selected from work volume, not hard-coded by game day.
    nodes=[];work=0
    for y,row in enumerate(tiles):
        for x,t in enumerate(row):
            ops=needs(obs,(x,y),t)
            if ops:
                nodes.append((x,y));work+=len(ops)
    work+=sum(3 if item in ANIMALS else 1 for xy,item in MEM['new'].items() if not (isinstance(tiles[xy[1]][xy[0]],dict) and (tiles[xy[1]][xy[0]].get('animal') or tiles[xy[1]][xy[0]].get('kind')=='PLANT')))
    if hour<=3:
        target=MEM.get('hire_target',3)
        while len(farm['hands'])+sum(o[0]=='HIRE' for o in orders)<target and len(orders)<cfg.get('maxMarketOrdersPerTurn',10):
            n=len(farm['hands'])+sum(o[0]=='HIRE' for o in orders);c=FIB[min(n,len(FIB)-1)]*cfg.get('farmHandCostMult',1)
            if cash<c+30:break
            orders.append(['HIRE']);cash-=c
    # Legal purchases enter the shed after unit actions, so never used this turn.
    lack=max(0,keepfeed-shed.get('WHEAT',0))
    if lack and day<29 and len(orders)<10:
        n=min(lack,int(cash/max(1,obs['market']['prices']['WHEAT']+1)))
        if n>0:orders.append(['BUY_PRODUCT','WHEAT',n]);cash-=n*(obs['market']['prices']['WHEAT']+1)
    lack=max(0,keepfert-shed.get('FERTILIZER',0))
    # Manure is useful on site, but input availability cannot rely on an unassigned collection.
    if lack and hour<17 and len(orders)<10:
        n=min(lack,int(max(0,cash-60)/max(1,obs['market']['prices']['FERTILIZER']+1)))
        if n>0:orders.append(['BUY_PRODUCT','FERTILIZER',n]);cash-=n*(obs['market']['prices']['FERTILIZER']+1)
    demand={}
    for (x,y),item in MEM['new'].items():
        t=tiles[y][x]
        if isinstance(t,dict) and (t.get('animal') or t.get('kind')=='PLANT'):continue
        demand[item]=demand.get(item,0)+1
    for item,n in demand.items():
        have=(obs['private']['seeds'].get(item,0) if item in CROPS else shed.get(item,0)+sum(v.get(item,0) for v in obs['private']['inventories']))
        q=min(max(0,n-have),int(max(0,cash-35)/COST[item]))
        if q and len(orders)<10:
            orders.append(['BUY_SEED' if item in CROPS else 'BUY_ANIMAL',item,q]);cash-=COST[item]*q
    if MEM['land'] and hour<4 and len(orders)<10:
        c=(1000,2000,4000)[len(farm['unlocked_quadrants'])-1] if len(farm['unlocked_quadrants'])<4 else 999999
        if cash>=c+200:orders.append(['BUY_LAND']);MEM['land']=False
    return orders[:cfg.get('maxMarketOrdersPerTurn',10)]


def pick_op(ops,inventory,seeds):
    for op,res,value,mandatory in ops:
        if op=='PLANT' and seeds.get(res,0)<1:continue
        if op in ('FEED','FERTILIZE','PLACE') and inventory.get(res,0)<1:continue
        return (op,res,value,mandatory)
    return None


def planned_nodes(obs):
    farm=obs['farms'][obs['player']];out={};goods=0
    for y,row in enumerate(farm['tiles']):
        for x,t in enumerate(row):
            xy=(x,y);ops=needs(obs,xy,t)
            if not ops:continue
            work=len(ops)
            item=MEM['new'].get(xy)
            active=isinstance(t,dict) and (t.get('kind')=='PLANT' or t.get('animal'))
            if item and not active:
                if item in ANIMALS:work+=3 if ops[0][0].startswith('BUILD') else (2 if ops[0][0]=='PLACE' else 4)
                else:work+=1 if ops[0][0]=='PLANT' else 2
            for op,res,v,m in ops:
                if op=='HARVEST':goods+=t.get('yield_units',0)
                if op=='COLLECT_FERTILIZER':goods+=1
            out[xy]=work
    return out,goods

def cover_route(obs,start,nodes,last=False,bank=False):
    """Exact local subset-path DP with an end-of-day service deadline.
    Daily tours are assembled from jobs; no farmer owns a production lifecycle.
    """
    tiles=obs['farms'][obs['player']]['tiles']
    def priority(xy):
        t=tiles[xy[1]][xy[0]]
        if isinstance(t,dict) and t.get('animal'):return 3
        if isinstance(t,dict) and t.get('kind')=='PLANT':return 2
        return 1
    anchor=max(nodes,key=lambda xy:(priority(xy),dist(xy,access(xy))))
    near=sorted((xy for xy in nodes if xy!=anchor),key=lambda xy:dist(xy,anchor)+.05*dist(xy,access(xy)))[:8]
    pts=[anchor]+near;n=len(pts);size=1<<n
    weights=[nodes[xy]+2 for xy in pts]
    values=[0.0]*size;work=[nodes[xy] for xy in pts]
    # Budget includes physical delivery on terminal day, not paper mark-to-market.
    distance=[[dist(a,b) for b in pts] for a in pts]
    dp=[[99]*n for _ in range(size)];parent=[[-1]*n for _ in range(size)]
    for j,p in enumerate(pts):dp[1<<j][j]=dist(start,p)+work[j]
    best=None;score=-1
    for mask in range(1,size):
        bit=mask&-mask;j=bit.bit_length()-1;values[mask]=values[mask^bit]+weights[j]
        for j in range(n):
            t=dp[mask][j]
            if t>20:continue
            end=t+(dist(pts[j],access(pts[j]))+1 if last or bank else 0)
            if mask&1 and end<=(19 if last else 20):
                val=values[mask]-.015*end
                if val>score:score=val;best=(mask,j)
            avail=(size-1)^mask
            while avail:
                low=avail&-avail;k=low.bit_length()-1;avail^=low
                tt=t+distance[j][k]+work[k]
                mm=mask|low
                if tt<dp[mm][k] and tt<=20:dp[mm][k]=tt;parent[mm][k]=j
    if best is None:return [anchor]
    mask,j=best;route=[]
    while j>=0:
        route.append(pts[j]);prev=parent[mask][j];mask^=1<<j;j=prev
    route.reverse();return route

def schedule(obs):
    if MEM.get('routes') is not None:return
    nodes,goods=planned_nodes(obs);last=obs['day']==29
    farm=obs['farms'][obs['player']];starts=[tuple(farm['farmer'])]+[tuple(p) for p in farm['hands']]
    routes=[]
    while nodes and len(routes)<=SETTINGS['max_hands']:
        u=len(routes);start=starts[u] if u<len(starts) else (4+(u%2),4+((u//2)%2))
        bank=SETTINGS['delivery_planning'] and goods>80
        rr=cover_route(obs,start,nodes,last,bank)
        for xy in rr:
            t=farm['tiles'][xy[1]][xy[0]]
            for op,res,val,mandatory in needs(obs,xy,t):
                if op=='HARVEST':goods-=t.get('yield_units',0)
                if op=='COLLECT_FERTILIZER':goods-=1
        routes.append(rr)
        for xy in rr:nodes.pop(xy,None)
    MEM['backlog']=list(nodes);MEM['hire_target']=max(2,len(routes)-1)
    while len(routes)<MEM['hire_target']+1:routes.append([])
    MEM['routes']=routes;MEM['route_count']=len(routes);MEM['need_return']=last;MEM['couriers']=set()

def route_inputs(obs,route):
    req={};tiles=obs['farms'][obs['player']]['tiles']
    for xy in route:
        t=tiles[xy[1]][xy[0]];item=MEM['new'].get(xy)
        if item in ANIMALS and not (isinstance(t,dict) and t.get('animal')):
            req[item]=req.get(item,0)+1
            if obs['day']<29:req['WHEAT']=req.get('WHEAT',0)+1
        else:
            for op,res,v,m in needs(obs,xy,t):
                if op in ('FEED','FERTILIZE','PLACE'):req[res]=req.get(res,0)+1
    return req

def execute(obs,cfg):
    schedule(obs)
    if obs['hour']==4:
        count=1+len(obs['farms'][obs['player']]['hands'])
        for rr in MEM['routes'][count:]:MEM['backlog'].extend(rr)
        MEM['routes']=MEM['routes'][:count]
    farm=obs['farms'][obs['player']];tiles=farm['tiles'];positions=[farm['farmer']]+farm['hands'];inventories=obs['private']['inventories'];hour=obs['hour'];last=obs['day']==29
    seeds=dict(obs['private']['seeds']);shed=dict(obs['private']['shed']);n=len(tiles);actions=[];deposit={};claimed=set()
    routes=MEM['routes']
    # Routes contain only intentions. Every operation is recomputed from the observation.
    for u,route in enumerate(routes):
        routes[u]=[xy for xy in route if needs(obs,xy,tiles[xy[1]][xy[0]])]
    totalcarry=sum(sum(v.values()) for v in inventories)
    courier=MEM.get('couriers',set())
    courier.intersection_update(range(len(positions)))
    payload={}
    for j,(posj,invj) in enumerate(zip(positions,inventories)):
        rj=route_inputs(obs,routes[j]);payload[j]=sum(max(0,invj.get(p,0)-rj.get(p,0)) for p in PRODUCTS)
        if payload[j]==0:courier.discard(j)
    _,uncollected=planned_nodes(obs)
    excess=max(0,totalcarry+uncollected-90)
    promised=sum(payload[j] for j in courier)
    if not last and hour>=7 and excess>promised:
        options=[]
        for j,pj in enumerate(positions):
            if j in courier or payload[j]<3:continue
            dh=dist(pj,access(pj));route=routes[j]
            remaining=sum(len(needs(obs,xy,tiles[xy[1]][xy[0]])) for xy in route)
            walking=0;pp=access(pj)
            for xy in route:walking+=dist(pp,xy);pp=xy
            if hour+dh+1+remaining+walking>24:continue
            options.append((payload[j]/(1+2*dh),j))
        for _,j in sorted(options,reverse=True):
            if promised>=excess:break
            courier.add(j);promised+=payload[j]
    MEM['couriers']=courier
    for u,(pos,inv) in enumerate(zip(positions,inventories)):
        pos=tuple(pos);depot=access(pos,n);dhome=dist(pos,depot);route=routes[u]
        req=route_inputs(obs,route)
        surplus={p:max(0,inv.get(p,0)-req.get(p,0)) for p in PRODUCTS}
        goods=sum(surplus.values())
        # Returning is a schedulable physical operation; no end-of-season paper liquidation.
        must_return=goods and ((u in courier) or (last and hour+dhome>=21) or (not route and hour+dhome<=22) or (farm['money']<100 and hour>5 and dhome<4))
        if must_return and hour+dhome<=22:
            if dhome:act=move(pos,depot)
            else:
                room=cfg.get('shedCapacity',100)-sum(shed.values())
                if sum(inv.values())<=room and not any(inv.get(p,0) and q for p,q in req.items()):
                    act=['DROP']
                    for p,q in inv.items():deposit[p]=deposit.get(p,0)+q;shed[p]=shed.get(p,0)+q
                else:
                    items=[(p,q) for p,q in surplus.items() if q>0]
                    if items and room>0:
                        p,q=max(items,key=lambda z:obs['market']['prices'][z[0]]);q=min(q,room);act=['PLACE',p,q];deposit[p]=deposit.get(p,0)+q;shed[p]=shed.get(p,0)+q
                    else:act=['PASS']
            actions.append(act);continue
        if not route and MEM.get('backlog'):
            candidates=[xy for xy in MEM['backlog'] if needs(obs,xy,tiles[xy[1]][xy[0]]) and hour+dist(pos,xy)+len(needs(obs,xy,tiles[xy[1]][xy[0]]))<(22 if last else 24)]
            if candidates:
                xy=min(candidates,key=lambda xy:dist(pos,xy));route.append(xy);MEM['backlog'].remove(xy)
        if not route:
            # Free workers can take pending visits from another worker. Ownership is not a prerequisite.
            candidates=[]
            for v,rr in enumerate(routes):
                if v==u or not rr or v>=len(positions):continue
                for j,xy in enumerate(rr):
                    if xy in claimed:continue
                    if j==0 and dist(positions[v],xy)<=dist(pos,xy):continue
                    if hour+dist(pos,xy)+2<(22 if last else 24):candidates.append((dist(pos,xy),v,j,xy))
            if candidates:
                _,v,j,xy=min(candidates);routes[v].pop(j);route.append(xy)
        req=route_inputs(obs,route)
        missing={p:max(0,q-inv.get(p,0)) for p,q in req.items() if q>inv.get(p,0)}
        if dhome==0:
            options=[p for p in (*ANIMALS,'WHEAT','FERTILIZER') if missing.get(p,0)>0 and shed.get(p,0)>0]
            if options:
                p=options[0];q=min(missing[p],shed[p]);shed[p]-=q;actions.append(['PICKUP',p,q]);continue
            # Wait one turn for market purchases, not an entire discarded task group.
            if missing and hour<3 and farm['money']>50:
                actions.append(['PASS']);continue
        if not route:
            actions.append(['PASS']);continue
        xy=route[0];t=tiles[xy[1]][xy[0]];ops=needs(obs,xy,t);choice=pick_op(ops,inv,seeds)
        if choice is None:
            # A missing feed/seed does not cancel collection or harvesting elsewhere.
            resource_available=any(shed.get(p,0)>0 for p in missing)
            if resource_available and hour+dhome+2<24:
                actions.append(move(pos,depot));continue
            alternatives=[]
            for j,other in enumerate(route[1:],1):
                if pick_op(needs(obs,other,tiles[other[1]][other[0]]),inv,seeds):alternatives.append((dist(pos,other),j))
            if alternatives:
                _,j=min(alternatives);route.insert(0,route.pop(j));xy=route[0];choice=pick_op(needs(obs,xy,tiles[xy[1]][xy[0]]),inv,seeds)
            else:
                if goods and hour+dhome<=22:
                    if dhome:act=move(pos,depot)
                    else:
                        act=['DROP']
                        for p,q in inv.items():deposit[p]=deposit.get(p,0)+q;shed[p]=shed.get(p,0)+q
                else:act=['PASS']
                actions.append(act);continue
        if choice is None:actions.append(['PASS']);continue
        op,res,val,mandatory=choice
        if op=='PLANT' and hour+dist(pos,xy)+2>24:actions.append(['PASS']);continue
        if last and op=='HARVEST' and hour+dist(pos,xy)+dist(xy,access(xy))+2>23:actions.append(['PASS']);continue
        claimed.add(xy)
        if pos!=xy:actions.append(move(pos,xy));continue
        act=[op]
        if op in ('PLANT','PLACE'):act.append(res)
        if op=='PLANT':seeds[res]-=1
        actions.append(act)
    return {'farmer':actions[0],'hands':actions[1:],'market':market_orders(obs,cfg,deposit)}


def agent(observation,configuration=None):
    obs=observation;cfg=configuration or {};step=obs.get('step',obs['day']*24+obs['hour'])
    if step==0 or not MEM or obs['day']<MEM.get('day',-1):
        MEM.clear();MEM['day']=-1;MEM['daily_records']=[]
    if obs['day']!=MEM['day']:daily_plan(obs,cfg)
    return execute(obs,cfg)
