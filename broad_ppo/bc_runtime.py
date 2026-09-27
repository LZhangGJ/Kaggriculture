"""Causal numeric encoding and short-job execution for replay BC.

Requires the existing kgrl support package on PYTHONPATH for worker mechanics.
No opponent-private state, future labels, or teacher identity enters encode().
"""
import math
import numpy as np

ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP')
CROPS=ITEMS[:5]
ANIMALS=('GOOSE','COW','SHEEP')
SHOPS=('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP','PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')
JOB_OPS=('PLANT','BUILD_PASTURE','BUILD_COOP','PLACE','DIG')
MARKET=[('STOP',None),('NOOP',None),('HIRE',None),('BUY_LAND',None)]
MARKET += [('BUY_SEED',x) for x in CROPS]+[('BUY_PRODUCT',x) for x in ('WHEAT','FERTILIZER')]
MARKET += [('BUY_ANIMAL',x) for x in ANIMALS]+[('SELL',x) for x in ITEMS[:9]]
SCHEMA='macro-bc-numeric-v1'

def scale(n):return math.copysign(math.log1p(abs(float(n))),float(n))/12

def job_key(job):return (job['op'],job.get('item'),int(job['x']),int(job['y']))

def prune_pending(pending,board,turn):
    kept=[]
    for j in pending:
        x,y=j['x'],j['y'];op=j['op'];item=j['item']
        done=(op=='PLANT' and board[3+CROPS.index(item),y,x]>0 or
              op=='BUILD_PASTURE' and board[8,y,x]>0 or
              op=='BUILD_COOP' and board[9,y,x]>0 or
              op=='PLACE' and board[10+('COW','SHEEP','GOOSE').index(item),y,x]>0 or
              op=='DIG' and board[1,y,x]>0)
        if not done and j['deadline']>=turn:kept.append(j)
    return kept

def program_features(pending,turn):
    if len(pending)>16:raise ValueError('More than 16 pending jobs')
    programs=np.zeros((16,32),np.float32);valid=np.zeros(16,bool)
    for i,job in enumerate(pending):
        valid[i]=True;programs[i,JOB_OPS.index(job['op'])]=1
        if job.get('item') in ITEMS:programs[i,5+ITEMS.index(job['item'])]=1
        programs[i,17:20]=[job['x']/9,job['y']/9,max(0,job.get('deadline',turn)-turn)/24]
    return programs,valid

def encode(obs,pending=()):
    player=int(obs['player']);farms=[obs['farms'][player],obs['farms'][1-player]]
    boards=np.zeros((2,24,10,10),np.float32)
    for side,farm in enumerate(farms):
        b=boards[side]
        for y,row in enumerate(farm['tiles']):
            for x,t in enumerate(row):
                if t=='LOCKED':b[0,y,x]=1;continue
                if t is None:b[1,y,x]=1;continue
                kind=t.get('kind')
                if kind=='WEED':b[2,y,x]=1
                if t.get('crop') in CROPS:b[3+CROPS.index(t['crop']),y,x]=1
                if kind=='PASTURE':b[8,y,x]=1
                if kind=='COOP':b[9,y,x]=1
                if t.get('animal') in ('COW','SHEEP','GOOSE'):b[10+('COW','SHEEP','GOOSE').index(t['animal']),y,x]=1
                b[13,y,x]=scale(t.get('yield_units',0))
                b[14,y,x]=(obs['day']-t.get('planted_day',t.get('placed_day',obs['day'])))/30
                b[15,y,x]=bool(t.get('watered_today',t.get('fed_today',False)))
                b[16,y,x]=bool(t.get('cared_today',False))
                b[17,y,x]=max(0,t.get('fertilized_until_day',-1)-obs['day']+1)/4
                b[18,y,x]=t.get('consecutive_unwatered',t.get('consecutive_unfed',0))/10
                b[21,y,x]=bool(t.get('fertilizer_available',False))
                b[22,y,x]=max(0,t.get('max_lifespan_step',-1)-obs.get('step',obs['day']*24+obs['hour']))/720
                b[23,y,x]=t.get('pending_care_bonus',0)/10
        x,y=farm['farmer'];b[19,y,x]=1
        for x,y in farm['hands']:b[20,y,x]+=0.1
    private=obs['private'];carry={item:sum(inv.get(item,0) for inv in private['inventories']) for item in ITEMS}
    values=[obs['day']/30,obs['hour']/24,(719-obs.get('step',obs['day']*24+obs['hour']))/719]
    for farm in farms:
        values += [scale(farm['money']),len(farm['hands'])/20,farm.get('hires_today',0)/20,len(farm['unlocked_quadrants'])/4]
    values += [scale(private['seeds'].get(x,0)) for x in CROPS]
    values += [scale(private['shed'].get(x,0)) for x in ITEMS]+[scale(carry[x]) for x in ITEMS]
    shops=obs.get('town',{}).get('unlocked_shops',[])
    values += [shops.count(x)/10 for x in SHOPS]
    global_features=np.zeros(96,np.float32);global_features[:len(values)]=values
    market=np.zeros((12,16),np.float32)
    for i,item in enumerate(ITEMS):
        market[i,i]=1
        market[i,12]=scale(obs['market']['prices'].get(item,0))
        market[i,13]=scale(obs['market']['inventory'].get(item,0))
        market[i,14]=scale(private['shed'].get(item,0))
        market[i,15]=scale(private['seeds'].get(item,0))
    programs,valid=program_features(pending,obs.get('step',obs['day']*24+obs['hour']))
    return dict(boards=boards,global_features=global_features,market=market,programs=programs,program_valid=valid)

def market_features():
    features=np.zeros((len(MARKET),64),np.float32)
    ops=('STOP','NOOP','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL')
    for i,(op,item) in enumerate(MARKET):
        features[i,ops.index(op)]=1
        if item:features[i,8+ITEMS.index(item)]=1
        features[i,63]=1
    return features

def market_label(slot):
    order=slot['target']
    if not order:return 1,0,bool(slot['loss_mask'])
    op=order[0];item=order[1] if op not in ('HIRE','BUY_LAND') else None
    return MARKET.index((op,item)),int(order[2]) if len(order)>2 else 0,bool(slot['loss_mask'])

def market_order(index,quantity=0):
    op,item=MARKET[int(index)]
    if op=='STOP':return None
    if op=='NOOP':return []
    if item is None:return [op]
    if int(quantity)<=0:raise ValueError('Quantity must be a positive integer')
    return [op,item,int(quantity)]

class JobExecutor:
    """Persistent production jobs; routine care cannot buy, hire, or sell."""
    def __init__(self):self.pending=[]

    def add(self,jobs,turn):
        keys={job_key(j) for j in self.pending}
        for j in jobs:
            key=job_key(j)
            if key in keys:continue
            if len(self.pending)>=16:raise ValueError('Pending job capacity exceeded')
            self.pending.append({k:j[k] for k in ('op','item','x','y')}|{'deadline':turn+24})
            keys.add(key)

    def workers(self,obs):
        from kgrl.commitments.mechanics import UnitLedger,CROPS as crop_rules,ANIMALS as animal_rules,nearest_shed,move,distance
        from kgrl.contracts import Command
        ledger=UnitLedger(obs);turn=obs['step'];alive=[]
        for j in self.pending:
            tile=ledger.tile((j['x'],j['y']));op=j['op']
            done=(op=='PLANT' and isinstance(tile,dict) and tile.get('crop')==j['item'] or
                  op.startswith('BUILD_') and isinstance(tile,dict) and tile.get('kind')==op[6:] or
                  op=='PLACE' and isinstance(tile,dict) and tile.get('animal')==j['item'] or
                  op=='DIG' and tile is None)
            if not done and j['deadline']>=turn:alive.append(j)
        self.pending=alive;claimed=set();commands=[]
        for worker,pos in enumerate(ledger.positions):
            inv=ledger.inventory(worker);tasks=[]
            for j in self.pending:
                target=(j['x'],j['y']);tile=ledger.tile(target);op=j['op'];item=j['item']
                if job_key(j) in claimed:continue
                if op=='PLANT' and (tile is not None or ledger.seeds.get(item,0)<=0):continue
                if op.startswith('BUILD_') and tile is not None:continue
                if op=='PLACE' and (not isinstance(tile,dict) or 'animal' in tile or tile.get('kind')!=animal_rules[item][1]):continue
                cmd=Command(op,item=item) if item else Command(op)
                if op=='PLACE' and inv.get(item,0)<=0:
                    if ledger.shed.get(item,0)<=0:continue
                    target=nearest_shed(pos);cmd=Command('PICKUP',item=item,quantity=1)
                tasks.append((distance(pos,target),target,cmd,job_key(j)))
            # Preserve existing assets while production jobs travel or wait.
            for y,row in enumerate(ledger.farm['tiles']):
                for x,t in enumerate(row):
                    if not isinstance(t,dict):continue
                    target=(x,y);options=[]
                    if t.get('crop') in crop_rules:
                        if not t.get('watered_today'):options.append(Command('WATER'))
                        if t.get('yield_units',0)>0 and obs['day']-t['planted_day']>=crop_rules[t['crop']][1]:options.append(Command('HARVEST'))
                    if t.get('animal') in animal_rules:
                        if not t.get('fed_today'):
                            if inv.get('WHEAT',0)>0:options.append(Command('FEED'))
                            elif ledger.shed.get('WHEAT',0)>0:
                                shed=nearest_shed(pos);tasks.append((distance(pos,shed)+1,shed,Command('PICKUP',item='WHEAT',quantity=1),('pickup',worker)))
                        if t.get('yield_units',0)>0:options.append(Command('HARVEST'))
                        if not t.get('cared_today'):options.append(Command('CARE'))
                        if t.get('fertilizer_available'):options.append(Command('COLLECT_FERTILIZER'))
                    for cmd in options:
                        key=(cmd.op,x,y)
                        if key not in claimed:tasks.append((distance(pos,target)+2,target,cmd,key))
            # Carrying output back is execution, not a hidden sale decision.
            hungry=any(isinstance(t,dict) and t.get('animal') and not t.get('fed_today') for row in ledger.farm['tiles'] for t in row)
            deliverable=any(inv.get(x,0)>0 for x in ITEMS[:9] if x!='WHEAT') or (inv.get('WHEAT',0)>0 and not hungry)
            if deliverable and not (inv.get('WHEAT',0)>0 and hungry):
                target=nearest_shed(pos)
                if ledger.room()>0:tasks.append((distance(pos,target),target,Command('DROP'),('drop',worker)))
            if tasks:
                _,target,cmd,key=min(tasks,key=lambda x:x[0]);claimed.add(key)
                if pos!=target:cmd=move(pos,target)
            else:cmd=Command('PASS')
            ledger.apply(worker,cmd)
            wire=[cmd.op]
            if cmd.item is not None:wire.append(cmd.item)
            if cmd.quantity is not None:wire.append(cmd.quantity)
            commands.append(wire)
        # Reserve seeds sequentially, so the official atomic PLANT rule cannot
        # invalidate a crop through over-requesting in the joint worker action.
        return {'farmer':commands[0],'hands':commands[1:],'market':[]}
