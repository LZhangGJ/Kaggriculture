"""Small economic feature set from one legal player observation.

No seed, opponent identity, route fingerprint, or private opponent inventory.
"""
import ctypes
import json
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP')
SHOPS=('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP','PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')


def observation_features(obs):
    day=int(obs['day']);seat=int(obs['player']);f={'day':day,'hour':int(obs['hour'])}
    for key in ('inventory','prices'):
        for item in ITEMS[:9]:f['market_'+key+'_'+item]=float(obs['market'][key][item])-(10000 if key=='inventory' else 0)
    shops=obs['town']['unlocked_shops']
    for shop in SHOPS:f['shop_'+shop]=shops.count(shop)
    for side,index in (('own',seat),('opponent',1-seat)):
        farm=obs['farms'][index];f[side+'_cash']=float(farm['money'])
        f[side+'_land']=len(farm['unlocked_quadrants']);f[side+'_workers']=len(farm['hands'])+1
        for item in ITEMS:
            for kind in ('count','age_sum','yield','unserved'):
                f[side+'_'+item+'_'+kind]=0
        f[side+'_empty']=f[side+'_weeds']=0
        for row in farm['tiles']:
            for tile in row:
                if tile is None:f[side+'_empty']+=1;continue
                if not isinstance(tile,dict):continue
                f[side+'_weeds']+=tile.get('kind')=='WEED'
                item=tile.get('animal') or tile.get('crop')
                if item not in ITEMS:continue
                prefix=side+'_'+item+'_';f[prefix+'count']+=1
                age=day-tile.get('placed_day' if tile.get('animal') else 'planted_day',day)
                f[prefix+'age_sum']+=age;f[prefix+'yield']+=tile.get('yield_units',0)
                f[prefix+'unserved']+=tile.get('consecutive_unfed',0) if tile.get('animal') else tile.get('consecutive_unwatered',0)
    private=obs['private']
    for item in ITEMS:
        f['own_stock_'+item]=private.get('shed',{}).get(item,0)+sum(b.get(item,0) for b in private.get('inventories',[]))
    for item in ITEMS[:5]:f['own_seed_'+item]=private.get('seeds',{}).get(item,0)
    return f


def history_features(actor):
    """Public sale timing already inferred by the actor from earlier observations."""
    actor=getattr(actor,'inner',actor)
    actor.lib.td_clock_json.argtypes=[ctypes.c_void_p]
    actor.lib.td_clock_json.restype=ctypes.c_char_p
    clock=json.loads(actor.lib.td_clock_json(actor.handle).decode())
    f={}
    for i,item in enumerate(ITEMS[:9]):
        if not 1<=i<=7:continue
        ours=clock['own'][i];theirs=clock['rival'][i]
        f['history_own_sale_days_'+item]=ours['days']
        f['history_rival_sale_days_'+item]=theirs['days']
        before=0.;early=0.
        for own,rival in zip(ours['density'],theirs['density']):
            early+=own*(before+.5*rival);before+=rival
        f['history_own_before_rival_'+item]=early
        for phase in range(4):
            f['history_rival_phase_'+str(phase)+'_'+item]=sum(theirs['density'][phase*6:phase*6+6])
    return f


def plan_features(actor,observation,codec,profiles,base):
    actor=getattr(actor,'inner',actor)
    packed=codec._pack(observation)
    configs=[dict(base,**changes) for changes in profiles.values()]
    values=(ctypes.c_double*(len(configs)*len(codec._ORDER)))(*(cfg[k] for cfg in configs for k in codec._ORDER))
    fn=actor.lib.td_profile_forecasts_json
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_size_t]
    fn.restype=ctypes.c_char_p
    raw=fn(actor.handle,packed,len(packed),values,len(configs),len(codec._ORDER)).decode()
    if raw.startswith('ERROR'):raise RuntimeError(raw)
    data=json.loads(raw)
    if len(data['profiles'])!=len(profiles):raise ValueError('Forecast count')
    selected={'candidate_predicted_value','candidate_land_target','immediate_spending','hire_target','common_objective_forecast'}
    prefixes=('forecast_','flow_difference_','target_','new_target_','buy_','sell_')
    result={}
    for name,row in zip(profiles,data['profiles']):
        if len(row)!=len(data['names']):raise ValueError('Forecast dimensions')
        for key,value in zip(data['names'],row):
            if key in selected or key.startswith(prefixes):result['plan_'+name+'_'+key]=float(value)
    return result


def economic_plan_features(context,profiles):
    """Explicit supply/demand ratios from legal forecasts, without future shops.

    Demand uses the official IID shop expectation, not a realized future town.
    The inventory projection excludes rival trades and is named accordingly.
    """
    products=((5,0),(5,0,3),(0,1,2,3),(3,6,0),(1,1),(6,2,0),(3,6),(7,7))
    day=int(context['day']);shops=sum(context['shop_'+s] for s in SHOPS)
    known=[sum(6*context['shop_'+s]*products[j].count(i) for j,s in enumerate(SHOPS)) for i in range(9)]
    average=[sum(6*p.count(i)/8 for p in products) for i in range(9)]
    out={}
    for name in profiles:
        prefix='plan_'+name+'_'
        out[prefix+'forecast_spending_cash_ratio']=context[prefix+'immediate_spending']/max(1.,context['own_cash'])
        for horizon in (1,3,7,14,30):
            for i,item in enumerate(ITEMS[:9]):
                demand=sum((1 if i<8 else 0)+known[i]+max(0,min(8,d//3)-shops)*average[i] for d in range(day,min(30,day+horizon)))
                flow=context[prefix+'forecast_flow_'+str(horizon)+'_'+str(i)]
                inventory=context['market_inventory_'+item]
                suffix=str(horizon)+'_'+str(i)
                out[prefix+'forecast_expected_demand_'+suffix]=demand
                out[prefix+'forecast_absorption_ratio_'+suffix]=flow/(1+max(0,-inventory)+demand)
                out[prefix+'forecast_own_only_net_inventory_'+suffix]=inventory+flow-demand
    return out


def policy_features(obs,config):
    f=observation_features(obs)
    # Seat is public; day-end random draws visit the two farms in seat order.
    f['own_seat']=int(obs['player'])
    # An agent knows its own settings. Later recourse models need that context.
    f.update({'current_setting_'+k:float(v) for k,v in config.items()})
    # Maturity and transport cost distinguish economically different portfolios
    # with the same asset count and summed ages. Coordinates are reduced to cost.
    first={'WHEAT':2,'CARROT':2,'TOMATO':8,'STRAWBERRY':10,'MELON':10,'GOOSE':4,'COW':8,'SHEEP':6}
    products={'GOOSE':'EGG','COW':'MILK','SHEEP':'WOOL'}
    for side,index in (('own',obs['player']),('opponent',1-obs['player'])):
        farm=obs['farms'][index]
        f[side+'_worker_depot_distance']=sum(min(abs(x-sx)+abs(y-sy) for sx,sy in ((4,4),(5,4),(4,5),(5,5))) for x,y in [farm['farmer'],*farm['hands']])
        for item in first:
            for kind in ('mature','needs_service','haul_distance','standing_value','age_0_3','age_4_7','age_8_13','age_14_plus','bonus'):
                f[side+'_'+item+'_'+kind]=0
        for y,row in enumerate(obs['farms'][index]['tiles']):
            for x,tile in enumerate(row):
                if not isinstance(tile,dict):continue
                item=tile.get('animal') or tile.get('crop')
                if item not in first:continue
                prefix=side+'_'+item+'_';age=obs['day']-tile.get('placed_day' if tile.get('animal') else 'planted_day',obs['day'])
                bucket='age_0_3' if age<4 else 'age_4_7' if age<8 else 'age_8_13' if age<14 else 'age_14_plus'
                f[prefix+bucket]+=1
                f[prefix+'bonus']+=tile.get('pending_care_bonus',0) if tile.get('animal') else tile.get('fertilized_until_day',-1)>=obs['day']
                f[prefix+'mature']+=age>=first[item]
                f[prefix+'needs_service']+=not tile.get('fed_today' if tile.get('animal') else 'watered_today',False)
                f[prefix+'haul_distance']+=min(abs(x-sx)+abs(y-sy) for sx,sy in ((4,4),(5,4),(4,5),(5,5)))
                f[prefix+'standing_value']+=tile.get('yield_units',0)*obs['market']['prices'][products.get(item,item)]
    for item in ITEMS[:9]:
        f['own_shed_'+item]=obs['private'].get('shed',{}).get(item,0)
    return f
