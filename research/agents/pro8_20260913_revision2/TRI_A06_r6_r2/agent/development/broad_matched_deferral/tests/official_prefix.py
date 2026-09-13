"""No games: compare one own unit+market prefix against frozen official rules.
Other orders are absent by construction. Stops before town/decay/day/randomness.
Does not assert a forecast or saved post-divergence observation is real future.
"""
import copy, importlib.util

def load_referee(fixtures):
    path=fixtures/'referee/cpu_runtime.py'
    spec=importlib.util.spec_from_file_location('r2_frozen_referee',path)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    return runtime,runtime.load_engine()

def decode_action(raw,codec):
    def atom(a):
        op,item,quantity=a;name=codec._OPS[op];v=[name]
        if item>=0:
            v.append(codec._ITEMS[item])
            if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):v.append(quantity)
        elif quantity!=1:v.append(quantity)
        return v
    return {'farmer':atom(raw['units'][0]),'hands':[atom(x) for x in raw['units'][1:]],'market':[atom(x) for x in raw['market']]}

def own_signature(ob,codec):
    z=list(codec._pack(ob));f0=4;f1=f0+6+2*len(ob['farms'][0]['hands'])+1500
    priv=f1+6+2*len(ob['farms'][1]['hands'])+1500
    own=(f0,f1) if ob['player']==0 else (f1,priv)
    # Town is not touched by the bounded prefix. Everything else uses exactly
    # the same observation codec consumed by the real production entry point.
    town=1+len(ob['town']['unlocked_shops'])
    return z[own[0]:own[1]]+z[priv:len(z)-town]

def check_prefix(ob,configuration,branch,codec,runtime,engine):
    ob=copy.deepcopy(ob);seat=ob['player'];act=decode_action(branch['actions'][0],codec)
    farm,private=ob['farms'][seat],ob['private']
    units=[act['farmer'],*act['hands']];counts={}
    for x in units:
        if x[0]=='PLANT':counts[x[1]]=counts.get(x[1],0)+1
    blocked={crop for crop,n in counts.items() if n>private['seeds'].get(crop,0)}
    cfg=runtime.AttrDict(configuration)
    for i,x in enumerate(units):
        if x[0]=='PLANT' and x[1] in blocked:continue
        engine._apply_unit_action(farm,private,i,x,10,ob['day'],24,configuration.get('shedCapacity',100))
    # No opponent private state was supplied or reconstructed. An empty mapping
    # is sufficient because there are explicitly no opponent orders in this test.
    state=[runtime.AttrDict(observation=runtime.AttrDict(farms=ob['farms'],market=ob['market'],private=private if p==seat else {}),action=act if p==seat else {}) for p in range(2)]
    engine._process_market(state,runtime.AttrDict(configuration=cfg))
    actual=own_signature(ob,codec);expected=branch['first_prefix_signature']
    differences=[{'index':i,'official':x,'native':y} for i,(x,y) in enumerate(zip(actual,expected)) if x!=y]
    result={'step':ob['step'],'candidate_id':branch['id'],'wait':branch['wait'],'official_values':len(actual),'native_values':len(expected),'differences':differences,'cash_after_prefix':farm['money'],'land_after_prefix':farm['unlocked_quadrants'],'action':act,'scope':'Single own unit+market prefix. Other orders absent. No future RNG or match.'}
    if len(actual)!=len(expected) or differences:
        result['official_signature']=actual;result['native_signature']=expected
    result['pass']=len(actual)==len(expected) and not differences
    return result
