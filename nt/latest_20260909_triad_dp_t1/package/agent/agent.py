"""C++ competitive daily DP with the disclosed, team-owned J7 reference candidate.
No rival name, true seed, future shop sequence or rival private stock is read.
Call sequentially, once per game tick. Rebuild agent.so using build_agent.py.
"""
import ctypes
import json
from pathlib import Path

_ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP')
_IDS={v:i for i,v in enumerate(_ITEMS)}
_SHOPS=('BAKERY','BRUNCH_SPOT','FARMERS_MARKET','ICE_CREAM_SHOP','PET_CAFE','PIZZA_SHOP','SMOOTHIE_SHOP','YARN_STORE')
_SHOP_IDS={v:i for i,v in enumerate(_SHOPS)}
_QUADS={'NW':1,'NE':2,'SW':4,'SE':8}
_KINDS={'EMPTY':0,'SOIL':0,'LOCKED':1,'WEED':2,'PLANT':3,'COOP':4,'PASTURE':5}
_OPS=('PASS','NORTH','SOUTH','EAST','WEST','DROP','PICKUP','PLACE','PLANT','WATER','HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','FEED','COLLECT_FERTILIZER','CARE','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL')
_LIB=None

def _get(obj,key,default=None):
    if isinstance(obj,dict):return obj.get(key,default)
    return getattr(obj,key,default)

def _pack(obs):
    step=int(_get(obs,'step',0));seat=int(_get(obs,'player',0))
    values=[step,int(_get(obs,'day',step//24)),int(_get(obs,'hour',step%24)),seat]
    farms=_get(obs,'farms',[])
    if len(farms)!=2:raise ValueError('Expected two public farms')
    for farm in farms:
        farmer=_get(farm,'farmer');hands=_get(farm,'hands',[])
        values.extend([float(_get(farm,'money')),farmer[0],farmer[1],len(hands)])
        for pos in hands:values.extend(pos)
        values.extend([sum(_QUADS[x] for x in _get(farm,'unlocked_quadrants',[])),int(_get(farm,'hires_today',0))])
        board=_get(farm,'tiles',[])
        if len(board)!=10 or any(len(row)!=10 for row in board):raise ValueError('Expected 10x10 board')
        for row in board:
            for tile in row:
                t={} if tile is None or isinstance(tile,str) else tile
                animal=_get(t,'animal');crop=_get(t,'crop')
                kind=6 if animal else _KINDS.get(tile if isinstance(tile,str) else _get(t,'kind','EMPTY'))
                if kind is None:raise ValueError('Unknown tile kind')
                values.extend([kind,_IDS[crop] if crop else -1,_IDS[animal] if animal else -1,
                    int(_get(t,'planted_day',0)),int(_get(t,'placed_day',0)),int(_get(t,'yield_units',0)),
                    int(_get(t,'consecutive_unwatered',0)),int(_get(t,'consecutive_unfed',0)),
                    int(_get(t,'fertilized_until_day',-1)),int(_get(t,'pending_care_bonus',0)),
                    int(_get(t,'max_lifespan_step',-1)),bool(_get(t,'watered_today',False)),
                    bool(_get(t,'fed_today',False)),bool(_get(t,'cared_today',False)),bool(_get(t,'fertilizer_available',False))])
    private=_get(obs,'private',{});shed=_get(private,'shed',{});seeds=_get(private,'seeds',{})
    values.extend(int(_get(shed,k,0)) for k in _ITEMS)
    values.extend(int(_get(seeds,k,0)) for k in _ITEMS[:5])
    inventories=_get(private,'inventories',[]);values.append(len(inventories))
    for bag in inventories:
        values.extend(int(_get(bag,k,0)) for k in _ITEMS)
        order=[_IDS[k] for k,v in bag.items() if int(v)!=0]
        values.append(len(order));values.extend(order)
    market=_get(obs,'market',{})
    for name in ('inventory','prices'):
        mapping=_get(market,name,{})
        values.extend(int(_get(mapping,k,0)) for k in _ITEMS[:9])
    shops=_get(_get(obs,'town',{}),'unlocked_shops',[])
    values.append(len(shops));values.extend(_SHOP_IDS[x] for x in shops)
    return (ctypes.c_double*len(values))(*values)


_LIB=ctypes.CDLL(str(Path(__file__).parent/'agent.so'))
_LIB.mp_error.restype=ctypes.c_char_p
_LIB.mp_act.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
_LIB.mp_config.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int]
_LIB.mp_config.restype=ctypes.c_int
_LIB.mp_debug.argtypes=[ctypes.c_int]
_LIB.mp_debug.restype=ctypes.c_char_p
_CONFIGURED=False
PARAMS=dict(competition=.75,supply=1.,replant=.8,capital_power=.5,discount=.01,work_scale=1.,land_rent=3.,action_cost=1.5,labor_hours=16.,future_shop=1.,reserve=70.,max_animals=18.,new_limit=30.,feed_cover=2.,hold=1.,expansion=0.,margin_weight=1.,risk=0.,finite_fertilize=0.,wheat_age=4.,melon_age=10.,mpc=0.,inventory_dp=0.,tape=0.,service_dp=0.,service_cost=3.,execution_variant=0.,admission_preview=0.,rotation_edits=0.,service_rounds=0.,growth_forecast=0.,harvest_threshold=0.,investment_blend=0.,investment_scope=1.,investment_start=1.,investment_margin=0.,investment_timing=1.,crop_dp=0.,crop_start=0.,crop_gain=0.,crop_action_cost=3.,crop_fert_floor=0.,crop_max_changes=100.,crop_timing=0.,crop_price_mode=0.,investment_consistent=0.,investment_paid_labor=0.,investment_crop_consistent=0.,calendar_samples=0.)
_PARAM_ORDER=tuple(PARAMS)
_config_path=Path(__file__).with_name('config.json')
if _config_path.exists():
    _overrides=json.loads(_config_path.read_text(encoding='utf-8'))
    if not isinstance(_overrides,dict) or set(_overrides)-set(PARAMS):
        raise ValueError('Unknown configuration keys')
    PARAMS.update(_overrides)

def debug_state(seat=0):
    return json.loads(_LIB.mp_debug(int(seat)).decode('utf-8'))

def agent(observation,configuration=None):
    global _CONFIGURED
    if not _CONFIGURED:
        x=(ctypes.c_double*len(_PARAM_ORDER))(*(float(PARAMS[k]) for k in _PARAM_ORDER))
        if _LIB.mp_config(x,len(x))<0:
            raise ValueError(_LIB.mp_error().decode('utf-8'))
        _CONFIGURED=True
    packed=_pack(observation);out=(ctypes.c_int32*16384)();size=_LIB.mp_act(packed,len(packed),out,len(out))
    if size<0:raise RuntimeError(_LIB.mp_error().decode())
    units,markets=out[0],out[1];at=2
    def atom():
        nonlocal at
        op,item,quantity=out[at:at+3];at+=3;name=_OPS[op];a=[name]
        if item>=0:
            a.append(_ITEMS[item])
            if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):a.append(quantity)
        elif quantity!=1:a.append(quantity)
        return a
    actions=[atom()for _ in range(units)];orders=[atom()for _ in range(markets)]
    return dict(farmer=actions[0]if actions else ['PASS'],hands=actions[1:],market=orders)
