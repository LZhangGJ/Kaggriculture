"""Frozen DP27 daily macro interventions with the original dynamic C++ executor."""
import ctypes
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

def _load():
    global _LIB
    if _LIB is None:
        path=Path(_load.__code__.co_filename).resolve().parent/'agent.so'
        lib=ctypes.CDLL(str(path))
        lib.dp27_act.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
        lib.dp27_act.restype=ctypes.c_int
        lib.dp27_error.restype=ctypes.c_char_p
        lib.dp27_reset.argtypes=[];lib.dp27_reset.restype=None
        lib.dp27_abi.restype=ctypes.c_int
        if lib.dp27_abi()!=1:raise RuntimeError('DP27 ABI mismatch')
        _LIB=lib
    return _LIB

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

def agent(observation,configuration=None):
    packed=_pack(observation);output=(ctypes.c_int32*16384)()
    lib=_load();status=lib.dp27_act(packed,len(packed),output,len(output))
    if status<0:raise RuntimeError(lib.dp27_error().decode('utf8','replace'))
    units,markets=output[0],output[1];at=2
    def atom():
        nonlocal at
        op,item,quantity=output[at:at+3];at+=3;name=_OPS[op];result=[name]
        if item>=0:
            result.append(_ITEMS[item])
            if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):result.append(quantity)
        elif quantity!=1:result.append(quantity)
        return result
    actions=[atom() for _ in range(units)];orders=[atom() for _ in range(markets)]
    if at!=status:raise RuntimeError('DP27 output length mismatch')
    return {'farmer':actions[0] if actions else ['PASS'],'hands':actions[1:],'market':orders}
