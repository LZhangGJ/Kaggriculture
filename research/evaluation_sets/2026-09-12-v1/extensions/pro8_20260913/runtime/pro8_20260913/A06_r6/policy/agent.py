"""Triad DP observation-only agent. Rebuild agent.so locally before use.
Each Agent owns one game context; module agent() isolates both seats.
Errors are explicit, not silently converted to replay or PASS fallbacks.
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

DEFAULTS = {'competition': 0.8, 'supply': 0.85, 'future_shop': 0.7, 'capital_power': 0.4, 'labor_hours': 15, 'work_price': 1.2, 'animal_work': 1.0, 'reserve': 120, 'max_animals': 20, 'max_hands': 14, 'max_land': 4, 'feed_cover': 2, 'rotation': 1, 'preview': 1, 'delivery': 2, 'intraday': 1, 'service': 1, 'replant': 0.7, 'land_rent': 2, 'discount': 0.015, 'tour_dp': 0, 'layout': 0, 'repeat': 1, 'animal_bias': 1, 'crop_bias': 1, 'portfolio_passes': 1, 'crop_fert': 1, 'harvest_threshold': 1, 'delay_sale': 0, 'opening_budget': 1, 'scenario': 0, 'keep_commitments': 1}
DEFAULTS.update(candidate_extra=0,service_reconcile=0,live_ledger=0,delivery_calendar=0,feed_finance=0,batch_delivery=0)
DEFAULTS.update(a06_reinvest=0,a06_roll_scope=0,a06_feed_contract=0,a06_state_contract=0,a06_execution_candidates=0)
DEFAULTS.update(a06_competitive_contract=0)
_ORDER = tuple(DEFAULTS)

class Agent:
    def __init__(self,config=None,binary_path=None):
        self.config=DEFAULTS.copy()
        path=Path(__file__).with_name('config.json')
        updates=json.loads(path.read_text()) if config is None and path.exists() else (config or {})
        if not isinstance(updates,dict) or set(updates)-set(self.config):
            raise ValueError('Unknown settings')
        self.config.update(updates)
        self.lib=ctypes.CDLL(str(Path(binary_path).resolve() if binary_path else Path(__file__).with_name('agent.so')))
        self.lib.td_new.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t];self.lib.td_new.restype=ctypes.c_void_p
        self.lib.td_delete.argtypes=[ctypes.c_void_p];self.lib.td_delete.restype=None
        self.lib.td_debug.argtypes=[ctypes.c_void_p];self.lib.td_debug.restype=ctypes.c_char_p
        self.lib.td_observe.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t];self.lib.td_observe.restype=ctypes.c_int
        self.lib.td_settings_count.argtypes=[];self.lib.td_settings_count.restype=ctypes.c_size_t
        if self.lib.td_settings_count()!=len(_ORDER):
            raise RuntimeError('Native/Python settings count mismatch; use matching source and binary')
        self.handle=None;self.last=-1;self.seat=None
        self.reset()
    def reset(self):
        self.close()
        params=(ctypes.c_double*len(_ORDER))(*(float(self.config[k]) for k in _ORDER))
        self.handle=self.lib.td_new(params,len(params))
        if not self.handle:raise ValueError('C++ settings validation failed')
        self.last=-1;self.seat=None
    def close(self):
        if getattr(self,'handle',None):
            self.lib.td_delete(self.handle);self.handle=None
    def __del__(self):
        self.close()
    def debug(self):
        if not self.handle:return {}
        text=self.lib.td_debug(self.handle).decode('utf-8')
        return json.loads(text) if not text.startswith('ERROR') else {'error':text}
    def __call__(self,observation,configuration=None):
        step=int(_get(observation,'step',0));seat=int(_get(observation,'player',0))
        if step==0 and self.last>=0:self.reset()
        if self.seat is not None and seat!=self.seat:raise ValueError('Do not reuse one Agent for both seats')
        if step<=self.last:raise ValueError('Observations must advance monotonically')
        packed=_pack(observation);out=(ctypes.c_int32*256)()
        size=self.lib.td_observe(self.handle,packed,len(packed),out,len(out))
        if size<0:raise RuntimeError(self.lib.td_debug(self.handle).decode())
        units,markets=out[0],out[1];at=2
        if size!=2+3*(units+markets):raise RuntimeError('Malformed native action')
        def atom():
            nonlocal at
            op,item,quantity=out[at:at+3];at+=3;name=_OPS[op];a=[name]
            if item>=0:
                a.append(_ITEMS[item])
                if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):a.append(quantity)
            elif quantity!=1:a.append(quantity)
            return a
        actions=[atom() for _ in range(units)];orders=[atom() for _ in range(markets)]
        self.last=step;self.seat=seat
        return dict(farmer=actions[0] if actions else ['PASS'],hands=actions[1:],market=orders)

_instances={}
def agent(observation,configuration=None):
    seat=int(_get(observation,'player',0))
    if seat not in _instances:_instances[seat]=Agent()
    return _instances[seat](observation,configuration)
