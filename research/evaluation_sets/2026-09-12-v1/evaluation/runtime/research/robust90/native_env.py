"""Offline C++ evaluator host. Must pass full-state parity before use."""
import ctypes
import json
from pathlib import Path

ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER','GOOSE','COW','SHEEP')
OPS=('PASS','NORTH','SOUTH','EAST','WEST','DROP','PICKUP','PLACE','PLANT','WATER','HARVEST','FERTILIZE','DIG','BUILD_COOP','BUILD_PASTURE','FEED','COLLECT_FERTILIZER','CARE','HIRE','BUY_LAND','BUY_SEED','BUY_PRODUCT','BUY_ANIMAL','SELL')


class NativeGame:
    def __init__(self,seed,engine=None,library='native_host.so'):
        from cpu_runtime import AttrDict,load_engine
        self.engine=engine or load_engine()
        self.configuration=AttrDict({k:v.get('default') if isinstance(v,dict) else v for k,v in self.engine.specification['configuration'].items()})
        self.configuration.update(seed=None,runTimeout=1200)
        self.lib=ctypes.CDLL(str(Path(__file__).with_name(library)))
        self.lib.rh_new.argtypes=[ctypes.c_uint64];self.lib.rh_new.restype=ctypes.c_void_p
        self.lib.rh_delete.argtypes=[ctypes.c_void_p];self.lib.rh_delete.restype=None
        self.lib.rh_observation.argtypes=[ctypes.c_void_p,ctypes.c_int];self.lib.rh_observation.restype=ctypes.c_char_p
        self.lib.rh_step.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t];self.lib.rh_step.restype=ctypes.c_int
        for name in ('rh_done','rh_step_count'):
            getattr(self.lib,name).argtypes=[ctypes.c_void_p];getattr(self.lib,name).restype=ctypes.c_int
        self.handle=self.lib.rh_new(seed)
        if not self.handle:raise RuntimeError('Native simulator allocation failed')
        self.t=0;self.done=False

    def observation(self,seat):
        from cpu_runtime import AttrDict
        raw=self.lib.rh_observation(self.handle,seat)
        if raw is None:raise RuntimeError('Native observation failed')
        return AttrDict(json.loads(raw))

    def advance(self,actions):
        if self.done:raise RuntimeError('Game already terminal')
        values=[]
        for action in actions:
            units=[action.get('farmer',['PASS'])]+action.get('hands',[])
            market=action.get('market',[]);values.extend([len(units),len(market)])
            for atom in units+market:
                if not isinstance(atom,(tuple,list)) or not atom:atom=['PASS']
                op=OPS.index(atom[0]);item=-1;quantity=1
                if len(atom)>1:
                    if isinstance(atom[1],str):item=ITEMS.index(atom[1])
                    else:quantity=int(atom[1])
                if len(atom)>2:quantity=int(atom[2])
                values.extend([op,item,quantity])
        packed=(ctypes.c_int32*len(values))(*values)
        result=self.lib.rh_step(self.handle,packed,len(values))
        if result:raise RuntimeError(f'Native step error {result}')
        self.t=self.lib.rh_step_count(self.handle);self.done=bool(self.lib.rh_done(self.handle))

    def close(self):
        if getattr(self,'handle',None):self.lib.rh_delete(self.handle);self.handle=None

    def __del__(self):self.close()


class ParityGame:
    def __init__(self,seed,engine=None,native_class=NativeGame):
        from cpu_runtime import LocalGame
        self.native=native_class(seed,engine);self.official=LocalGame(seed,engine)
        self.configuration=self.official.configuration;self.t=0;self.done=False
        self.observations_checked=0

    def observation(self,seat):
        a=self.official.observation(seat);b=self.native.observation(seat)
        mismatch=difference(a,b)
        if mismatch:raise AssertionError(f'Native parity step {self.t} seat {seat}: {mismatch}')
        self.observations_checked+=1
        return a

    def advance(self,actions):
        self.official.advance(actions);self.native.advance(actions)
        if self.official.t!=self.native.t or self.official.done!=self.native.done:raise AssertionError('Native clock/status mismatch')
        self.t=self.official.t;self.done=self.official.done

    def close(self):self.native.close()


class DirectGame(NativeGame):
    def __init__(self,seed,engine=None,library='native_direct_host.so'):
        super().__init__(seed,engine,library)
        self.lib.rh_policy_act.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
        self.lib.rh_policy_act.restype=ctypes.c_int

    def agent_act(self,agent,seat):
        if agent.last>=self.t or agent.seat is not None and agent.seat!=seat:raise ValueError('Policy clock/seat mismatch')
        out=(ctypes.c_int32*256)()
        count=self.lib.rh_policy_act(self.handle,seat,agent.handle,ctypes.cast(agent.lib.td_act,ctypes.c_void_p),out,len(out))
        if count<0:raise RuntimeError(f'Direct policy error {count}: {agent.debug()}')
        units,markets=out[:2]
        if count!=2+3*(units+markets):raise ValueError('Malformed direct policy action')
        commands=[]
        for at in range(2,count,3):
            op,item,quantity=out[at:at+3];name=OPS[op];atom=[name]
            if item>=0:
                atom.append(ITEMS[item])
                if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):atom.append(quantity)
            elif quantity!=1:atom.append(quantity)
            commands.append(atom)
        agent.last=self.t;agent.seat=seat
        return dict(farmer=commands[0] if units else ['PASS'],hands=commands[1:units],market=commands[units:])


class DirectParityGame(ParityGame):
    def __init__(self,seed,engine=None):super().__init__(seed,engine,DirectGame)
    def agent_act(self,agent,seat):return self.native.agent_act(agent,seat)


class FutureDirectGame(DirectGame):
    """Offline continuations with independently sampled future weather/towns."""
    def __init__(self,seed,engine=None):
        super().__init__(seed,engine,'native_future_host.so')

    def reseed_future(self,seed):
        if not 0<=seed<=0xffffffff:raise ValueError('Future seed range')
        before=[self.observation(s) for s in (0,1)]
        fn=self.lib.rh_reseed_future;fn.argtypes=[ctypes.c_void_p,ctypes.c_uint64];fn.restype=ctypes.c_int
        if fn(self.handle,seed):raise RuntimeError('Future reseed failed')
        assert before==[self.observation(s) for s in (0,1)]


class FutureDirectParityGame(ParityGame):
    def __init__(self,seed,engine=None):super().__init__(seed,engine,FutureDirectGame)
    def agent_act(self,agent,seat):return self.native.agent_act(agent,seat)
    def reseed_future(self,seed):
        before=[self.observation(s) for s in (0,1)]
        self.native.reseed_future(seed);self.official.info['seed']=seed
        assert self.official.configuration.seed is None
        assert before==[self.observation(s) for s in (0,1)]


def difference(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        if set(a)!=set(b):return path+': keys '+str(set(a)^set(b))
        if path.startswith('/private/inventories/') and path.count('/')==3 and list(a)!=list(b):
            return path+': inventory insertion order'
        for k in a:
            d=difference(a[k],b[k],path+'/'+str(k))
            if d:return d
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return path+': length'
        for i,(x,y) in enumerate(zip(a,b)):
            d=difference(x,y,path+'/'+str(i))
            if d:return d
    elif a!=b:return path+': '+repr(a)+' != '+repr(b)
    return None

