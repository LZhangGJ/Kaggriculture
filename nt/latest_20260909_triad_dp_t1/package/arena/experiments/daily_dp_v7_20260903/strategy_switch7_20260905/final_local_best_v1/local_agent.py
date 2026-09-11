"""Frozen local winner for the S9 selection scope: J7_03, no switch tree.

This is a workspace-local observation entry, not a Kaggle submission package.
Create one Agent per process and call reset() before a new fixture.
"""
from pathlib import Path
import ctypes,importlib.util,json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('s9_final_obs_pack',ROOT/'submission/pending_dp27_fixed_macro_dynamic_20260904/main.py')
packing=importlib.util.module_from_spec(spec);spec.loader.exec_module(packing)

class Agent:
    def __init__(self):
        self.policy=json.loads((HERE/'compact_policy.json').read_text(encoding='utf-8'))
        self.lib=ctypes.CDLL(str(HERE/'agent.so'))
        self.lib.s8_error.restype=ctypes.c_char_p
        self.lib.s8_act.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t]
        self.lib.s8_target.argtypes=[ctypes.c_int];self.lib.s8_switch_day.argtypes=[ctypes.c_int]
        self.reset()

    def reset(self):
        self.lib.s8_reset()

    def agent(self,observation,configuration=None):
        packed=packing._pack(observation);empty=(ctypes.c_float*0)();out=(ctypes.c_int32*16384)()
        size=self.lib.s8_act(packed,len(packed),empty,0,out,len(out))
        if size<0:raise RuntimeError(self.lib.s8_error().decode())
        units,markets=out[0],out[1];at=2
        def atom():
            nonlocal at
            op,item,quantity=out[at:at+3];at+=3;name=packing._OPS[op];action=[name]
            if item>=0:
                action.append(packing._ITEMS[item])
                if name in ('PLACE','PICKUP','BUY_SEED','BUY_ANIMAL','BUY_PRODUCT','SELL'):action.append(quantity)
            elif quantity!=1:action.append(quantity)
            return action
        actions=[atom() for _ in range(units)];orders=[atom() for _ in range(markets)]
        assert at==size
        return dict(farmer=actions[0] if actions else ['PASS'],hands=actions[1:],market=orders)

    def __call__(self,observation,configuration=None):
        return self.agent(observation,configuration)

    def debug(self,seat):
        compact=self.lib.s8_target(int(seat));source=self.policy['source_indices'][compact] if compact>=0 else -1
        return dict(target=source,switch_day=self.lib.s8_switch_day(int(seat)))
