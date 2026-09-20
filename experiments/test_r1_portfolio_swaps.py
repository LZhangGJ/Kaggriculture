#!/usr/bin/env python3
"""Default-binary parity and opt-in smoke for bounded portfolio swaps."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawAgent:
    def __init__(self, module, binary, swaps=0):
        config = module.DEFAULTS.copy()
        config.update(json.loads((ROOT / "policy/r1/config.json").read_text()))
        config["portfolio_swaps"] = swaps
        self.module = module
        self.lib = ctypes.CDLL(str(binary))
        self.lib.td_settings_count.restype = ctypes.c_size_t
        count = self.lib.td_settings_count()
        if count not in (len(module._ORDER), len(module._ORDER) - 1, len(module._ORDER) - 3):
            raise RuntimeError("unexpected settings ABI")
        values = [float(config[key]) for key in module._ORDER[:count]]
        self.lib.td_new.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        self.lib.td_new.restype = ctypes.c_void_p
        self.lib.td_observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
        self.lib.td_observe.restype = ctypes.c_int
        self.lib.td_delete.argtypes = [ctypes.c_void_p]
        packed = (ctypes.c_double * count)(*values)
        self.handle = self.lib.td_new(packed, count)
        if not self.handle:
            raise RuntimeError("failed to create raw R1")

    def __call__(self, observation):
        packed = self.module._pack(observation)
        out = (ctypes.c_int32 * 256)()
        size = self.lib.td_observe(self.handle, packed, len(packed), out, len(out))
        if size < 0:
            raise RuntimeError("native action failed")
        units, markets, at = out[0], out[1], 2
        def atom():
            nonlocal at
            op, item, quantity = out[at:at + 3]; at += 3
            action = [self.module._OPS[op]]
            if item >= 0:
                action.append(self.module._ITEMS[item])
                if action[0] in ("PLACE", "PICKUP", "BUY_SEED", "BUY_ANIMAL",
                                 "BUY_PRODUCT", "SELL"):
                    action.append(quantity)
            elif quantity != 1:
                action.append(quantity)
            return action
        actions = [atom() for _ in range(units)]
        orders = [atom() for _ in range(markets)]
        assert size == at
        return {"farmer": actions[0] if actions else ["PASS"],
                "hands": actions[1:], "market": orders}

    def close(self):
        if self.handle:
            self.lib.td_delete(self.handle); self.handle = None


def observation(state, player):
    value = json.loads(json.dumps(state[player].observation))
    value["step"] = value.get("day", 0) * 24 + value.get("hour", 0)
    value["player"] = player
    return value


def main():
    os.environ.pop("R1_CONFIG_OVERRIDES", None)
    module = load(ROOT / "policy/r1/agent.py", "portfolio_swap_r1")
    old = RawAgent(module, ROOT / "work/agent-before-portfolio-swaps.so")
    new = RawAgent(module, ROOT / "policy/r1/agent.so")
    rival = load(ROOT / "opponents/thomas_2945/main.py", "portfolio_swap_rival").agent
    env = make("kaggriculture", configuration={"seed": 2610200002}, debug=True)
    state = env.reset()
    frames = 0
    while not env.done and frames < 320:
        own, other = observation(state, 0), observation(state, 1)
        before, after = old(own), new(own)
        assert before == after, (frames, before, after)
        state = env.step([before, rival(other)])
        frames += 1
    old.close(); new.close()

    before_pair = RawAgent(module, ROOT / "work/agent-before-pair-swaps.so", swaps=1)
    after_pair = RawAgent(module, ROOT / "policy/r1/agent.so", swaps=1)
    rival = load(ROOT / "opponents/thomas_2945/main.py", "portfolio_pair_parity_rival").agent
    env = make("kaggriculture", configuration={"seed": 2610200002}, debug=True)
    state = env.reset()
    pair_frames = 0
    while not env.done and pair_frames < 320:
        own, other = observation(state, 0), observation(state, 1)
        before, after = before_pair(own), after_pair(own)
        assert before == after, (pair_frames, before, after)
        state = env.step([before, rival(other)])
        pair_frames += 1
    before_pair.close(); after_pair.close()

    os.environ["R1_CONFIG_OVERRIDES"] = '{"portfolio_swaps":1}'
    enabled = load(ROOT / "policy/r1/agent.py", "portfolio_swap_enabled").Agent()
    assert enabled.config["portfolio_swaps"] == 1
    enabled.close()
    os.environ["R1_CONFIG_OVERRIDES"] = '{"portfolio_swaps":1,"portfolio_swap_min_gain":20}'
    thresholded = load(ROOT / "policy/r1/agent.py", "portfolio_swap_thresholded").Agent()
    assert thresholded.config["portfolio_swap_min_gain"] == 20
    thresholded.close()
    os.environ["R1_CONFIG_OVERRIDES"] = '{"portfolio_swaps":2}'
    paired = load(ROOT / "policy/r1/agent.py", "portfolio_pair_enabled").Agent()
    rival = load(ROOT / "opponents/thomas_2945/main.py", "portfolio_pair_smoke_rival").agent
    env = make("kaggriculture", configuration={"seed": 2610400001}, debug=True)
    state = env.reset()
    for _ in range(72):
        state = env.step([paired(observation(state, 0)), rival(observation(state, 1))])
    assert paired.debug()["portfolio_pair_trials"] > 0
    paired.close()
    try:
        os.environ["R1_CONFIG_OVERRIDES"] = '{"portfolio_swaps":1.5}'
        load(ROOT / "policy/r1/agent.py", "portfolio_swap_invalid").Agent()
    except ValueError:
        pass
    else:
        raise AssertionError("fractional portfolio_swaps was accepted")
    finally:
        os.environ.pop("R1_CONFIG_OVERRIDES", None)
    print(json.dumps({"status": "PASS", "default_equal_frames": frames,
                      "single_swap_equal_frames": pair_frames,
                      "switch": "R1_CONFIG_OVERRIDES.portfolio_swaps"}))


if __name__ == "__main__":
    main()

