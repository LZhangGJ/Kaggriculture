"""CPU host for the unmodified official Kaggriculture 1.32.7 interpreter.

This is a local analysis host, NOT Kaggle's sandbox, timeout or schema validator.
Only the official seed utility import is redirected to its frozen source copy.
"""
from __future__ import annotations
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parent


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_agent(path):
    path = Path(path).resolve()
    name = "local_agent_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.agent


def load_engine():
    utils_spec = importlib.util.spec_from_file_location("frozen_seed_utils", ROOT / "official/seed_utils.py")
    utils = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(utils)
    original_import = builtins.__import__

    def frozen_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kaggle_environments.utils" and "resolve_episode_seed" in fromlist:
            return utils
        return original_import(name, globals, locals, fromlist, level)

    path = ROOT / "official/kaggriculture.py"
    module = types.ModuleType("frozen_kaggriculture_1327")
    module.__file__ = str(path)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=frozen_import)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class LocalGame:
    def __init__(self, seed, engine=None):
        self.engine = engine or load_engine()
        config = {}
        for key, value in self.engine.specification["configuration"].items():
            config[key] = copy.deepcopy(value.get("default") if isinstance(value, dict) else value)
        config.update(seed=int(seed), runTimeout=1200)
        self.configuration = AttrDict(config)
        self.info = {}
        self.done = False
        self.t = 0
        self.state = [AttrDict(observation=AttrDict(step=0, remainingOverageTime=60, farms=[]),
                               action={}, reward=0, status="ACTIVE") for _ in range(2)]
        self.engine.interpreter(self.state, self)
        assert self.configuration.seed is None  # Never expose the random script to either agent.

    def observation(self, seat):
        obs = copy.deepcopy(self.state[seat].observation)
        obs.step = self.t
        assert "seed" not in obs and "private" in obs
        return obs

    def advance(self, actions):
        if self.done:
            raise RuntimeError("Episode already finished")
        for seat, action in enumerate(actions):
            if not isinstance(action, dict):
                raise TypeError(f"Seat {seat} must return a dict action")
            self.state[seat].action = copy.deepcopy(action)
            self.state[seat].observation.step = self.t
        self.engine.interpreter(self.state, self)
        self.t += 1
        for state in self.state:
            state.observation.step = self.t
        self.done = all(s.status == "DONE" for s in self.state)

    def snapshot(self):
        return copy.deepcopy(self.state)


def pass_agent(obs, configuration=None):
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in obs["farms"][obs["player"]]["hands"]], "market": []}


def compare_frame(game, expected):
    for seat in (0, 1):
        actual = game.state[seat]
        observed = expected[seat]["observation"]
        for key in ("farms", "private", "market", "town", "day", "hour", "player"):
            assert actual.observation[key] == observed[key], (game.t, seat, key)
        assert actual.reward == expected[seat]["reward"], (game.t, seat, "reward")
        assert actual.status == expected[seat]["status"], (game.t, seat, "status")

