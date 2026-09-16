"""CPU policy with a frozen sequence of observation-only plan selectors."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


codec = module('_robust90_codec', 'codec.py')
features = module('_robust90_features', 'features.py')
selector = module('_robust90_selector', 'selector.py')


class Agent:
    def __init__(self):
        self.base = json.loads((ROOT / 'config.json').read_text())
        self.config = self.base.copy()
        manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
        for name, digest in manifest['files'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Package hash mismatch: ' + name)
        self.models = [json.loads((ROOT / name).read_text()) for name in manifest['selectors']]
        self.restore_days = manifest.get('restore_days', [])
        self.inner = self.new_inner()
        self.last = -1
        self.selected = []

    def new_inner(self):
        inner = codec.Agent(config=self.base, binary_path=ROOT / 'agent.so')
        inner.lib.td_reconfigure.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
        inner.lib.td_reconfigure.restype = ctypes.c_int
        return inner

    def __call__(self, observation, configuration=None):
        step = int(observation['step'])
        if step == 0 and self.last >= 0:
            self.inner.close()
            self.inner = self.new_inner()
            self.config = self.base.copy()
            self.selected = []
            self.last = -1
        if step in [day * 24 for day in self.restore_days]:
            self.config = self.base.copy()
            self.inner.config.update(self.config)
            params = (ctypes.c_double * len(codec._ORDER))(*(self.config[k] for k in codec._ORDER))
            if self.inner.lib.td_reconfigure(self.inner.handle, params, len(params)) != 0:
                raise RuntimeError('Base restoration failed')
        for model in self.models:
            if step == model['day'] * 24:
                values = features.policy_features(observation, self.config)
                if any(k.startswith('history_') for k in model['features']):
                    values.update(features.history_features(self.inner))
                if any(k.startswith('plan_') for k in model['features']):
                    values.update(features.plan_features(self.inner, observation, codec, model['profiles'], self.base))
                if model.get('economic_features'):
                    values.update(features.economic_plan_features(values, model['profiles']))
                name, changes, scores = selector.choose(model, values)
                self.config = dict(self.base, **changes)
                self.inner.config.update(self.config)
                params = (ctypes.c_double * len(codec._ORDER))(*(self.config[k] for k in codec._ORDER))
                if self.inner.lib.td_reconfigure(self.inner.handle, params, len(params)) != 0:
                    raise RuntimeError('Plan reconfiguration failed')
                self.selected.append(dict(day=model['day'], name=name, scores=scores))
        action = self.inner(observation, configuration)
        self.last = step
        return action

    def debug(self):
        return dict(native=self.inner.debug(), selections=self.selected)

    def close(self):
        self.inner.close()


def create_agent():
    return Agent()


_instances = {}


def agent(observation, configuration=None):
    seat = int(observation['player'])
    if seat not in (0, 1):
        raise ValueError('Invalid player')
    if seat not in _instances:
        _instances[seat] = Agent()
    return _instances[seat](observation, configuration)


def reset():
    for instance in _instances.values():
        instance.close()
    _instances.clear()
