"""Port the cash-backed initial wheat intent onto an unchanged native policy.

No opponent identity, hidden state, or seed is used. Original orders and unit
actions are retained, with the same two-slot/cash/warehouse guards as Liquidity.
"""
from pathlib import Path
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_fusion_base', ROOT / 'base_entry.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
settings = json.loads((ROOT / 'overlay.json').read_text())


def wheat_price(inventory):
    delta = abs(inventory - 10000)
    value = 25 + math.sqrt(delta) if inventory < 10000 else 25 - 5 / math.log1p(400) * math.log1p(delta)
    return max(1, round(value))


class Agent:
    def __init__(self, binary_path=None):
        self.inner = base.create_agent(binary_path)
        self.config = self.inner.config

    def close(self):
        self.inner.close()

    def debug(self):
        return self.inner.debug()

    def __call__(self, obs, configuration=None):
        action = self.inner(obs, configuration)
        orders = action.get('market', [])
        if int(obs['step']) != 0 or len(orders) > 8:
            return action
        requested = min(10, max(0, int(settings['opening_liquidity'])))
        money = obs['farms'][int(obs['player'])]['money']
        reserve = max(20, float(self.config.get('reserve', 120)))
        inventory = obs['market']['inventory']['WHEAT']
        cost = quantity = 0
        for n in range(1, requested + 1):
            cost += wheat_price(inventory - n)
            if cost + reserve > money:
                break
            quantity = n
        quantity = min(quantity, max(0, 100 - sum(obs['private']['shed'].values())))
        if quantity:
            return dict(action, market=[['BUY_PRODUCT', 'WHEAT', quantity], ['SELL', 'WHEAT', quantity]] + orders)
        return action


def create_agent(binary_path=None):
    return Agent(binary_path)


seats = {}


def agent(observation, configuration=None):
    seat = int(observation['player'])
    if seat not in seats:
        seats[seat] = create_agent()
    return seats[seat](observation, configuration)
