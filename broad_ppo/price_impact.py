"""price-impact-v1 (audit): the official per-unit market price function, as an exact integer LUT shared by the GPU
collector, replay and the CPU/submission path.

Port of kaggle_environments 1.32.7 kaggriculture.market_price / _shape / MARKET_PARAMS (default configuration), same
Python float arithmetic, evaluated once per (product, inventory) on the host; every path indexes the same int64 table, so
the paths agree bit-for-bit by construction. Execution semantics mirrored from _process_market/_commit_unit:
  SELL unit:        price(inv), then inv += 1 only if price > 1
  BUY_PRODUCT unit: price(inv - 1), then inv -= 1   (WHEAT, FERTILIZER only)
Trade value of q units from inventory I (own order in isolation, pre-turn inventory; the rival's interleaved units are not
known in advance):  sell  sum_{j<q} price(I + j)  (floor price 1 is absorbing),  buy  sum_{j=1..q} price(I - j).
"""
import math
from functools import lru_cache
import numpy as np
import torch

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
MARKET_I0 = 10000
PRICE_FLOOR = 1
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
HINGE_GAIN = 8.0
LUT_MIN, LUT_MAX = -32768, 65535  # gpu_sim MARKET_MIN/MAX_INVENTORY
LUT_SIZE = LUT_MAX - LUT_MIN + 1


def _shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    p = (params or MARKET_PARAMS)[item]
    base = p["base"]; I0 = p["I0"]; T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


@lru_cache(None)
def lut_numpy():
    """[9, LUT_SIZE] int64 prices and [9, LUT_SIZE+1] int64 prefix sums (C[k] = sum_{i<k} price(LUT_MIN + i))."""
    lut = np.array([[market_price(item, inv) for inv in range(LUT_MIN, LUT_MAX + 1)] for item in PRODUCTS], np.int64)
    prefix = np.concatenate([np.zeros((9, 1), np.int64), np.cumsum(lut, 1)], 1)
    return lut, prefix


class PriceImpact:
    def __init__(self, device):
        lut, prefix = lut_numpy()
        self.lut = torch.as_tensor(lut, device=device)
        self.prefix = torch.as_tensor(prefix, device=device)

    def price(self, item, inv):
        return self.lut[item, (inv - LUT_MIN).clamp(0, LUT_SIZE - 1)]

    def _sum(self, item, a, b):
        """sum of price(i) for i in [a, b) with a <= b (int64 tensors, any range; outside the LUT the edge price holds)."""
        lo = (a - LUT_MIN).clamp(0, LUT_SIZE); hi = (b - LUT_MIN).clamp(0, LUT_SIZE)
        inside = self.prefix[item, hi] - self.prefix[item, lo]
        below = (torch.minimum(b, torch.full_like(b, LUT_MIN)) - a).clamp_min(0) * self.lut[item, torch.zeros_like(item)]
        above = (b - torch.maximum(a, torch.full_like(a, LUT_MAX + 1))).clamp_min(0) * self.lut[item, torch.full_like(item, LUT_SIZE - 1)]
        return inside + below + above

    def trade(self, item, inv, q, sell):
        """item [n] (0..8), inv [n], q [n, V] >= 0 effective units, sell [n] bool -> (value, post_price) int64 [n, V].
        Sell: units at price(I..I+q-1); inventory stops rising at the floor (price 1), which the LUT already makes absorbing
        for value; post price = price(I+q). Buy: price(I-1..I-q), post price = price(I-q)."""
        item = item.long()[:, None].expand_as(q); inv = inv.long()[:, None].expand_as(q); q = q.long()
        s = sell[:, None].expand_as(q)
        value = torch.where(s, self._sum(item, inv, inv + q), self._sum(item, inv - q, inv))
        post = torch.where(s, self.price(item, inv + q), self.price(item, inv - q))
        return value, post
