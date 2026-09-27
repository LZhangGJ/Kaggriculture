"""long-market-v1 (audit): long-horizon market / rival-activity summaries, attached zero-init.

Per market item i (12 tokens; the 3 animal tokens stay zero), 12 features, from public data plus own requests only:
  0:3   price level vs EMA (half-life-ish windows 24/72/240 turns): scaled(p - ema_k(p)) * 3
  3:6   net public market-inventory flow over the window: scaled(sum_k(dS)) * 3,  dS = inv_t - inv_{t-1}
  6:9   residual flow (rival + town): dS - own_sold + own_bought, own_sold = min(requested SELL, own shed at request time)
  9:12  own requested sells over the window: scaled(sum_k(own_sold)) * 3
  12    town demand per day for the item (6 x unlocked-shop demand + town centre), /20
  13    days of market stock at that demand: scaled(inventory / max(demand, 1)) * 3
  14:16 demand of the first and second unlocked shop for the item (unlock order; /2)
where sum_k(x) is an EMA with alpha = 1/k times k. Identical recursions in float64 on GPU (GPUHistory) and CPU
(LongMarketHistory, official observations). The model adds market_long(x['market_long']) to its 12 market tokens before
the Transformer; the last layer is zero-initialised, so outputs are unchanged at init.
"""
import types
import numpy as np
import torch
from torch import nn

WINDOWS = (24., 72., 240.)
F = 16
SHOP_NAMES = ('BAKERY', 'BRUNCH_SPOT', 'FARMERS_MARKET', 'ICE_CREAM_SHOP', 'PET_CAFE', 'PIZZA_SHOP', 'SMOOTHIE_SHOP', 'YARN_STORE')
SHOP_DEMAND = ((1, 0, 0, 0, 0, 1, 0, 0, 0), (1, 0, 0, 1, 0, 1, 0, 0, 0), (1, 1, 1, 1, 0, 0, 0, 0, 0), (1, 0, 0, 1, 0, 0, 1, 0, 0),
               (0, 2, 0, 0, 0, 0, 0, 0, 0), (1, 0, 1, 0, 0, 0, 1, 0, 0), (0, 0, 0, 1, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 0, 0, 2, 0))
CENTER = (1, 1, 1, 1, 1, 1, 1, 1, 0)
KEYS = ('market_long.0.weight', 'market_long.0.bias', 'market_long.2.weight', 'market_long.2.bias')
PRODUCTS = 9


def scaled3(x):
    return x.sign() * x.abs().log1p() / 4


def consts(device):
    return dict(table=torch.tensor(SHOP_DEMAND, dtype=torch.float64, device=device),
                center=torch.tensor(CENTER, dtype=torch.float64, device=device),
                w=torch.tensor(WINDOWS, dtype=torch.float64, device=device),
                alpha=torch.tensor([1 / w for w in WINDOWS], dtype=torch.float64, device=device))


def features_from(pema, fema, price, stock, shops, c):
    """pema [n,9,3], fema [n,9,3(dS,resid,own),3], price/stock [n,9] (float64), shops [n,8] long (-1 = none) -> [n,12,F]."""
    n = price.shape[0]
    safe = shops.long().clamp_min(0); on = (shops >= 0).double()
    per = c['table'][safe] * on[:, :, None]  # [n,8,9]
    demand = 6 * per.sum(1) + c['center']
    out = torch.zeros(n, 12, F, dtype=torch.float64, device=price.device)
    w = c['w']
    out[:, :PRODUCTS, 0:3] = scaled3(price[:, :, None] - pema)
    out[:, :PRODUCTS, 3:6] = scaled3(fema[:, :, 0] * w)
    out[:, :PRODUCTS, 6:9] = scaled3(fema[:, :, 1] * w)
    out[:, :PRODUCTS, 9:12] = scaled3(fema[:, :, 2] * w)
    out[:, :PRODUCTS, 12] = demand / 20
    out[:, :PRODUCTS, 13] = scaled3(stock / demand.clamp_min(1))
    out[:, :PRODUCTS, 14] = per[:, 0] / 2
    out[:, :PRODUCTS, 15] = per[:, 1] / 2
    return out.float()


def step(state, price, stock, shed_prev, req_sell, req_buy, first, shops, c):
    """One observed turn. state: dict(pema, fema, stock) float64. first: bool [n]. c: consts(device). Returns features."""
    alpha = c['alpha']
    price = price.double(); stock = stock.double()
    ds = stock - state['stock']
    own_sold = torch.minimum(req_sell.double(), shed_prev.double().clamp_min(0))
    own_bought = req_buy.double()
    resid = ds - own_sold + own_bought
    f = first[:, None, None]
    pema = torch.where(f, price[:, :, None].expand(-1, -1, 3), state['pema'] + alpha * (price[:, :, None] - state['pema']))
    x = torch.stack((ds, resid, own_sold), 2)[..., None]  # [n,9,3,1]
    fema = torch.where(first[:, None, None, None], torch.zeros_like(state['fema']), state['fema'] + alpha * (x - state['fema']))
    state['pema'].copy_(pema); state['fema'].copy_(fema); state['stock'].copy_(stock)
    return features_from(pema, fema, price, stock, shops, c)


def new_state(n, device):
    return dict(pema=torch.zeros(n, PRODUCTS, 3, dtype=torch.float64, device=device),
                fema=torch.zeros(n, PRODUCTS, 3, 3, dtype=torch.float64, device=device),
                stock=torch.zeros(n, PRODUCTS, dtype=torch.float64, device=device))


class LongMarketHistory:
    """CPU/official-observation twin of the GPU recursion (one seat). Call observe(obs) each turn before encoding and
    requests(list of official orders) after acting."""
    ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

    def __init__(self):
        self.state = new_state(1, 'cpu'); self.count = 0
        self.req_sell = torch.zeros(1, PRODUCTS, dtype=torch.float64); self.req_buy = torch.zeros_like(self.req_sell)
        self.shed_prev = torch.zeros_like(self.req_sell); self.features = torch.zeros(1, 12, F); self.c = consts('cpu')

    def observe(self, obs):
        price = torch.tensor([[obs['market']['prices'].get(i, 0) for i in self.ITEMS]], dtype=torch.float64)
        stock = torch.tensor([[obs['market']['inventory'].get(i, 0) for i in self.ITEMS]], dtype=torch.float64)
        names = list(obs['town']['unlocked_shops'])[:8]
        shops = torch.full((1, 8), -1, dtype=torch.long)
        for j, name in enumerate(names):
            shops[0, j] = SHOP_NAMES.index(name)
        self.features = step(self.state, price, stock, self.shed_prev, self.req_sell, self.req_buy, torch.tensor([self.count == 0]), shops, self.c)
        self.shed_prev = torch.tensor([[obs['private']['shed'].get(i, 0) for i in self.ITEMS]], dtype=torch.float64)
        self.req_sell.zero_(); self.req_buy.zero_(); self.count += 1
        return self.features[0]

    def requests(self, orders):
        for o in orders or []:
            if isinstance(o, list) and len(o) > 2 and o[1] in self.ITEMS and type(o[2]) is int:
                i = self.ITEMS.index(o[1])
                if o[0] == 'SELL':
                    self.req_sell[0, i] += max(0, o[2])
                elif o[0] == 'BUY_PRODUCT':
                    self.req_buy[0, i] += max(0, o[2])


def _encode_features(self, x):
    b = x['boards'].shape[0]
    cells = self.board(x['boards'].reshape(b * 2, 36, 10, 10)).flatten(2).transpose(1, 2)
    cells = cells + self.position
    summaries = self.pool(self.spatial_queries.expand(b * 2, -1, -1), cells, cells, need_weights=False)[0].reshape(b, 2, 4, 192)
    summaries = (summaries + self.farm_role.weight[None, :, None]).reshape(b, 8, 192)
    market = self.market_encoder(x['market'])
    if 'market_long' in x:
        market = market + self.market_long(x['market_long'])
    groups = [self.global_encoder(x['global_features'])[:, None], self.farm_encoder(x['farms']),
              summaries, self.worker_encoder(x['workers']), market,
              self.program_encoder(x['programs']), self.history_encoder(x['history'])]
    groups = [v + self.types.weight[i] for i, v in enumerate(groups)]
    groups[1] = groups[1] + self.farm_role.weight[None]
    tokens = torch.cat(groups, 1)
    padding = torch.cat([torch.zeros(b, 11, device=tokens.device, dtype=torch.bool),
                         ~x['worker_valid'], torch.zeros(b, 12, device=tokens.device, dtype=torch.bool),
                         ~x['program_valid'], ~x['history_valid']], 1)
    for layer in self.layers:
        tokens = layer(tokens, src_key_padding_mask=padding)
    w = x['workers'].shape[1]
    context = dict(cells=cells.reshape(b, 200, 192), workers=tokens[:, 11:11 + w],
                   market=tokens[:, 11 + w:23 + w], programs=tokens[:, 23 + w:39 + w])
    return tokens[:, 0], context


def attach_long_market(model):
    if hasattr(model, 'market_long'):
        raise ValueError('long-market already attached')
    model.market_long = nn.Sequential(nn.Linear(F, 64), nn.SiLU(), nn.Linear(64, 192))
    nn.init.zeros_(model.market_long[2].weight); nn.init.zeros_(model.market_long[2].bias)
    model.encode_features = types.MethodType(_encode_features, model)
    return model
