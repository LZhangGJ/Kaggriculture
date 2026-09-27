"""networth-shaping-v1: mark-to-market worth of each seat from the batched GPU simulator state.

The state is the dict recorded per turn by ppo.gpu_collect (data['state/<field>'], turn-major), or one
env.tensors() dict. Leading dims are arbitrary; the last two dims of the result are (game, seat).

Version A ('networth'):
    worth = money + sum_{p in 9 products} price[p] * (shed[p] + carried[p] + ready[p])
      shed[p]    = state.shed[..., p]                        (items 0..8; animals 9..11 are excluded)
      carried[p] = sum over ACTIVE units of unit_inventory[..., p]
      ready[p]   = sum of tile_yield on PLANT tiles growing crop p (p < 5, mature or not)
                 + sum of tile_yield on animal tiles whose animal produces p (GOOSE->EGG, COW->MILK, SHEEP->WOOL)
      price      = state.market_price (current market price of each product)
    This is exactly the worth of loss-diagnosis-20260923/aggregate.py (money + shed + carried + ready yields at the
    current price, from replay_analyze.py farm_summary). Integer valued; computed in int64, returned as float32.

Version B ('networth-growing'), heuristic, off unless selected:
    A + kappa * price * (expected future units that do not exist yet), with kappa = GROWING_KAPPA:
      - animals on tiles: one unit per remaining due daily refresh (refreshes after days today..28; the day-29 refresh
        never runs because the episode ends at step 719), assuming it is fed every day;
      - ongoing crops (TOMATO, STRAWBERRY): one unit per remaining due refresh with production_count <= max yield;
      - one-shot crops (WHEAT, CARROT, MELON): min(max yield - current yield, remaining watering days in the growth
        window), only when the crop matures by day 29.
    Seeds held, unplaced animals and fertiliser from animals are not valued. See the receipt for its noise.
"""
import torch

NUM_PRODUCTS = 9
TILE_PLANT, TILE_COOP, TILE_PASTURE = 3, 4, 5
ANIMAL_PRODUCT = (5, 6, 7)
CROP_FIRST_YIELD_DAY = (2, 2, 8, 10, 10)
CROP_MAX_YIELD_DAY = (4, 3, 8, 10, 12)
CROP_INTERVAL = (0, 0, 1, 2, 0)
CROP_MAX_YIELD = (6, 4, 4, 4, 6)
CROP_ONGOING = (False, False, True, True, False)
ANIMAL_FIRST_YIELD_DAY = (4, 8, 6)
ANIMAL_INTERVAL = (1, 2, 3)
FLAG_WATERED = 1
LAST_REFRESH_DAY = 28  # end-of-day refresh runs at steps 23, 47, ..., 695; step 719 is never processed
GROWING_KAPPA = 0.5
POTENTIALS = ('cash', 'networth', 'networth-growing')
FIELDS = ('money', 'market_price', 'shed', 'unit_inventory', 'unit_active', 'tile_kind', 'tile_crop', 'tile_animal', 'tile_yield')
FIELDS_B = FIELDS + ('step', 'tile_origin_day', 'tile_flags')


def _tile_product(kind, crop, animal):
    """Product index held on each tile (-1 none), as replay_analyze.farm_summary assigns yield_units."""
    animal_product = torch.as_tensor(ANIMAL_PRODUCT, device=kind.device, dtype=torch.long)[animal.long().clamp(0, 2)]
    plant = kind == TILE_PLANT
    has_animal = (~plant) & (animal >= 0)
    return torch.where(plant & (crop >= 0), crop.long(), torch.where(has_animal, animal_product, -1))


def _tile_price(price, product):
    # price [..., G, 9] -> per tile [..., G, 2, 10, 10]
    lead = product.shape[:-3]
    table = price.long()[..., None, :].expand(*lead, product.shape[-3], NUM_PRODUCTS)
    flat = torch.gather(table, -1, product.clamp_min(0).flatten(-2))
    return flat.reshape(product.shape) * (product >= 0)


def networth_a(state):
    """Version A, int64 [..., G, 2]."""
    price = state['market_price'].long()                               # [..., G, 9]
    shed = state['shed'][..., :NUM_PRODUCTS].long()                   # [..., G, 2, 9]
    inv = state['unit_inventory'][..., :NUM_PRODUCTS].long()          # [..., G, 2, U, 9]
    carried = (inv * state['unit_active'][..., None].long()).sum(-2)  # [..., G, 2, 9]
    goods = ((shed + carried) * price[..., None, :]).sum(-1)
    product = _tile_product(state['tile_kind'], state['tile_crop'], state['tile_animal'])
    ready = (state['tile_yield'].long() * _tile_price(price, product)).flatten(-2).sum(-1)
    return state['money'].long() + goods + ready


def growing_units(state):
    """Version B addition in future product units per tile, float64 [..., G, 2, 10, 10] with its product index."""
    kind, crop, animal = state['tile_kind'], state['tile_crop'], state['tile_animal']
    dev = kind.device
    day = (state['step'].long() // 24)[..., None, None, None]          # [..., G, 1, 1, 1]
    origin = state['tile_origin_day'].long()
    t = lambda v: torch.as_tensor(v, device=dev, dtype=torch.long)
    plant = (kind == TILE_PLANT) & (crop >= 0)
    has_animal = (kind != TILE_PLANT) & (animal >= 0)
    sc, sa = crop.long().clamp(0, 4), animal.long().clamp(0, 2)
    # Remaining refreshes run after days d in [day, 28]; the refresh after d sees s = d + 1 - origin - first and
    # produces when s >= 0 and s % interval == 0. Count those s in closed form (no per-day tensor).
    def multiples(lo, hi, every):
        lo = lo.clamp_min(0)
        return torch.where(hi >= lo, hi // every - (lo - 1) // every, 0)
    # animals
    a_first, a_int = t(ANIMAL_FIRST_YIELD_DAY)[sa], t(ANIMAL_INTERVAL)[sa]
    base = origin + a_first
    animal_units = torch.where(has_animal, multiples(day + 1 - base, LAST_REFRESH_DAY + 1 - base, a_int), 0)
    # ongoing crops: production_count = s // interval + 1 <= max yield  <=>  s <= (max - 1) * interval
    c_first, c_int, c_max = t(CROP_FIRST_YIELD_DAY)[sc], t(CROP_INTERVAL)[sc].clamp_min(1), t(CROP_MAX_YIELD)[sc]
    base = origin + c_first
    ongoing = t(CROP_ONGOING).bool()[sc]
    hi = torch.minimum(LAST_REFRESH_DAY + 1 - base, (c_max - 1) * c_int)
    ongoing_units = torch.where(plant & ongoing, multiples(day + 1 - base, hi, c_int), 0)
    # one-shot crops: remaining watering days inside [origin + (maxday+1)//2, origin + maxday], today only if unwatered
    maxday = t(CROP_MAX_YIELD_DAY)[sc]
    w_lo, w_hi = origin + (maxday + 1) // 2, origin + maxday
    watered_today = (state['tile_flags'].long() & FLAG_WATERED) > 0
    first_day = torch.where(watered_today, day + 1, day)
    days = (torch.minimum(w_hi, t(29)) - torch.maximum(w_lo, first_day) + 1).clamp_min(0)
    matures = origin + c_first <= 29
    oneshot_units = torch.where(plant & ~ongoing & matures, torch.minimum(days, (c_max - state['tile_yield'].long()).clamp_min(0)), 0)
    units = (animal_units + ongoing_units + oneshot_units).double()
    return units, _tile_product(kind, crop, animal)


def networth_b(state, kappa=GROWING_KAPPA):
    units, product = growing_units(state)
    price = state['market_price'].long()
    extra = (units * _tile_price(price, product).double()).flatten(-2).sum(-1)
    return networth_a(state).double() + kappa * extra


def networth(state, version='networth'):
    """float32 [..., G, 2] worth for potential `version` ('networth' = A, 'networth-growing' = B)."""
    if version == 'networth':return networth_a(state).to(torch.float32)
    if version == 'networth-growing':return networth_b(state).to(torch.float32)
    raise ValueError(f'Unknown networth version: {version}')


@torch.no_grad()
def potential_level(data, potential, turns, chunk=24):
    """Per-turn level fed to the dense potential, [turns, N] with N = 2*games (flat seat index 2*game+seat).
    'cash' returns data['state/money'] reshaped exactly as shaped-reward-v6/v7 does. Others are computed in turn
    chunks so the peak temporary memory stays small next to a full production rollout."""
    if potential == 'cash':return data['state/money'].reshape(turns, -1)
    if potential not in POTENTIALS:raise ValueError(f'Unknown reward potential: {potential}')
    fields = FIELDS_B if potential == 'networth-growing' else FIELDS
    missing = [f for f in fields if 'state/' + f not in data]
    if missing:raise KeyError(f'Rollout lacks simulator fields for {potential}: {missing}')
    out = []
    for begin in range(0, turns, chunk):
        state = {f: data['state/' + f][begin:min(turns, begin + chunk)] for f in fields}
        out.append(networth(state, potential).reshape(len(state['money']), -1))
    level = torch.cat(out, 0)
    if not torch.isfinite(level).all():raise FloatingPointError('Nonfinite networth potential')
    return level


MIGRATION_KEYS = ('reward_potential',)


def migration_allows(prior_config, current_config, migration):
    """networth-shaping-v1 contract migration: the config may differ ONLY in MIGRATION_KEYS, and both sides must match the
    receipt exactly (an absent key, i.e. the popped 'cash' default, is recorded as None)."""
    prior, current = dict(prior_config), dict(current_config)
    old = {k: prior.pop(k, None) for k in MIGRATION_KEYS}
    new = {k: current.pop(k, None) for k in MIGRATION_KEYS}
    return prior == current and new == migration.get('reward_potential') and old == migration.get('old_reward_potential')


def dense_rewards_weighted(level, final_cash, margin, beta, sigma, cash_weight, cash_center, shape, potential_weight):
    """networth-shaping-v2. level [T, N] (L_t before each turn), final_cash [N]. Per-turn potential W_p*squash((L_t-c)/sigma),
    terminal potential cash_weight*squash((final_cash-c)/sigma) (unchanged). Returns [T, N] rewards whose sum is
    margin_terms(margin) + Phi_T(final_cash) - W_p*squash((L_0-c)/sigma)."""
    from ppo.replay import cash_potential, margin_terms
    level = torch.as_tensor(level, dtype=torch.float32); final_cash = torch.as_tensor(final_cash, dtype=torch.float32)
    phi = torch.cat([cash_potential(level, float(potential_weight), cash_center, sigma, shape),
                     cash_potential(final_cash[None], cash_weight, cash_center, sigma, shape)], 0)
    rewards = phi[1:] - phi[:-1]
    rewards[-1] = rewards[-1] + margin_terms(margin, beta, sigma, shape)
    return rewards


MIGRATION_KEYS_V2 = ('reward_potential', 'reward_potential_weight')


def migration_allows_v2(prior_config, current_config, migration):
    """networth-shaping-v2 contract migration: the config may differ ONLY in reward_potential / reward_potential_weight, and
    both sides must equal the receipt's 'reward_shaping' / 'old_reward_shaping' dicts (absent key = None = the v42 default)."""
    prior, current = dict(prior_config), dict(current_config)
    old = {k: prior.pop(k, None) for k in MIGRATION_KEYS_V2}
    new = {k: current.pop(k, None) for k in MIGRATION_KEYS_V2}
    return prior == current and new == migration.get('reward_shaping') and old == migration.get('old_reward_shaping')
