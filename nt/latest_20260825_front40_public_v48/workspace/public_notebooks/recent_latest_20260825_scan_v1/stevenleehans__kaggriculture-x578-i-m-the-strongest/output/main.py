"""X594 — live-calibrated adaptive agent.

This standalone agent uses the public closed-loop planner architecture but
matches the opening observed in all 149 downloaded Ryo replays: five hires,
two sheep, two cows, eleven melons, six wheat seeds, and four wheat feed.
The later decisions remain observation-driven rather than replay-taped.
"""

"""Kaggriculture agent -- closed-loop planner (no replay tape).

Strategy summary
----------------
* Day 0: 4 sheep on the tiles next to the shed (wool + fertilizer are the
  early cash engine), 10 melons on the far NW tiles (dumped on day 10),
  a little wheat.  Animals sit in a compact zone around the shed so the
  daily feed / care / collect / harvest round is short.
* Cows are added from day 4 as cash arrives (8 by ~day 8, fewer if the town
  never opens a milk shop); sheep scale with yarn stores.
* Strawberries are planted early (from day 3, ~36-42 by day 11), fertilized
  on ages 9 and 13 (doubles every scheduled production) and harvested the
  moment fruit is on the plant.  Wheat fills everything else and is replanted
  until day 27; feed is bought when the farm is short.
* Land: NE around day 5-6, SW after the melon money.  A second melon wave is
  planted on day 11 only if the melon market is not glutted.
* Every turn a job list is built from the observed farm state and units
  (farmer + hands) are matched to jobs by urgency tier, priority, distance and
  stickiness; units finish the work on the tile they stand on before moving.
* Market: fertilizer / melon / premium goods are sold as soon as they reach
  the shed (whoever sells first gets the price), wheat beyond the feed
  reserve is sold, and everything is liquidated by step 718.
"""
import math
import os


def _knob(name, default):
    try:
        return type(default)(os.environ.get(name, default))
    except Exception:
        return default

# ----------------------------------------------------------------------------
# Engine constants (mirrors kaggle_environments/envs/kaggriculture)
# ----------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10, "first": 2, "maxday": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first": 2, "maxday": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first": 8, "maxday": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxday": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first": 10, "maxday": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
PREMIUM = ("MILK", "WOOL", "STRAWBERRY", "MELON")
MARKET_I0 = 10000
MARKET_PARAMS = {
    "WHEAT":      {"base": 25, "T": 400, "bf": "sqrt", "bt": 0.80, "af": "log", "at": 0.20},
    "CARROT":     {"base": 35, "T": 450, "bf": "hinge", "bt": 1.00, "af": "sqrt", "at": 0.70},
    "TOMATO":     {"base": 60, "T": 200, "bf": "hinge", "bt": 0.40, "af": "sqrt", "at": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "bf": "sqrt", "bt": 0.70, "af": "linear", "at": 1.60},
    "MELON":      {"base": 250, "T": 300, "bf": "log", "bt": 0.20, "af": "sq", "at": 3.60},
    "EGG":        {"base": 50, "T": 332, "bf": "hinge", "bt": 0.40, "af": "log", "at": 0.20},
    "MILK":       {"base": 160, "T": 122, "bf": "sqrt", "bt": 0.60, "af": "linear", "at": 1.60},
    "WOOL":       {"base": 200, "T": 105, "bf": "log", "bt": 0.20, "af": "sq", "at": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "bf": "linear", "bt": 0.40, "af": "linear", "at": 0.40},
}
SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
BOARD = 10
TPD = 24
LAST_DAY = 29
LAST_STEP = 718          # last step whose actions are executed
SHED_CAP = 100
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
CENTER = (4.5, 4.5)


def _shape(func, x, T):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "hinge":
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inv):
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    if inv < MARKET_I0:
        amp = p["bt"] * base / _shape(p["bf"], T, T)
        price = base + amp * _shape(p["bf"], MARKET_I0 - inv, T)
    else:
        amp = p["at"] * base / _shape(p["af"], T, T)
        price = base - amp * _shape(p["af"], inv - MARKET_I0, T)
    return max(1, int(round(price)))


def units_above_price(item, inv, floor_price, max_units):
    """How many units can be sold (one at a time) while price >= floor_price."""
    n = 0
    while n < max_units:
        pr = market_price(item, inv)
        if pr < floor_price:
            break
        n += 1
        if pr > 1:
            inv += 1
    return n


def fib_cost(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_total(n):
    return sum(fib_cost(i) for i in range(n))


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed(p):
    best, bd = SHED_TILES[0], 99
    for s in SHED_TILES:
        d = dist(p, s)
        if d < bd:
            best, bd = s, d
    return best, bd


def center_dist(t):
    return abs(t[0] - CENTER[0]) + abs(t[1] - CENTER[1])


def euclid2(t):
    return (t[0] - CENTER[0]) ** 2 + (t[1] - CENTER[1]) ** 2


def step_toward(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"


def quadrant(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


# ----------------------------------------------------------------------------
# persistent per-player state
# ----------------------------------------------------------------------------
ANIMAL_ZONE = 12
ANIMAL_FAR_LIMIT = _knob("KG_ANIMAL_FAR", 4.0)
SHEEP_CAP = _knob("KG_SHEEP_CAP", 10)
SHEEP_PER_YARN = _knob("KG_SHEEP_PER_YARN", 3)
SHEEP_LAST_BUY = _knob("KG_SHEEP_LAST_BUY", 21)
MELON2_MAX = _knob("KG_MELON2_MAX", 105)
MELON2_FULL = _knob("KG_MELON2_FULL", 75)
STRAW_SECTOR = _knob("KG_STRAW_SECTOR", 0)
STRAW_SECTORS = _knob("KG_STRAW_SECTORS", 6)
CARROT_CAP = _knob("KG_CARROT_CAP", 12)
CARROT_MIN_CAFES = _knob("KG_CARROT_MIN_CAFES", 2)
CARROT_LAST_PLANT = _knob("KG_CARROT_LAST", 25)
STRAW_CAP = _knob("KG_STRAW_CAP", 42)
STRAW_LAST_PLANT = _knob("KG_STRAW_LAST", 13)
OPEN_MELONS = 11
OPEN_SHEEP = 2
OPEN_COWS = 2
OPEN_WHEAT_SEEDS = 6
OPEN_WHEAT_FEED = 4
OPEN_HIRES = 5


class State:
    def __init__(self):
        self.last_step = -1
        self.roles = {}            # (x,y) -> "ANIMAL" | "MELON" | "STRAW" | "WHEAT"
        self.targets = {}          # unit idx -> (x,y) target last turn
        self.anchors = {}          # unit idx -> (x,y) territory anchor for today
        self.melon_tiles = []
        self.melon_plant_day = 0   # day on which melon tiles may be planted
        self.melon_wave2 = False
        self.melon_wave2_n = 12
        self.plan_day = -1
        self.hires_wanted = 0
        self.straw_target = 0
        self.carrot_target = 0
        self.cow_target = OPEN_COWS
        self.sheep_target = OPEN_SHEEP
        self.bought_today = {}
        self.pending_animals = {}  # animal -> count bought but not yet placed (approx)
        self.log = []


_STATES = {}


def get_state(player, step):
    st = _STATES.get(player)
    if st is None or step <= 0 or step < st.last_step:
        st = State()
        _STATES[player] = st
    st.last_step = step
    return st


# ----------------------------------------------------------------------------
# observation helpers
# ----------------------------------------------------------------------------
def G(d, k, default=None):
    if d is None:
        return default
    try:
        v = d.get(k, default) if hasattr(d, "get") else getattr(d, k, default)
    except Exception:
        v = default
    return default if v is None else v


def is_plant(t):
    return isinstance(t, dict) and t.get("kind") == "PLANT"


def is_weed(t):
    return isinstance(t, dict) and t.get("kind") == "WEED"


def is_struct(t):
    return isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")


def has_animal(t):
    return isinstance(t, dict) and "animal" in t and t.get("animal") is not None


def is_empty_struct(t):
    return is_struct(t) and not has_animal(t)


class World:
    pass


def parse(obs):
    W = World()
    W.player = int(G(obs, "player", 0))
    W.step = int(G(obs, "step", 0))
    W.day = int(G(obs, "day", W.step // TPD))
    W.hour = int(G(obs, "hour", W.step % TPD))
    farms = G(obs, "farms", [])
    W.farm = farms[W.player]
    W.opp = farms[1 - W.player] if len(farms) > 1 else None
    W.money = float(G(W.farm, "money", 0))
    W.tiles = G(W.farm, "tiles", [])
    W.farmer = tuple(G(W.farm, "farmer", (4, 4)))
    W.hands = [tuple(h) for h in G(W.farm, "hands", [])]
    W.units = [W.farmer] + W.hands
    W.unlocked = list(G(W.farm, "unlocked_quadrants", ["NW"]))
    W.hires_today = int(G(W.farm, "hires_today", 0))
    priv = G(obs, "private", {})
    W.shed = dict(G(priv, "shed", {}))
    W.seeds = dict(G(priv, "seeds", {}))
    invs = G(priv, "inventories", [{}])
    W.invs = [dict(iv) for iv in invs]
    while len(W.invs) < len(W.units):
        W.invs.append({})
    market = G(obs, "market", {})
    W.inv = dict(G(market, "inventory", {}))
    W.prices = dict(G(market, "prices", {}))
    town = G(obs, "town", {})
    W.shops = list(G(town, "unlocked_shops", []))
    W.steps_left = LAST_STEP - W.step          # actions after this one that still execute
    return W


def tile_at(W, p):
    x, y = p
    if 0 <= x < BOARD and 0 <= y < BOARD:
        return W.tiles[y][x]
    return "LOCKED"


def shop_count(W, product):
    n = 0
    for s in W.shops:
        prods = SHOPS.get(s, [])
        if product in prods:
            n += 2 if len(prods) == 1 else 1
    return n


def shed_total(W):
    return sum(v for v in W.shed.values() if v)


# ----------------------------------------------------------------------------
# tile roles / layout
# ----------------------------------------------------------------------------
def unlocked_tiles(W):
    out = []
    for y in range(BOARD):
        for x in range(BOARD):
            if W.tiles[y][x] != "LOCKED":
                out.append((x, y))
    return out


def assign_roles(st, W):
    """(Re)compute roles for all unlocked tiles. Animal zone = nearest tiles to
    center, then strawberry ring, then wheat. Melon tiles are fixed on day 0."""
    tiles = unlocked_tiles(W)
    n_animal = max(st.cow_target + st.sheep_target, 4)
    if W.day >= 6:
        n_animal = max(n_animal, ANIMAL_ZONE)
    elif W.day >= 2:
        n_animal = max(n_animal, 8)
    # existing animal structures always keep the ANIMAL role
    ordered = sorted(tiles, key=lambda t: (center_dist(t), euclid2(t), t[1], t[0]))
    roles = {}
    if W.day == 0 and not st.melon_tiles:
        # melons on the farthest NW tiles (rows 0-2 mostly)
        nw = [t for t in ordered if quadrant(*t) == "NW"]
        far = sorted(nw, key=lambda t: (-center_dist(t), t[1], t[0]))
        st.melon_tiles = far[:OPEN_MELONS]
    if st.melon_wave2 and not st.melon_tiles:
        # second wave: farthest free tiles of the unlocked farm
        far = sorted([t for t in tiles if tile_at(W, t) is None or is_weed(tile_at(W, t))],
                     key=lambda t: (-center_dist(t), t[1], t[0]))
        st.melon_tiles = far[:st.melon_wave2_n]
    melon_set = set(st.melon_tiles)
    animal_left = n_animal
    for t in ordered:
        tl = tile_at(W, t)
        if is_struct(tl):
            roles[t] = "ANIMAL"
            animal_left -= 1
    for t in ordered:
        if t in roles:
            continue
        if t in melon_set:
            tl = tile_at(W, t)
            if st.melon_plant_day == W.day or (is_plant(tl) and tl.get("crop") == "MELON"):
                roles[t] = "MELON"
                continue
    # animal zone: strictly the nearest tiles to the shed (regardless of what
    # currently grows there -- crops there are not replanted and the tile is
    # converted to a pasture once free).  Keeps every daily animal visit short.
    def free(t):
        tl = tile_at(W, t)
        return tl is None or is_weed(tl)
    cand = [t for t in ordered if t not in roles]
    n_zone = sum(1 for r in roles.values() if r == "ANIMAL")
    for t in cand:
        if animal_left <= 0:
            break
        tl = tile_at(W, t)
        # never claim a growing strawberry (17-day crop) for the zone
        if is_plant(tl) and tl.get("crop") == "STRAWBERRY":
            continue
        # beyond the core zone only accept tiles that are still close to the
        # shed -- a far pasture means a long daily feed trip and missed feeds
        if n_zone >= ANIMAL_ZONE and center_dist(t) > ANIMAL_FAR_LIMIT:
            continue
        roles[t] = "ANIMAL"
        animal_left -= 1
        n_zone += 1
    # strawberries: existing plants first, then free tiles nearest the centre
    straw_left = st.straw_target
    for t in ordered:
        if t not in roles:
            tl = tile_at(W, t)
            if is_plant(tl) and tl.get("crop") == "STRAWBERRY":
                roles[t] = "STRAW"
                straw_left -= 1
    rest = [t for t in ordered if t not in roles]
    if STRAW_SECTOR:
        # fill strawberries sector by sector (angular wedges around the shed) so
        # that plants of the same age -- which need water on the same days --
        # sit next to each other and one hand can sweep them.
        def skey(t):
            ang = math.atan2(t[1] - CENTER[1], t[0] - CENTER[0])
            sec = int((ang + math.pi) / (2 * math.pi) * STRAW_SECTORS) % STRAW_SECTORS
            return (0 if free(t) else 1, sec, center_dist(t))
    else:
        def skey(t):
            return (0 if free(t) else 1, center_dist(t))
    for t in sorted(rest, key=skey):
        if straw_left <= 0:
            break
        roles[t] = "STRAW"
        straw_left -= 1
    # carrots (only when pet cafes create demand): existing carrot plants keep
    # the role, then free tiles nearest the centre
    carrot_left = st.carrot_target
    for t in ordered:
        if t not in roles:
            tl = tile_at(W, t)
            if is_plant(tl) and tl.get("crop") == "CARROT":
                roles[t] = "CARROT"
                carrot_left -= 1
    rest = [t for t in ordered if t not in roles]
    for t in sorted(rest, key=lambda t: (0 if free(t) else 1, center_dist(t))):
        if carrot_left <= 0:
            break
        roles[t] = "CARROT"
        carrot_left -= 1
    for t in ordered:
        if t not in roles:
            roles[t] = "WHEAT"
    st.roles = roles


# ----------------------------------------------------------------------------
# daily macro plan
# ----------------------------------------------------------------------------
def count_animals(W, kind=None, farm=None):
    farm = farm if farm is not None else W.farm
    tiles = G(farm, "tiles", [])
    n = 0
    for row in tiles:
        for t in row:
            if has_animal(t) and (kind is None or t.get("animal") == kind):
                n += 1
    return n


def count_crop(W, crop, farm=None):
    farm = farm if farm is not None else W.farm
    n = 0
    for row in G(farm, "tiles", []):
        for t in row:
            if is_plant(t) and t.get("crop") == crop:
                n += 1
    return n


def update_targets(st, W):
    day = W.day
    yarn = shop_count(W, "WOOL") // 2          # yarn store counts double
    milk_shops = shop_count(W, "MILK")
    straw_shops = shop_count(W, "STRAWBERRY")
    # cows
    if day < 4:
        st.cow_target = OPEN_COWS
    elif day < 5:
        st.cow_target = 2
    elif day == 5:
        st.cow_target = 3
    elif day == 6:
        st.cow_target = 5
    elif day == 7:
        st.cow_target = 7
    else:
        st.cow_target = 8 if milk_shops >= 1 else 6
        if day >= 12 and milk_shops >= 4:
            st.cow_target = 9
    if day > 15:
        st.cow_target = min(st.cow_target, owned_animals(W, "COW"))
    # sheep: each yarn store eats 12 wool/day (~9 sheep worth); scale with them
    if day < 6:
        st.sheep_target = OPEN_SHEEP
    else:
        st.sheep_target = max(OPEN_SHEEP, min(SHEEP_CAP, OPEN_SHEEP + SHEEP_PER_YARN * yarn))
    if day > SHEEP_LAST_BUY:
        st.sheep_target = min(st.sheep_target, owned_animals(W, "SHEEP"))
    # second melon wave: only if the melon market is not glutted after day 10
    if day in (10, 11) and not st.melon_wave2 and count_crop(W, "MELON") == 0:
        melon_inv = W.inv.get("MELON", MARKET_I0)
        if melon_inv <= MARKET_I0 + MELON2_MAX and W.money >= 2000:
            st.melon_wave2 = True
            st.melon_wave2_n = 12 if melon_inv <= MARKET_I0 + MELON2_FULL else 6
            st.melon_tiles = []
            st.melon_plant_day = day
    # strawberries: plant early (they sell at scarcity prices from day ~13 on)
    # and grow the field as land / cash arrive.  Shops modulate the final size.
    if day < 3:
        tgt = 0
    elif day <= 5:
        tgt = 6
    elif day <= 7:
        tgt = 18
    elif day <= 9:
        tgt = 30
    else:
        tgt = 36
    if day >= 9:
        tgt += 3 * straw_shops - (4 if straw_shops == 0 else 0)
        tgt = max(24, min(STRAW_CAP, tgt))
    n_now = count_crop(W, "STRAWBERRY")
    if day > STRAW_LAST_PLANT:
        tgt = n_now
    st.straw_target = max(tgt, n_now)
    # carrots: pet cafes eat 12/day each (hinge scarcity curve -> price climbs
    # steeply once the town has drained more than 450); farmers markets 6/day.
    pet_cafes = sum(1 for s in W.shops if s == "PET_CAFE")
    if day <= CARROT_LAST_PLANT and pet_cafes >= CARROT_MIN_CAFES:
        st.carrot_target = min(CARROT_CAP, 4 * pet_cafes)
    else:
        st.carrot_target = 0


def desired_hands(st, W):
    """Estimate today's workload and convert to hires."""
    day = W.day
    if day == 0:
        return 5
    if day == 1:
        return 1
    if day >= LAST_DAY:
        return 8
    work = 0.0
    n_animals = count_animals(W)
    work += n_animals * 4.0
    plants = 0
    for row in W.tiles:
        for t in row:
            if is_plant(t):
                plants += 1
    work += plants * 1.25
    # planting / building work
    empties = sum(1 for t in unlocked_tiles(W) if tile_at(W, t) is None)
    work += min(empties, 20) * 1.0
    if day == 8:
        work += 40
    walking = HIRE_WALK
    n = int(math.ceil(work * walking / 22.0)) - 1
    if day <= 5:
        lo, hi = 3, 5
    elif day <= 7:
        lo, hi = 7, 9
    elif day <= 9:
        lo, hi = 10, 12
    elif day <= 27:
        lo, hi = MIN_HANDS, MAX_HANDS
        if 10 <= day <= 25 and W.money >= 6000:
            hi = MAX_HANDS_RICH
        lo = min(lo, hi)
    else:
        lo, hi = 9, 11
    n = max(lo, min(hi, n))
    return n


def plan_day(st, W):
    """Called at hour 0 (or first turn we see of a day)."""
    st.plan_day = W.day
    update_targets(st, W)
    assign_roles(st, W)
    st.hires_wanted = desired_hands(st, W)
    st.anchors = compute_anchors(W, st.hires_wanted)
    st.bought_today = {}


# ----------------------------------------------------------------------------
# job model
# ----------------------------------------------------------------------------
class Job:
    __slots__ = ("tile", "actions", "prio", "need", "need_n", "key", "taken", "kind")

    def __init__(self, tile, actions, prio, need=None, need_n=1, kind=""):
        self.tile = tile
        self.actions = actions
        self.prio = prio
        self.need = need
        self.need_n = need_n
        self.key = (tile, kind)
        self.taken = False
        self.kind = kind


def water_wanted(crop, age, tl, day):
    """Should this plant be watered today?"""
    if age == 0:
        return True
    cu = int(tl.get("consecutive_unwatered", 0))
    if cu >= 1:
        return True                       # would die tonight otherwise
    if crop == "WHEAT":
        return age in (2, 3, 4)
    if crop == "CARROT":
        return age in (2, 3)
    if crop == "MELON":
        if age >= 6:
            return True                   # bonus window 6..10 (+1 per watered day)
        return age % 2 == 0
    if crop == "STRAWBERRY":
        if age >= 9:
            return age % 2 == 1 or age <= 9
        return age % 2 == 0
    if crop == "TOMATO":
        return True
    return age % 2 == 0


STRAW_MIN_YIELD = _knob("KG_STRAW_MIN_YIELD", 1)
PREM_DROP_SCALE = _knob("KG_PREM_DROP_SCALE", 2.0)


def strawberry_harvest_wanted(age, yield_units, day):
    # strawberry price collapses linearly with supply: sell as early as we can,
    # so harvest whatever is on the plant whenever it is ready.
    if yield_units <= 0 or age < 10:
        return False
    if age >= 16 or day >= LAST_DAY or yield_units >= STRAW_MIN_YIELD:
        return True
    # a production tonight would exceed the held cap of 4
    return age in (11, 13, 15) and yield_units >= 3


def next_production_day(tl, kind):
    a = ANIMALS[kind]
    placed = int(tl.get("placed_day", 0))
    return placed, a


def build_jobs(st, W):
    jobs = []
    day, hour = W.day, W.hour
    endgame = day >= LAST_DAY
    shed_wheat = int(W.shed.get("WHEAT", 0))
    shed_fert = int(W.shed.get("FERTILIZER", 0))
    roles = st.roles
    melon_day = day >= 8
    build_left = structures_needed(st, W)
    plant_wheat_prio = 66 if day >= 14 else 58

    # count seeds available for planting
    seeds = dict(W.seeds)
    animals_in_shed = {a: int(W.shed.get(a, 0)) for a in ANIMALS}
    # animals carried by units count too
    wheat_available = shed_wheat
    fert_available = shed_fert
    for iv in W.invs:
        for a in ANIMALS:
            animals_in_shed[a] += int(iv.get(a, 0))
        wheat_available += int(iv.get("WHEAT", 0))
        fert_available += int(iv.get("FERTILIZER", 0))

    for (x, y), role in roles.items():
        tl = W.tiles[y][x]
        t = (x, y)
        if tl == "LOCKED":
            continue
        # ---------------- plants ----------------
        if is_plant(tl):
            crop = tl.get("crop")
            age = day - int(tl.get("planted_day", day))
            yu = int(tl.get("yield_units", 0))
            watered = bool(tl.get("watered_today", False))
            fert_until = int(tl.get("fertilized_until_day", -1))
            acts = []
            prio = 0
            if crop == "MELON":
                # engine: HARVEST is a no-op before first_yield_day (age 10)
                if age >= 10:
                    acts2 = []
                    if not watered and yu < 6 and age <= 12 and not endgame:
                        acts2.append("WATER")
                    acts2.append("HARVEST")
                    jobs.append(Job(t, acts2, 100, kind="harvest"))
                    continue
                if not watered and water_wanted(crop, age, tl, day):
                    acts.append("WATER")
                    prio = max(prio, 76 if age < 6 else 79)
                    if int(tl.get("consecutive_unwatered", 0)) >= 1:
                        prio = max(prio, 82 if hour < DYING_HOUR else 91)
                    if hour >= URGENT_HOUR:
                        prio = max(prio, 92)
                if acts:
                    jobs.append(Job(t, acts, prio, kind="tend"))
                continue
            if crop == "WHEAT" or crop == "CARROT":
                mx = CROPS[crop]["maxday"]
                if age >= mx or (endgame and yu > 0 and age >= CROPS[crop]["first"]):
                    acts2 = []
                    win_start = (mx + 1) // 2
                    if (not watered and win_start <= age <= mx and yu < CROPS[crop]["max_yield"]
                            and not (endgame and hour >= 21)):
                        acts2.append("WATER")      # +1 unit before harvesting
                    acts2.append("HARVEST")
                    pr = 84 if age >= mx else 70
                    if age > mx:
                        pr = 96
                    jobs.append(Job(t, acts2, pr, kind="tend"))
                    continue
                if not watered and water_wanted(crop, age, tl, day):
                    pr = 74
                    if int(tl.get("consecutive_unwatered", 0)) >= 1:
                        pr = 82 if hour < DYING_HOUR else 91
                    if hour >= URGENT_HOUR:
                        pr = max(pr, 92)
                    jobs.append(Job(t, ["WATER"], pr, kind="tend"))
                continue
            if crop == "STRAWBERRY":
                acts = []
                pr = 0
                need = None
                if age >= 16:
                    # last production landed at end of age 15: harvest and clear the
                    # tile right away so wheat can follow (it would decay into a weed)
                    if yu > 0:
                        jobs.append(Job(t, ["HARVEST", "DIG"] if day <= 27 else ["HARVEST"], 88, kind="tend"))
                    elif day <= 27:
                        jobs.append(Job(t, ["DIG"], 64, kind="dig"))
                    continue
                if strawberry_harvest_wanted(age, yu, day):
                    acts.append("HARVEST")
                    pr = 76 if age < 16 else 88
                    if endgame:
                        pr = 95
                # fertilizer doubles the scheduled productions (end of ages 9,11,13,15)
                # while active (3 days).  Ideal: fertilize on age 9 and 13.  Fallbacks
                # if a window was missed: age 11 (covers 11,13) / age 15 (covers 15).
                fz = False
                if not endgame and fert_until < day and fert_available > 0:
                    if age in (9, 13):
                        fz, fp = True, FERT_PRIO
                    elif age == 11 or age == 15:
                        fz, fp = True, FERT_PRIO - 6
                if fz:
                    acts.append("FERTILIZE")
                    need = "FERTILIZER"
                    fert_available -= 1
                    pr = max(pr, fp)
                if not watered and water_wanted(crop, age, tl, day) and not endgame:
                    acts.append("WATER")
                    p2 = 74
                    if age in (9, 11, 13, 15):
                        p2 = 78
                    if int(tl.get("consecutive_unwatered", 0)) >= 1:
                        p2 = 82 if hour < DYING_HOUR else 91
                    if hour >= URGENT_HOUR:
                        p2 = max(p2, 92)
                    pr = max(pr, p2)
                if acts:
                    jobs.append(Job(t, acts, pr, need=need, kind="tend"))
                continue
            # other crops (tomato) -- generic
            if yu > 0 and age >= CROPS[crop]["first"]:
                jobs.append(Job(t, ["HARVEST"], 70, kind="tend"))
            elif not watered and not endgame:
                jobs.append(Job(t, ["WATER"], 74, kind="tend"))
            continue

        # ---------------- animals ----------------
        if has_animal(tl):
            kind = tl.get("animal")
            a = ANIMALS[kind]
            placed = int(tl.get("placed_day", 0))
            yu = int(tl.get("yield_units", 0))
            fed = bool(tl.get("fed_today", False))
            cared = bool(tl.get("cared_today", False))
            pend = int(tl.get("pending_care_bonus", 0))
            fert_av = bool(tl.get("fertilizer_available", False))
            cu = int(tl.get("consecutive_unfed", 0))
            # future production days (end-of-day e) : e+1 - placed - first == k*interval, e <= 28
            prod_tonight = False
            future_prod = False
            for e in range(day, 29):
                ds = e + 1 - placed - a["first"]
                if ds >= 0 and ds % a["interval"] == 0:
                    if e == day:
                        prod_tonight = True
                    future_prod = True
                    break
            acts = []
            pr = 0
            if endgame:
                if yu > 0:
                    jobs.append(Job(t, ["HARVEST"], 95, kind="animal"))
                continue
            need_feed = (not fed) and (day < 28 or prod_tonight) and day >= 1
            if day == 28 and not fed and not prod_tonight:
                need_feed = False
            if need_feed and wheat_available <= 0:
                need_feed = False          # nothing to feed with right now
            if need_feed:
                acts.append("FEED")
                pr = 80
                if cu >= 1:
                    pr = 86
                if hour >= FEED_URGENT_HOUR:
                    pr = max(pr, 90)
                if hour >= URGENT_HOUR:
                    pr = max(pr, 93)
            if not cared and future_prod and pend < 5 and (fed or need_feed) and day < 28:
                acts.append("CARE")
                pr = max(pr, CARE_PRIO)
            if yu > 0:
                acts.append("HARVEST")
                pr = max(pr, ANIMAL_HARVEST_PRIO)
            if fert_av and day < LAST_DAY:
                acts.append("COLLECT_FERTILIZER")
                pr = max(pr, COLLECT_PRIO)
            if acts:
                needwheat = "FEED" in acts
                jobs.append(Job(t, acts, pr, need="WHEAT" if needwheat else None, kind="animal"))
            continue

        # ---------------- empty structure ----------------
        if is_empty_struct(tl):
            kind_needed = "COW" if tl.get("kind") == "PASTURE" else "GOOSE"
            # choose which animal to place: prefer whichever we have
            for cand in (["COW", "SHEEP"] if tl.get("kind") == "PASTURE" else ["GOOSE"]):
                if animals_in_shed.get(cand, 0) > 0:
                    animals_in_shed[cand] -= 1
                    jobs.append(Job(t, ["PLACE", cand], 72, need=cand, kind="place"))
                    break
            else:
                if role != "ANIMAL" and not endgame and day < 27:
                    jobs.append(Job(t, ["DIG"], 20, kind="dig"))
            continue

        # ---------------- weeds ----------------
        if is_weed(tl):
            if not endgame and day <= 27 and role in ("WHEAT", "STRAW", "ANIMAL", "MELON", "CARROT"):
                jobs.append(Job(t, ["DIG"], 60 if (role == "STRAW" and 8 <= day <= 15) else (62 if day >= 14 else 52), kind="dig"))
            continue

        # ---------------- empty tile ----------------
        if tl is None:
            if endgame:
                continue
            if role == "ANIMAL":
                # build a pasture only for animals that are on hand or about to
                # be bought (nearest free zone tiles first -- roles are ordered)
                if build_left > 0:
                    build_left -= 1
                    jobs.append(Job(t, ["BUILD_PASTURE"], 66, kind="build"))
                elif (day <= 1 or day >= 18) and day <= WHEAT_LAST_PLANT and seeds.get("WHEAT", 0) > 0:
                    seeds["WHEAT"] -= 1
                    jobs.append(Job(t, ["PLANT", "WHEAT"], plant_wheat_prio, kind="plant"))
                continue
            if role == "MELON":
                if day == st.melon_plant_day and seeds.get("MELON", 0) > 0:
                    seeds["MELON"] -= 1
                    jobs.append(Job(t, ["PLANT", "MELON"], 70, kind="plant"))
                elif day > st.melon_plant_day and seeds.get("WHEAT", 0) > 0 and day <= WHEAT_LAST_PLANT:
                    seeds["WHEAT"] -= 1
                    jobs.append(Job(t, ["PLANT", "WHEAT"], plant_wheat_prio, kind="plant"))
                continue
            if role == "STRAW":
                if seeds.get("STRAWBERRY", 0) > 0 and day <= STRAW_LAST_PLANT:
                    seeds["STRAWBERRY"] -= 1
                    jobs.append(Job(t, ["PLANT", "STRAWBERRY"], 68, kind="plant"))
                elif seeds.get("WHEAT", 0) > 0 and day <= WHEAT_LAST_PLANT:
                    seeds["WHEAT"] -= 1
                    jobs.append(Job(t, ["PLANT", "WHEAT"], plant_wheat_prio - 2, kind="plant"))
                continue
            if role == "CARROT":
                if seeds.get("CARROT", 0) > 0 and day <= CARROT_LAST_PLANT:
                    seeds["CARROT"] -= 1
                    jobs.append(Job(t, ["PLANT", "CARROT"], plant_wheat_prio, kind="plant"))
                elif seeds.get("WHEAT", 0) > 0 and day <= WHEAT_LAST_PLANT:
                    seeds["WHEAT"] -= 1
                    jobs.append(Job(t, ["PLANT", "WHEAT"], plant_wheat_prio - 2, kind="plant"))
                continue
            if role == "WHEAT":
                if seeds.get("WHEAT", 0) > 0 and day <= WHEAT_LAST_PLANT:
                    seeds["WHEAT"] -= 1
                    jobs.append(Job(t, ["PLANT", "WHEAT"], plant_wheat_prio, kind="plant"))
                continue
    return jobs


def owned_animals(W, kind):
    """Placed + in shed + carried."""
    n = count_animals(W, kind) + int(W.shed.get(kind, 0))
    for iv in W.invs:
        n += int(iv.get(kind, 0))
    return n


def structures_needed(st, W):
    """How many more pastures we should have ready: unplaced animals + animals
    we still intend to buy, minus empty structures already standing."""
    if W.day > SHEEP_LAST_BUY:
        return 0
    cows = owned_animals(W, "COW")
    sheep = owned_animals(W, "SHEEP")
    placed = count_animals(W)
    unplaced = (cows + sheep) - placed
    empties = 0
    for row in W.tiles:
        for t in row:
            if is_empty_struct(t):
                empties += 1
    want = max(0, st.cow_target - cows) + max(0, st.sheep_target - sheep)
    return max(0, unplaced + want - empties)


def want_more_structures(st, W):
    return structures_needed(st, W) > 0


# ----------------------------------------------------------------------------
# assignment
# ----------------------------------------------------------------------------
URGENT_HOUR = _knob("KG_URGENT_HOUR", 15)
FERT_PRIO = _knob("KG_FERT_PRIO", 84)
CARE_PRIO = _knob("KG_CARE_PRIO", 70)
ANIMAL_HARVEST_PRIO = _knob("KG_AH_PRIO", 68)
COLLECT_PRIO = _knob("KG_COLLECT_PRIO", 62)
SELL_HOUR_ONLY = _knob("KG_SELL_HOUR_ONLY", 0)
SELL_HOUR_LAST = _knob("KG_SELL_HOUR_LAST", 0)
FEED_URGENT_HOUR = _knob("KG_FEED_URGENT_HOUR", 11)
MAX_HANDS = _knob("KG_MAX_HANDS", 13)
HIRE_WALK = _knob("KG_HIRE_WALK", 2.1)
MIN_HANDS = _knob("KG_MIN_HANDS", 12)
MAX_HANDS_RICH = _knob("KG_MAX_HANDS_RICH", 14)
DYING_HOUR = _knob("KG_DYING_HOUR", 13)
WHEAT_LAST_PLANT = _knob("KG_WHEAT_LAST", 26)
BULK_DROP = _knob("KG_BULK_DROP", 0)
OVERFLOW_SELL = _knob("KG_OVERFLOW", 1)
WALK_COST = 25.0
STICK_BONUS = 60.0
ON_TILE_BONUS = _knob("KG_ON_TILE", 3500.0)
TIER_GAP = 3000.0
PRIO_W = 8.0


def job_tier(prio):
    if prio >= 90:
        return 3
    if prio >= 60:
        return 2
    return 1


ANCHOR_W = _knob("KG_ANCHOR_W", 4.0)


def compute_anchors(W, n_hands):
    """Split the worked tiles into angular sectors around the shed and give
    each hired hand the centroid of one sector as its territory anchor.
    (Hand indices are 1..n_hands; the farmer stays a free agent.)"""
    anchors = {}
    if n_hands <= 0 or ANCHOR_W <= 0:
        return anchors
    pts = []
    for y in range(BOARD):
        for x in range(BOARD):
            tl = W.tiles[y][x]
            if tl == "LOCKED":
                continue
            # tiles that will need visits: plants and animals; empty tiles a bit
            w = 1.0
            if tl is None:
                w = 0.35
            pts.append((math.atan2(y - CENTER[1], x - CENTER[0]), x, y, w))
    if not pts:
        return anchors
    pts.sort()
    total_w = sum(p[3] for p in pts)
    per = total_w / n_hands
    acc = 0.0
    grp = []
    gi = 1
    for p in pts:
        grp.append(p)
        acc += p[3]
        if acc >= per * gi and gi <= n_hands:
            sx = sum(q[1] * q[3] for q in grp) / max(1e-9, sum(q[3] for q in grp))
            sy = sum(q[2] * q[3] for q in grp) / max(1e-9, sum(q[3] for q in grp))
            anchors[gi] = (sx, sy)
            gi += 1
            grp = []
    if grp and gi <= n_hands:
        sx = sum(q[1] * q[3] for q in grp) / max(1e-9, sum(q[3] for q in grp))
        sy = sum(q[2] * q[3] for q in grp) / max(1e-9, sum(q[3] for q in grp))
        anchors[gi] = (sx, sy)
    return anchors


def assign(st, W, jobs):
    """Return list of unit actions (index aligned with W.units)."""
    n_units = len(W.units)
    actions = [["PASS"] for _ in range(n_units)]
    day, hour = W.day, W.hour
    endgame = day >= LAST_DAY
    turns_left_today = TPD - 1 - hour            # turns after this one today
    shed = dict(W.shed)
    shed_room = SHED_CAP - shed_total(W)

    # per-unit inventories (mutable copies for reservation)
    invs = [dict(iv) for iv in W.invs]

    # ---- special: drop jobs for units carrying valuable stuff or endgame ----
    unit_forced = {}
    for i, pos in enumerate(W.units):
        iv = invs[i]
        val = 0.0
        cnt = 0
        for it, n in iv.items():
            if it in MARKET_PARAMS and n > 0:
                val += n * W.prices.get(it, MARKET_PARAMS[it]["base"])
                cnt += n
        sd, dsh = nearest_shed(pos)
        if cnt == 0:
            continue
        must_drop = False
        if endgame:
            # must reach the shed and drop before the last executed step
            if dsh >= W.steps_left - 1:
                must_drop = True
        melons = iv.get("MELON", 0)
        if melons >= 6:
            must_drop = True
        # premium goods: bank them promptly so they are sold today (ahead of
        # opponents who sell the next morning)
        prem_val = sum(iv.get(it, 0) * W.prices.get(it, MARKET_PARAMS[it]["base"]) for it in PREMIUM)
        if prem_val >= 300 * PREM_DROP_SCALE and dsh <= 2:
            must_drop = True
        if prem_val >= 900 * PREM_DROP_SCALE and dsh <= 4:
            must_drop = True
        if prem_val >= 2000 * PREM_DROP_SCALE and dsh <= 7:
            must_drop = True
        # bulk: do not walk around with a big load (it all lands in the shed at
        # night and overflow is destroyed; selling earlier is better anyway)
        if BULK_DROP:
            if cnt >= 8 and dsh <= 2:
                must_drop = True
            if cnt >= 14 and dsh <= 4:
                must_drop = True
            if hour >= 20 and cnt >= 4 and dsh <= 3:
                must_drop = True
        if hour >= 22 and dsh <= 1 and shed_room > cnt:
            must_drop = True
        if must_drop:
            unit_forced[i] = ("DROP", sd)

    # ---- item logistics: how many jobs need each item, how much is carried ----
    need_jobs = {}
    for job in jobs:
        if job.need is not None:
            need_jobs[job.need] = need_jobs.get(job.need, 0) + 1
    carried = {}
    carriers = {}
    for i, iv in enumerate(invs):
        for it, n in iv.items():
            if n > 0:
                carried[it] = carried.get(it, 0) + n
                carriers[it] = carriers.get(it, 0) + 1

    # ---- build candidate (unit, job) scores ----
    pairs = []
    for i, pos in enumerate(W.units):
        if i in unit_forced:
            continue
        iv = invs[i]
        stick = st.targets.get(i)
        for j, job in enumerate(jobs):
            d = dist(pos, job.tile)
            extra = 0
            carrier_bonus = 0
            if job.need is not None:
                if iv.get(job.need, 0) > 0:
                    carrier_bonus = 45
                else:
                    # need to fetch from shed first
                    if shed.get(job.need, 0) <= 0:
                        continue
                    sd, dsh = nearest_shed(pos)
                    extra = dsh + dist(sd, job.tile) - d
            total_d = d + extra
            if endgame:
                # must be able to do the job, get back to the shed and drop
                _, dback = nearest_shed(job.tile)
                if total_d + len(job.actions) + dback + 1 > W.steps_left:
                    continue
            else:
                if total_d + len(job.actions) > turns_left_today + 1 and job.prio < 100:
                    continue
            score = job_tier(job.prio) * TIER_GAP + job.prio * PRIO_W - total_d * WALK_COST + carrier_bonus
            if stick == job.tile:
                score += STICK_BONUS
            anc = st.anchors.get(i)
            if anc is not None and job.kind != "animal":
                score -= ANCHOR_W * (abs(job.tile[0] - anc[0]) + abs(job.tile[1] - anc[1]))
            if d == 0 and extra == 0:
                # finish the work on the tile we stand on: an action here costs
                # no walking, so it is nearly always the best use of the turn
                score += ON_TILE_BONUS
            score += 6 * len(job.actions)
            pairs.append((score, i, j, extra))
    pairs.sort(key=lambda p: -p[0])

    unit_done = set(unit_forced.keys())
    job_taken = set()
    new_targets = {}
    fetch_reserved = {}     # item -> count reserved for pickup this turn
    for score, i, j, extra in pairs:
        if i in unit_done or j in job_taken:
            continue
        job = jobs[j]
        pos = W.units[i]
        iv = invs[i]
        job_taken.add(j)
        unit_done.add(i)
        new_targets[i] = job.tile
        # need item?
        if job.need is not None and iv.get(job.need, 0) <= 0:
            sd, dsh = nearest_shed(pos)
            if dsh == 0:
                # pick up now
                avail = shed.get(job.need, 0) - fetch_reserved.get(job.need, 0)
                if avail <= 0:
                    unit_done.discard(i)
                    job_taken.discard(j)
                    continue
                k = 1
                if job.need in ("WHEAT", "FERTILIZER"):
                    # jobs still needing this item that no carried stock covers
                    remaining = sum(1 for jj, jb in enumerate(jobs) if jb.need == job.need and jj not in job_taken) + 1
                    uncovered = max(1, remaining - carried.get(job.need, 0))
                    k = max(1, min(avail, uncovered, 5))
                    carried[job.need] = carried.get(job.need, 0) + k
                fetch_reserved[job.need] = fetch_reserved.get(job.need, 0) + k
                actions[i] = ["PICKUP", job.need, k]
            else:
                actions[i] = [step_toward(pos, sd)]
            continue
        if pos == job.tile:
            act = list(job.actions[0]) if isinstance(job.actions[0], list) else None
            a0 = job.actions[0]
            if a0 == "PLACE":
                actions[i] = ["PLACE", job.actions[1]]
            elif a0 == "PLANT":
                actions[i] = ["PLANT", job.actions[1]]
            else:
                actions[i] = [a0]
        else:
            actions[i] = [step_toward(pos, job.tile)]

    for i, (kind, sd) in unit_forced.items():
        pos = W.units[i]
        if pos == sd or pos in SHED_TILES:
            actions[i] = ["DROP"]
        else:
            actions[i] = [step_toward(pos, sd)]
        new_targets[i] = sd

    # idle units: on the last day walk to shed; otherwise drift toward shed a bit
    for i, pos in enumerate(W.units):
        if i in unit_done:
            continue
        iv = invs[i]
        cnt = sum(v for k, v in iv.items() if k in MARKET_PARAMS)
        sd, dsh = nearest_shed(pos)
        if cnt > 0:
            actions[i] = ["DROP"] if dsh == 0 else [step_toward(pos, sd)]
            new_targets[i] = sd
        elif dsh > 1:
            # drift back toward the centre so the next job is close
            actions[i] = [step_toward(pos, sd)]
        else:
            actions[i] = ["PASS"]
    st.targets = new_targets
    return actions


# ----------------------------------------------------------------------------
# market
# ----------------------------------------------------------------------------
def daily_drain(W, item):
    """Units per day the town removes from the market for this product."""
    d = 1 if item != "FERTILIZER" else 0
    d += shop_count(W, item) * (TPD // 4)
    return d


def premium_floor(item, W):
    """Minimum sale price. Premium goods glut fast and the seller who moves
    first gets the price, so we sell nearly everything immediately and only
    hold when the market is deeply glutted AND the town drain will lift it."""
    base = MARKET_PARAMS[item]["base"]
    if W.step >= LAST_STEP - 1:
        return 1
    day = W.day
    if item == "MELON":
        return 1
    drain = daily_drain(W, item)
    if item == "WOOL":
        f = 0.55 if drain >= 7 else 0.20
    elif item == "MILK":
        f = 0.55 if drain >= 7 else 0.30
    else:  # strawberry
        f = 0.50 if drain >= 7 else 0.30
    if day >= 25:
        # linear decay to 0 at last step
        frac = (LAST_STEP - W.step) / float(LAST_STEP - 25 * TPD)
        f = f * max(0.0, min(1.0, frac))
    return max(1, int(base * f))


def market_orders(st, W, actions):
    orders = []
    day, hour = W.day, W.hour
    endgame = day >= LAST_DAY
    money = W.money
    shed = W.shed
    n_animals = count_animals(W)
    n_cows = owned_animals(W, "COW")
    n_sheep = owned_animals(W, "SHEEP")
    # cash we must keep for tomorrow's hires and feed
    cash_reserve = hire_total(min(12, st.hires_wanted)) + n_animals * 45 + 60
    if day == 0:
        cash_reserve = 0

    # ---------------- sells ----------------
    # planned fertilizer needs (only imminent ones)
    fert_need = 0
    for (x, y), role in st.roles.items():
        tl = W.tiles[y][x]
        if is_plant(tl):
            age = day - int(tl.get("planted_day", day))
            crop = tl.get("crop")
            fu = int(tl.get("fertilized_until_day", -1))
            if crop == "STRAWBERRY" and (age in (8, 9, 11, 12, 13, 15)) and fu < day:
                fert_need += 1
    fert_need = min(fert_need, 30)
    if day >= 27:
        fert_need = 0
    if money < 40 and day < 8:
        fert_need = 0            # cash emergency: feed comes first
    fert = int(shed.get("FERTILIZER", 0))
    sell_f = fert - fert_need
    if sell_f > 0 and (W.prices.get("FERTILIZER", 100) >= 12 or endgame or shed_total(W) > 80):
        orders.append(["SELL", "FERTILIZER", sell_f])

    wheat = int(shed.get("WHEAT", 0))
    days_left = LAST_DAY - day
    reserve = n_animals * min(3, max(0, 28 - day + 1)) if day < 28 else 0
    if day >= 27:
        reserve = n_animals * max(0, 28 - day + 1)
    if endgame:
        reserve = 0
    sell_w = wheat - reserve
    if sell_w > 0 and (W.prices.get("WHEAT", 25) >= 22 or endgame or shed_total(W) > 75):
        orders.append(["SELL", "WHEAT", sell_w])

    melon = int(shed.get("MELON", 0))
    if melon > 0:
        orders.append(["SELL", "MELON", melon])

    total = shed_total(W)
    carried_total = sum(sum(int(v) for k, v in iv.items() if k in MARKET_PARAMS) for iv in W.invs)
    # everything carried lands in the shed at the end of the day; whatever does
    # not fit is destroyed -> from the afternoon on, sell down to make room
    projected = total + carried_total
    if hour >= 16 and OVERFLOW_SELL:
        overflow = max(0, projected - 88)
    else:
        overflow = max(0, total - 80)
    # already-planned wheat/fertilizer sales reduce the overflow
    for o in orders:
        overflow -= int(o[2])
    overflow = max(0, overflow)
    # cheapest-first order for forced sales
    prem = ["MILK", "WOOL", "STRAWBERRY", "EGG", "CARROT", "TOMATO"]
    prem.sort(key=lambda it: W.prices.get(it, MARKET_PARAMS[it]["base"]))
    for item in prem:
        n = int(shed.get(item, 0))
        if n <= 0:
            continue
        inv = W.inv.get(item, MARKET_I0)
        if item in ("EGG", "CARROT", "TOMATO"):
            floor_p = 1 if endgame else int(MARKET_PARAMS[item]["base"] * 0.7)
        else:
            floor_p = premium_floor(item, W)
        k = units_above_price(item, inv, floor_p, n)
        if SELL_HOUR_ONLY and item in ("MILK", "WOOL", "STRAWBERRY") and not endgame and day < 27:
            # sell premium goods first thing in the morning (before the typical
            # opponent's hour-1 dump, after the overnight town drain), not during
            # the day into a depressed market
            if hour > SELL_HOUR_LAST:
                k = 0
        if overflow > 0 and k < n:
            extra = min(n - k, overflow)
            k += extra
            overflow -= extra
        if endgame:
            k = n
        if k > 0:
            orders.append(["SELL", item, k])
    if overflow > 0:
        # still too much: dump wheat reserve / fertilizer reserve too
        w_left = int(shed.get("WHEAT", 0)) - sum(int(o[2]) for o in orders if o[1] == "WHEAT")
        if w_left > 0:
            k = min(w_left, overflow)
            orders.append(["SELL", "WHEAT", k])
            overflow -= k
        f_left = int(shed.get("FERTILIZER", 0)) - sum(int(o[2]) for o in orders if o[1] == "FERTILIZER")
        if overflow > 0 and f_left > 0:
            orders.append(["SELL", "FERTILIZER", min(f_left, overflow)])
    # estimated proceeds (conservative) to fund purchases this turn
    proceeds = 0.0
    for o in orders:
        if o[0] == "SELL":
            inv = W.inv.get(o[1], MARKET_I0)
            tot = 0
            for _ in range(int(o[2])):
                pr = market_price(o[1], inv)
                tot += pr
                if pr > 1:
                    inv += 1
            proceeds += tot
    cash = money + 0.9 * proceeds

    # ---------------- hires ----------------
    if not endgame or hour <= 2:
        want = st.hires_wanted
        if endgame:
            want = min(want, 8)
        have = W.hires_today
        n_hire = max(0, want - have)
        # only hire in the first hours of the day
        if hour > 3:
            n_hire = 0
        cost = 0
        k = 0
        while k < n_hire and len(orders) < 10:
            c = fib_cost(have + k)
            if cash - cost - c < 0:
                break
            cost += c
            k += 1
        for _ in range(k):
            orders.append(["HIRE"])
        cash -= cost

    # ---------------- land ----------------
    n_unl = len(W.unlocked)
    if n_unl < 3 and not endgame and day <= 16 and len(orders) < 10:
        price = LAND_PRICES[n_unl - 1]
        ok = False
        cow_deficit = max(0, st.cow_target - n_cows)
        if n_unl == 1 and day >= 4 and cash - price >= cash_reserve:
            ok = True
        if n_unl == 2 and day >= 8 and cash - price >= cash_reserve + 400 * cow_deficit + 300:
            ok = True
        if ok:
            orders.append(["BUY_LAND"])
            cash -= price

    # ---------------- seeds ----------------
    if not endgame and len(orders) < 10:
        want = {}
        for (x, y), role in st.roles.items():
            tl = W.tiles[y][x]
            if tl is None or is_weed(tl):
                if role == "MELON" and day == st.melon_plant_day:
                    want["MELON"] = want.get("MELON", 0) + 1
                elif role == "STRAW" and day >= 3 and day <= STRAW_LAST_PLANT:
                    want["STRAWBERRY"] = want.get("STRAWBERRY", 0) + 1
                elif role == "CARROT" and day <= CARROT_LAST_PLANT:
                    want["CARROT"] = want.get("CARROT", 0) + 1
                elif role in ("WHEAT", "STRAW", "MELON", "CARROT") and day >= 1 and day <= WHEAT_LAST_PLANT:
                    want["WHEAT"] = want.get("WHEAT", 0) + 1
                elif role == "ANIMAL" and (day <= 1 or day >= 18) and day <= WHEAT_LAST_PLANT and not want_more_structures(st, W):
                    want["WHEAT"] = want.get("WHEAT", 0) + 1
        # only buy for what can be planted today (labor bound) + small buffer
        max_plant_today = max(2, len(W.units) * 2)
        for crop in ("MELON", "STRAWBERRY", "CARROT", "WHEAT"):
            n = want.get(crop, 0) - int(W.seeds.get(crop, 0))
            if crop == "STRAWBERRY":
                n = min(n, 40)
            elif crop == "WHEAT":
                n = min(n, max_plant_today)
            if n <= 0:
                continue
            c = CROPS[crop]["seed"]
            reserve = cash_reserve if crop not in ("WHEAT", "CARROT") else min(cash_reserve, 120)
            k = int(max(0, (cash - reserve) // c))
            k = min(k, n)
            if k > 0:
                orders.append(["BUY_SEED", crop, k])
                cash -= k * c

    # ---------------- animals ----------------
    if not endgame and day <= SHEEP_LAST_BUY and hour <= 6 and len(orders) < 10:
        # how many pastures exist / can host
        empty_pastures = 0
        empty_animal_tiles = 0
        for (x, y), role in st.roles.items():
            tl = W.tiles[y][x]
            if is_empty_struct(tl) and tl.get("kind") == "PASTURE":
                empty_pastures += 1
            if role == "ANIMAL" and tl is None:
                empty_animal_tiles += 1
        capacity = empty_pastures + empty_animal_tiles
        unplaced = (n_cows - count_animals(W, "COW")) + (n_sheep - count_animals(W, "SHEEP"))
        capacity -= unplaced
        buy_c = max(0, st.cow_target - n_cows)
        buy_s = max(0, st.sheep_target - n_sheep)
        # buy limited by cash & capacity, prefer cows early
        order_list = []
        for kind, cnt in (("COW", buy_c), ("SHEEP", buy_s)):
            k = 0
            while k < cnt and capacity > 0:
                c = ANIMALS[kind]["cost"]
                if cash - c < cash_reserve:
                    break
                cash -= c
                capacity -= 1
                k += 1
            if k > 0:
                order_list.append(["BUY_ANIMAL", kind, k])
        orders.extend(order_list)

    # ---------------- feed wheat ----------------
    if not endgame and len(orders) < 10:
        wheat_now = int(shed.get("WHEAT", 0))
        # wheat carried by units will feed today; tomorrow needs n_animals
        need_today = 0
        for (x, y), role in st.roles.items():
            tl = W.tiles[y][x]
            if has_animal(tl) and not tl.get("fed_today", False):
                need_today += 1
        carried = sum(int(iv.get("WHEAT", 0)) for iv in W.invs)
        short = 0
        if hour <= 20:
            short = need_today - carried - wheat_now
        else:
            short = n_animals - wheat_now - carried
        if day >= 28:
            short = min(short, 0)
        if short > 0:
            pr = W.prices.get("WHEAT", 25) + 3
            k = int(min(short, max(0, (cash - 5) // pr)))
            if k > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", k])
                cash -= k * pr

    # ---------------- day 0 special ordering ----------------
    if W.step == 0:
        orders = [["HIRE"]] * OPEN_HIRES + [
            ["BUY_ANIMAL", "SHEEP", OPEN_SHEEP],
            ["BUY_ANIMAL", "COW", OPEN_COWS],
            ["BUY_SEED", "MELON", OPEN_MELONS],
            ["BUY_SEED", "WHEAT", OPEN_WHEAT_SEEDS],
            ["BUY_PRODUCT", "WHEAT", OPEN_WHEAT_FEED],
        ]
    return orders[:10]


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------
def act(obs):
    W = parse(obs)
    st = get_state(W.player, W.step)
    if st.plan_day != W.day:
        plan_day(st, W)
    else:
        # refresh roles when land unlocked mid-day
        if len(st.roles) != len(unlocked_tiles(W)):
            assign_roles(st, W)
    jobs = build_jobs(st, W)
    actions = assign(st, W, jobs)
    orders = market_orders(st, W, actions)
    farmer = actions[0] if actions else ["PASS"]
    hands = actions[1:len(W.units)]
    return {"farmer": farmer, "hands": hands, "market": orders}


def agent(obs, config=None):
    try:
        return act(obs)
    except Exception:
        try:
            farms = G(obs, "farms", [])
            p = int(G(obs, "player", 0))
            hands = G(farms[p], "hands", [])
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}
        except Exception:
            return {"farmer": ["PASS"], "hands": [], "market": []}

__version__ = "X594-ryo-live-opening"
