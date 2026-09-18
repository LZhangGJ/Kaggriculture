"""Exact own-worker-phase projection, not an environment or market simulator.

Rules transcribed from the supplied 1.32.7 interpreter's _apply_unit_action.
No daily refresh, opponent action, hidden state, or settlement prediction here.
"""
from __future__ import annotations
import copy
from kgrl.contracts import (API_VERSION, DEFAULT_CONFIG, ENGINE_SHA256, SPEC_SHA256,
                            Command, ContractError, UnsupportedConfiguration)

# seed cost, first harvest, latest growth day, maximum yield, ongoing
CROPS = {'WHEAT': (10, 2, 4, 6, False), 'CARROT': (20, 2, 3, 4, False),
         'TOMATO': (50, 8, 8, 4, True), 'STRAWBERRY': (100, 10, 10, 4, True),
         'MELON': (80, 10, 12, 6, False)}
# purchase cost, structure, first production, product
ANIMALS = {'GOOSE': (300, 'COOP', 4, 'EGG'), 'COW': (400, 'PASTURE', 8, 'MILK'),
           'SHEEP': (500, 'PASTURE', 6, 'WOOL')}
MOVES = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'EAST': (1, 0), 'WEST': (-1, 0)}
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
LAND_COST = (1000, 2000, 4000)


def check_rules(rules):
    if (rules.api_version != API_VERSION or rules.engine_sha256 != ENGINE_SHA256
            or rules.specification_sha256 != SPEC_SHA256 or rules.package_version != '1.32.7'):
        raise ContractError('W03 requires the frozen kgrl-v1 engine/spec/package identity')
    cfg = dict(DEFAULT_CONFIG)
    if set(rules.configuration) - set(cfg):
        raise UnsupportedConfiguration('W03 configuration contains unknown fields or a seed')
    cfg.update(rules.configuration)
    if cfg != DEFAULT_CONFIG:
        raise UnsupportedConfiguration('W03 supports the declared default game configuration only')


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed(pos):
    return min(SHED, key=lambda t: distance(pos, t))


def move(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if not (dx or dy):
        return Command('PASS')
    return Command('EAST' if dx > 0 else 'WEST' if dx < 0 else 'SOUTH' if dy > 0 else 'NORTH')


def take(inv, item, n=1):
    if inv.get(item, 0) < n:
        return False
    inv[item] -= n
    if not inv[item]:
        del inv[item]
    return True


class UnitLedger:
    """A private copy with the official worker execution order and insertion order."""
    def __init__(self, observation):
        self.farm = copy.deepcopy(observation['farms'][observation['player']])
        self.private = copy.deepcopy(observation['private'])
        self.day = int(observation['day'])
        self.turn = int(observation['step'])
        self.positions = [tuple(self.farm['farmer']), *map(tuple, self.farm['hands'])]
        if len(self.private['inventories']) != len(self.positions):
            raise ContractError('worker/inventory cardinality mismatch')
        if len(self.farm['tiles']) != 10 or any(len(r) != 10 for r in self.farm['tiles']):
            raise UnsupportedConfiguration('W03 requires a 10x10 board')

    @property
    def shed(self):
        return self.private['shed']

    @property
    def seeds(self):
        return self.private['seeds']

    def inventory(self, worker):
        return self.private['inventories'][worker]

    def tile(self, target):
        return self.farm['tiles'][target[1]][target[0]]

    def room(self):
        return max(0, 100 - sum(self.shed.values()))

    def apply(self, worker, command):
        """Apply one command. Return whether it changed own state; PASS is False."""
        if not 0 <= worker < len(self.positions):
            raise ContractError('new hires may not act in the current worker phase')
        op, item = command.op, command.item
        n = 1 if command.quantity is None else command.quantity
        if type(n) is not int or n <= 0:
            raise ContractError('nonpositive/noninteger command quantity')
        x, y = self.positions[worker]
        inv = self.inventory(worker)
        tiles = self.farm['tiles']
        tile = tiles[y][x]
        if op in MOVES:
            dx, dy = MOVES[op]
            p = (x + dx, y + dy)
            if not (0 <= p[0] < 10 and 0 <= p[1] < 10):
                return False
            self.positions[worker] = p
            if worker:
                self.farm['hands'][worker - 1] = list(p)
            else:
                self.farm['farmer'] = list(p)
            return True
        if op == 'PASS':
            return False
        adjacent = (x, y) in SHED
        if op == 'DROP':
            if not adjacent or not inv:
                return False
            for key, count in list(inv.items()):
                got = min(max(count, 0), self.room())
                if got:
                    self.shed[key] = self.shed.get(key, 0) + got
                del inv[key]
            return True
        if op == 'PICKUP':
            got = min(n, self.shed.get(item, 0)) if adjacent else 0
            if got <= 0:
                return False
            self.shed[item] -= got
            inv[item] = inv.get(item, 0) + got
            return True
        if op == 'PLACE':
            if (item in ANIMALS and isinstance(tile, dict)
                    and tile.get('kind') == ANIMALS[item][1] and 'animal' not in tile):
                if not take(inv, item):
                    return False
                tiles[y][x] = dict(kind=ANIMALS[item][1], animal=item, placed_day=self.day,
                                  yield_units=0, consecutive_unfed=0, fed_today=False,
                                  cared_today=False, fertilizer_available=False, pending_care_bonus=0)
                return True
            got = min(n, inv.get(item, 0), self.room()) if adjacent else 0
            if got <= 0:
                return False
            take(inv, item, got)
            self.shed[item] = self.shed.get(item, 0) + got
            return True
        if tile == 'LOCKED':
            return False
        if op == 'PLANT':
            if tile is not None or item not in CROPS or self.seeds.get(item, 0) <= 0:
                return False
            _, _, latest, _, ongoing = CROPS[item]
            self.seeds[item] -= 1
            tiles[y][x] = dict(kind='PLANT', crop=item, planted_day=self.day,
                              watered_today=False, consecutive_unwatered=1,
                              yield_units=0 if ongoing else 1,
                              max_lifespan_step=-1 if ongoing else (self.day + latest + 1) * 24,
                              fertilized_until_day=-1)
            return True
        if op in ('BUILD_COOP', 'BUILD_PASTURE'):
            if tile is not None:
                return False
            tiles[y][x] = {'kind': op[6:]}
            return True
        if op == 'DIG':
            if tile is None or (isinstance(tile, dict) and 'animal' in tile):
                return False
            tiles[y][x] = None
            return True
        if not isinstance(tile, dict):
            return False
        plant = tile.get('kind') == 'PLANT'
        animal = 'animal' in tile
        if op == 'WATER' and plant and not tile['watered_today']:
            tile['watered_today'] = True
            _, _, latest, maximum, ongoing = CROPS[tile['crop']]
            if not ongoing and (latest + 1) // 2 <= self.day - tile['planted_day'] <= latest:
                tile['yield_units'] = min(maximum, tile['yield_units'] +
                                          (2 if tile['fertilized_until_day'] >= self.day else 1))
            return True
        if op == 'HARVEST' and tile.get('yield_units', 0) > 0:
            if plant:
                crop = tile['crop']
                if self.day - tile['planted_day'] < CROPS[crop][1]:
                    return False
                product = crop
            elif animal:
                product = ANIMALS[tile['animal']][3]
            else:
                return False
            inv[product] = inv.get(product, 0) + tile['yield_units']
            tile['yield_units'] = 0
            if plant and not CROPS[product][4]:
                tiles[y][x] = None
            return True
        if op == 'FERTILIZE' and plant and take(inv, 'FERTILIZER'):
            tile['fertilized_until_day'] = max(tile.get('fertilized_until_day', -1), self.day + 2)
            return True
        if op == 'FEED' and animal and not tile['fed_today'] and take(inv, 'WHEAT'):
            tile['fed_today'] = True
            return True
        if op == 'CARE' and animal and not tile['cared_today']:
            tile['cared_today'] = True
            return True
        if op == 'COLLECT_FERTILIZER' and animal and tile['fertilizer_available']:
            tile['fertilizer_available'] = False
            inv['FERTILIZER'] = inv.get('FERTILIZER', 0) + 1
            return True
        return False
