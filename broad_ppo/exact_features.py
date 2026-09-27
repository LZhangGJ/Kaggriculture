"""Observation-only features for exact worker and market behavior cloning.

No replay targets, synthetic plans, or opponent-private state are accepted.
The encoder retains the established V2 tensor shapes for a bounded warm start.
"""
import numpy as np
from features_v2 import encode as encode_v2, lifecycle
from bc_runtime import CROPS, ANIMALS, ITEMS
from kgrl.commitments.mechanics import nearest_shed

SCHEMA = 'exact-worker-market-features-v1'
EXACT_REGISTRY = {
    'farms.53:57': 'nominal_maintenance_action_load_estimate by horizon; not exact demand',
    'farms.94': 'nominal maintenance values are estimates',
    'workers.34:45': 'current Manhattan distance to opportunity class, divided by 18',
    'workers.45:56': 'opportunity class exists; zero distance is unknown when false',
    'workers.56:61': 'own resource compatibility flags; unknown for opponent',
}
OPPORTUNITIES = ('shed', 'harvest', 'water', 'feed', 'care', 'fertilizer_ready',
                 'fertilizable_crop', 'empty', 'diggable', 'empty_coop', 'empty_pasture')
MOVES = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0))


def opportunity_points(obs, player):
    """Public targets, with separate validity when a target class is absent."""
    points = [[] for _ in OPPORTUNITIES]
    # Movement is unobstructed in this engine, including through LOCKED cells.
    # Derive all access tiles from the pinned support function, not an assumed
    # corner/center shed layout.
    for y in range(10):
        for x in range(10):
            if tuple(nearest_shed((x, y))) == (x, y):
                points[0].append((x, y))
            tile = obs['farms'][player]['tiles'][y][x]
            flags = [False] * 10
            flags[6] = tile is None
            if isinstance(tile, dict):
                crop, animal = tile.get('crop'), tile.get('animal')
                left, _, _, known, _ = lifecycle(tile, obs['day'], obs['hour'])
                has_animal_slot = 'animal' in tile
                flags[0] = tile.get('yield_units', 0) > 0 and (animal in ANIMALS or known and left == 0)
                flags[1] = crop in CROPS and not tile.get('watered_today', False)
                flags[2] = animal in ANIMALS and not tile.get('fed_today', False)
                flags[3] = animal in ANIMALS and not tile.get('cared_today', False)
                flags[4] = animal in ANIMALS and bool(tile.get('fertilizer_available', False))
                flags[5] = crop in CROPS
                flags[7] = not has_animal_slot
                flags[8] = tile.get('kind') == 'COOP' and not has_animal_slot
                flags[9] = tile.get('kind') == 'PASTURE' and not has_animal_slot
            for index, yes in enumerate(flags, 1):
                if yes:
                    points[index].append((x, y))
    return [np.asarray(p, dtype=np.int16).reshape(-1, 2) for p in points]


def directional_opportunities(obs, player, position, points=None):
    """[here,N,S,E,W] x target-class x [distance, improvement, known].

    These are geometric features, not selected goals or action masks. An invalid
    move leaves position unchanged, matching the engine.
    """
    positions = []
    for dx, dy in MOVES:
        x, y = position[0] + dx, position[1] + dy
        positions.append((x, y) if 0 <= x < 10 and 0 <= y < 10 else tuple(position))
    positions = np.asarray(positions)
    result = np.zeros((5, len(OPPORTUNITIES), 3), np.float32)
    if points is None:
        points = opportunity_points(obs, player)
    for i, targets in enumerate(points):
        if len(targets):
            distance = np.abs(positions[:, None] - targets[None]).sum(-1).min(-1)
            result[:, i, 0] = distance / 18
            result[:, i, 1] = (distance[0] - distance) / 18
            result[:, i, 2] = 1
    return result


def encode_exact(obs, history):
    x = encode_v2(obs, [], history)
    # An absent queue is not an empty queue with free capacity.
    x['programs'].fill(0)
    x['program_valid'].fill(False)
    x['boards'][:, 25:30] = 0
    x['global_features'][7] = 0
    x['farms'][:, 6] = 0
    x['farms'][:, 61:65] = 0
    x['farms'][:, 77:89] = 0
    x['farms'][:, 94] = 1
    x['workers'][:, 6:8] = 0
    x['workers'][:, 22:33] = 0
    # Reuse previously empty worker columns for causal navigation context.
    index = 0
    for player in (obs['player'], 1 - obs['player']):
        farm = obs['farms'][player]
        points = opportunity_points(obs, player)
        for worker, position in enumerate([farm['farmer'], *farm['hands']]):
            near = directional_opportunities(obs, player, position, points)[0]
            x['workers'][index, 34:45] = near[:, 0]
            x['workers'][index, 45:56] = near[:, 2]
            if player == obs['player']:
                inv = obs['private']['inventories'][worker]
                x['workers'][index, 56:61] = [
                    any(obs['private']['seeds'].get(c, 0) > 0 for c in CROPS),
                    inv.get('WHEAT', 0) > 0, inv.get('FERTILIZER', 0) > 0,
                    any(inv.get(item, 0) > 0 for item in ITEMS[:9]),
                    any(inv.get(item, 0) > 0 for item in ANIMALS)]
            index += 1
    return x
