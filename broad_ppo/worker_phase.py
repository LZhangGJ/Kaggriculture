"""Pinned worker-phase resolver, including whole-turn planting cancellation.

A partial action list resolves the stated prefix with remaining workers passing.
It is a counterfactual prefix state, not the final state of an unfinished action.
Always restart from the turn observation: later PLANT requests can cancel earlier
requests. No market orders, decay, or end-of-day updates run here.
"""
from copy import deepcopy
from functools import cache
from importlib import import_module
from importlib.metadata import version
from cache_identity import engine_identity


@cache
def engine():
    if version('kaggle-environments') != '1.32.7':
        raise RuntimeError('Worker resolver requires kaggle-environments 1.32.7')
    engine_identity()  # Existing engine/spec SHA guard; once per process.
    return import_module('kaggle_environments.envs.kaggriculture.kaggriculture')


def command_slots(action):
    action = action if isinstance(action, dict) else {}
    hands = action.get('hands', [])
    hands = hands if isinstance(hands, list) else []
    return [action.get('farmer', ['PASS']), *hands]


def resolve_worker_phase(observation, slots):
    """Resolve supplied raw slots exactly; extra hands still affect PLANT demand."""
    official = engine()
    obs = dict(observation)
    obs['farms'] = list(observation['farms'])
    own = observation['player']
    farm = obs['farms'][own] = deepcopy(observation['farms'][own])
    private = obs['private'] = deepcopy(observation['private'])
    demand = {}
    for action in slots:
        if isinstance(action, list) and len(action) >= 2 and action[0] == 'PLANT':
            demand[action[1]] = demand.get(action[1], 0) + 1
    blocked = {crop for crop, amount in demand.items()
               if amount > private['seeds'].get(crop, 0)}
    for index, action in enumerate(slots):
        if (isinstance(action, list) and len(action) >= 2
                and action[0] == 'PLANT' and action[1] in blocked):
            action = ['PASS']
        official._apply_unit_action(farm, private, index, action, 10,
                                   observation['day'], 24, 100)
    return obs, demand, blocked
