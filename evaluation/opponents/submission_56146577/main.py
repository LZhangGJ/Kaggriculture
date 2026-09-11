"""Portable P16 JointAFS R1 Kaggriculture entry."""
from afs_runtime.policy import Agent as _Agent

_players = {}

def agent(observation, configuration=None):
    seat = int(observation.get('player', 0))
    if seat not in _players:
        _players[seat] = _Agent()
    return _players[seat](observation, configuration)
