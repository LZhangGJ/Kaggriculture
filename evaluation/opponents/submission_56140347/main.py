"""Frozen P16 public-supply planner. Portable Kaggle entry."""
from p16_runtime.policy import Agent as _Agent

_players = {}

def agent(observation, configuration=None):
    seat = int(observation.get('player', 0))
    if seat not in _players:
        _players[seat] = _Agent()
    return _players[seat](observation, configuration)
