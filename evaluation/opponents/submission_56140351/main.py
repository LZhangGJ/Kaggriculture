"""P16 with T3 execution-aligned repairs. Portable Kaggle entry."""
from t3r1_runtime.policy import Agent as _Agent

_players = {}

def agent(observation, configuration=None):
    seat = int(observation.get('player', 0))
    if seat not in _players:
        _players[seat] = _Agent()
    return _players[seat](observation, configuration)
