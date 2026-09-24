"""A06R12-T65-F100: rule-only daily portfolio MPC with discounted forecast tail.
Linux x86-64, Python 3, C++20 native runtime. No network or opponent files.
"""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_a06_native_codec', ROOT / 'policy' / 'agent.py')
codec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(codec)


def create_agent(binary_path=None):
    """Return an independent context. Caller should close() after a game."""
    config = json.loads((ROOT / 'policy' / 'config.json').read_text(encoding='utf-8'))
    binary = Path(binary_path) if binary_path is not None else ROOT / 'policy' / 'a06.so'
    if not binary.is_file():
        raise FileNotFoundError(f'Native runtime missing: {binary}; run python3 build.py')
    return codec.Agent(config=config, binary_path=binary)


_seats = {}


def reset():
    """Release local game contexts; step zero also resets the native policy."""
    for instance in _seats.values():
        instance.close()
    _seats.clear()


# Keep the explicitly intended submission entrypoint last.
def agent(observation, configuration=None):
    seat = int(codec._get(observation, 'player', 0))
    if seat not in (0, 1):
        raise ValueError('Expected player 0 or 1')
    if seat not in _seats:
        _seats[seat] = create_agent()
    return _seats[seat](observation, configuration)
