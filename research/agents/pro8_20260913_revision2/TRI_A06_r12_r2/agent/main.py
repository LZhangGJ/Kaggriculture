"""TRI_A06_r12_r2: calendar-consistent terminal planting; awaiting central win-rate acceptance."""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_a06_native_codec', ROOT / 'policy' / 'agent.py')
codec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(codec)


def create_agent(binary_path=None):
    """Create an independent game context. Explicit close() is supported."""
    config = json.loads((ROOT / 'policy' / 'config.json').read_text(encoding='utf-8'))
    binary = Path(binary_path) if binary_path is not None else ROOT / 'policy' / 'a06.so'
    if not binary.is_file():
        raise FileNotFoundError(f'Native runtime missing: {binary}. Run python build.py first.')
    return codec.Agent(config=config, binary_path=binary)


_seats = {}

def agent(observation, configuration=None):
    """Kaggle entry. Each seat has its own context; a new step-0 resets that context."""
    seat = int(codec._get(observation, 'player', 0))
    if seat not in (0, 1):
        raise ValueError('Expected player 0 or 1')
    if seat not in _seats:
        _seats[seat] = create_agent()
    return _seats[seat](observation, configuration)


def reset():
    """Release all native contexts, e.g. between local tournaments."""
    for instance in _seats.values():
        instance.close()
    _seats.clear()
