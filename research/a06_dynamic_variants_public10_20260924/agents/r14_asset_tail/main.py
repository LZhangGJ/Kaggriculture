"""A06 R14 AssetTail: held/funded asset critic with the R13 live executor."""
from pathlib import Path
import importlib.util
import json
import sys

# Kaggle's raw loader supplies an execution directory via sys.path but does
# not necessarily set __file__. Prefer the explicit module path when present.
_roots = []
if globals().get('__file__'):
    _roots.append(Path(__file__).resolve().parent)
_roots.extend(Path(p).resolve() for p in reversed(sys.path) if p)
_roots.append(Path('/kaggle_simulations/agent'))
ROOT = next((p for p in _roots if (p / 'policy' / 'agent.py').is_file()
             and (p / 'policy' / 'config.json').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('A06 policy directory not found beside the entry')
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

# Kaggle raw-source loader selects the last callable, not a function by name.
submission_agent = agent
