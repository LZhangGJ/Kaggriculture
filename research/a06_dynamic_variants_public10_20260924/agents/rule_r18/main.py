"""A06 R12 derived rule/DP agent. Import API and raw-source Kaggle entry supported."""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else None
codec = None
_seats = {}

def _ensure_runtime(configuration=None):
    global ROOT, codec
    if codec is not None:
        return
    if ROOT is None:
        raw = configuration.get('__raw_path__') if isinstance(configuration, dict) else getattr(configuration, '__raw_path__', None)
        if raw:
            ROOT = Path(raw).resolve().parent
        else:
            ROOT = Path.cwd()
    path = ROOT / 'policy' / 'agent.py'
    if not path.is_file():
        raise FileNotFoundError(f'Keep main.py and policy/ together. Runtime not found: {path}')
    spec = importlib.util.spec_from_file_location('_a06_native_codec', path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)

# Normal imports initialize the codec immediately, retaining the offline API.
if ROOT is not None:
    _ensure_runtime()

def create_agent(binary_path=None):
    """One isolated game context; call close() to release its native memory."""
    _ensure_runtime()
    config = json.loads((ROOT / 'policy' / 'config.json').read_text(encoding='utf-8'))
    binary = Path(binary_path) if binary_path is not None else ROOT / 'policy' / 'a06.so'
    if not binary.is_file():
        raise FileNotFoundError(f'Native runtime missing: {binary}. Run python build.py first.')
    return codec.Agent(config=config, binary_path=binary)

def agent(observation, configuration=None):
    _ensure_runtime(configuration)
    seat = int(codec._get(observation, 'player', 0))
    if seat not in (0, 1):
        raise ValueError('Expected player 0 or 1')
    if seat not in _seats:
        _seats[seat] = create_agent()
    return _seats[seat](observation, configuration)

def reset():
    for instance in _seats.values():
        instance.close()
    _seats.clear()

# Must remain the last callable in namespace for Kaggle's raw-source loader.
def submission_entry(observation, configuration=None):
    return agent(observation, configuration)
