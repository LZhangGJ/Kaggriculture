"""Read-only access to the merged, hash-pinned evaluation runtime."""
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = HERE.parent / 'evaluation_sets/2026-09-12-v1'
RUNTIME = BUNDLE / 'evaluation/runtime'
PACKAGE = RUNTIME / 'nt/latest_20260911_p16_jointafs_r1/agent'
sys.path.insert(0, str(PACKAGE / 'referee'))
from cpu_runtime import LocalGame, load_engine

ENGINE = load_engine()
ITEMS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER', 'GOOSE', 'COW', 'SHEEP')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def clean_obs(obs):
    return {k: copy.deepcopy(obs[k]) for k in ('player', 'step', 'day', 'hour', 'farms', 'private', 'market', 'town')}


def buffers(n):
    import numpy as np
    return dict(tiles=np.empty((n*2, 48, 10, 10), np.float32), glob=np.empty((n*2, 128), np.float32),
                features=np.empty((n*2*1024, 59), np.float32), owners=np.empty(n*2*1024, np.int64),
                bounds=np.empty(n*2*1024, np.int64), offsets=np.empty(n*2+1, np.int64), ids=np.empty(n*2, np.int64))


def public_seeds():
    reserved = set()
    for name in ('representative', 'stress', 'stress_pool'):
        reserved.update(json.loads((BUNDLE / 'manifests' / (name + '.json')).read_text())['seeds'])
    return reserved
