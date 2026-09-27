import os
"""Pinned real arena programs for synchronous, on-policy collection."""
import hashlib
import json
import random
import sys
import tempfile
import uuid
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_pool(path):
    pool = json.loads(Path(path).read_text())
    if pool['protocol'] != 'real-arena-training-v1' or not pool['opponents']:
        raise ValueError('Invalid arena training pool')
    if len({x['id'] for x in pool['opponents']}) != len(pool['opponents']):
        raise ValueError('Duplicate arena opponents')
    for row in pool['opponents']:
        if row['image'] != pool['config']['image']:
            raise ValueError('Arena image differs from validated opponent image')
        if file_sha(row['file']) != row['archive']:
            raise ValueError('Arena artifact changed: ' + row['id'])
    for path, expected in pool['runtime_files'].items():
        if file_sha(path) != expected:
            raise ValueError('Arena runtime changed: ' + path)
    sys.path.insert(0, pool['arena_repo'])
    if os.environ.get('PPO_ARENA_SANDBOX', 'docker') == 'chroot':  # chroot-v1 (source-v73): no docker on the host;
        from ppo.arena_chroot import check_runtime  # the exported runtime must be the pinned image digest
        check_runtime(pool['config'])
    else:
        from tools.arena.sandbox import preflight
        preflight(pool['config'])
    return pool


def assign(jobs, pool, seed, iteration, family_weights=None):
    """Replace every history_slot job with a real arena program, without touching game or seed IDs.

    family_weights: optional {family: weight}; families absent from the map get weight 1. None -> uniform."""
    families = {}
    for row in pool['opponents']:
        families.setdefault(row['family'], []).append(row)
    names = sorted(families)
    weights = [float(family_weights.get(n, 1.0)) for n in names] if family_weights else None
    if weights is not None and (min(weights) <= 0 or sum(weights) <= 0):
        raise ValueError('Arena family weights must be positive')
    result = []
    for original in jobs:
        row = dict(original)
        if row['family'] == 'history_slot':
            rng = random.Random(f'arena-v1:{seed}:{iteration}:{row["game"]}')
            family = rng.choices(names, weights=weights, k=1)[0] if weights is not None else rng.choice(names)
            opponent = rng.choice(sorted(families[family], key=lambda x: x['id']))
            seat = row['learner_seats'][0]
            row['policies'] = ['learner', 'learner']
            row['policies'][1-seat] = 'script:arena:' + opponent['id']
            row.update(family='arena', opponent_sha256=opponent['archive'],
                       arena_opponent=opponent, arena_config=pool['config'],
                       arena_repo=pool['arena_repo'], arena_family=family,
                       arena_runtime_files=pool['runtime_files'])
        result.append(row)
    return result


class SandboxOpponent:
    def __init__(self, assignment):
        self.name = assignment.get('arena_container_name') or 'ppo-practice-' + uuid.uuid4().hex
        self.proc = None
        self.manifest = None
        self.config = assignment['arena_config']
        sys.path.insert(0, assignment['arena_repo'])
        from tools.arena.worker import AgentProcess
        from tools.arena.sandbox import docker_args
        row = assignment['arena_opponent']
        if file_sha(row['file']) != row['archive']:
            raise ValueError('Arena archive changed before launch')
        for path, expected in assignment['arena_runtime_files'].items():
            if file_sha(path) != expected:
                raise ValueError('Arena runtime changed before launch')
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(row['manifest'], f)
                self.manifest = Path(f.name)
            self.manifest.chmod(0o644)
            command = docker_args(self.config, self.name,
                [(row['file'], '/input.zip'), (self.manifest, '/manifest.json')],
                ['python', '/opt/arena/inside.py', 'run', str(assignment['seed'])],
                row['manifest'].get('resources', {}).get('scratch_mb', 512))
            command.insert(2, '-i')
            self.proc = AgentProcess(command, self.config['turn_timeout_seconds'])
            self.proc.ready(120)
        except BaseException:
            self.close()
            raise

    def __call__(self, obs, configuration):
        return self.proc(obs, configuration)

    def close(self):
        from tools.arena.sandbox import cleanup
        try:
            if self.proc is not None:
                self.proc.close()
        finally:
            cleanup(self.name)
            if self.manifest is not None:
                self.manifest.unlink(missing_ok=True)


FLOOR = .5
EMA = .9


PFSP_FLOOR = .2


def family_weights_pfsp(ema_win):
    """pfsp-v1 (source-v77): weight = max(PFSP_FLOOR, (1 - p)^2), p = EMA win rate; focus multipliers; mean-normalised."""
    import os as _os
    names = sorted(ema_win)
    if not names:
        return {}
    focus = json.loads(_os.environ.get('PPO_FAMILY_FOCUS') or '{}')
    raw = {n: max(PFSP_FLOOR, (1. - float(ema_win[n])) ** 2) * float(focus.get(n, 1.)) for n in names}
    scale = len(names) / sum(raw.values())
    return {n: round(raw[n] * scale, 6) for n in names}


def family_weights_from_margins(ema):
    """weight = FLOOR + deficit/mean(deficit), normalised to mean 1; deficit = max(0, -EMA margin)."""
    names = sorted(ema)
    if not names:
        return {}
    deficit = {n: max(0., -float(ema[n])) for n in names}
    mean = sum(deficit.values()) / len(names)
    raw = {n: (FLOOR + deficit[n] / mean) if mean > 0 else 1. for n in names}
    import os as _os
    focus = json.loads(_os.environ.get('PPO_FAMILY_FOCUS') or '{}')  # family-focus-v1 (source-v75)
    raw = {n: raw[n] * float(focus.get(n, 1.)) for n in names}
    scale = len(names) / sum(raw.values())
    return {n: round(raw[n] * scale, 6) for n in names}


def update_family_weights(path, games, iteration):
    """Rank 0 only, after every update: fold this update's per-family mean margins into the EMA and write weights."""
    margins = {}
    wins = {}  # pfsp-v1
    for g in games:
        if g.get('family') != 'arena' or len(g.get('learner_seats', [])) != 1 or not g.get('cash') or any(g.get('faults', [])):
            continue
        seat = g['learner_seats'][0]
        margins.setdefault(g['arena_family'], []).append(g['cash'][seat] - g['cash'][1 - seat])
        d = g['cash'][seat] - g['cash'][1 - seat]; wins.setdefault(g['arena_family'], []).append(1. if d > 0 else (.5 if d == 0 else 0.))
    path = Path(path)
    previous = json.loads(path.read_text()) if path.exists() else {}
    ema = dict(previous.get('ema', {}))
    for family, rows in margins.items():
        mean = sum(rows) / len(rows)
        ema[family] = mean if family not in ema else EMA * ema[family] + (1 - EMA) * mean
    ema_win = dict(previous.get('ema_win', {}))  # pfsp-v1: tracked always, used when PPO_FAMILY_WEIGHTING=pfsp
    for family, rows in wins.items():
        m = sum(rows) / len(rows)
        ema_win[family] = m if family not in ema_win else EMA * ema_win[family] + (1 - EMA) * m
    import os as _os
    mode = _os.environ.get('PPO_FAMILY_WEIGHTING') or 'deficit'
    weights = family_weights_pfsp({k: ema_win.get(k, .5) for k in ema}) if mode == 'pfsp' else family_weights_from_margins(ema)
    record = dict(update=iteration, ema={k: round(v, 1) for k, v in ema.items()}, ema_win={k: round(v, 4) for k, v in ema_win.items()}, mode=mode, weights=weights,
                  observed={k: dict(games=len(v), mean_margin=round(sum(v) / len(v), 1)) for k, v in margins.items()},
                  rule=dict(floor=FLOOR, ema=EMA, formula='weight=(floor+deficit/mean(deficit))*focus, mean-normalised', focus=json.loads(__import__('os').environ.get('PPO_FAMILY_FOCUS') or '{}')))
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)
    return record
