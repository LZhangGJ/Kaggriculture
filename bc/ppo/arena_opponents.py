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
    from tools.arena.sandbox import preflight
    preflight(pool['config'])
    return pool


def assign(jobs, pool, seed, iteration):
    """Replace exactly the final quarter, without touching game or seed IDs."""
    families = {}
    for row in pool['opponents']:
        families.setdefault(row['family'], []).append(row)
    names = sorted(families)
    result = []
    for original in jobs:
        row = dict(original)
        if row['family'] == 'history_slot':
            rng = random.Random(f'arena-v1:{seed}:{iteration}:{row["game"]}')
            family = rng.choice(names)
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
