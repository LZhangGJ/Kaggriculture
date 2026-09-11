"""Frozen P16 versus the unchanged route candidate across P16's complete public pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PACKAGE = WORKSPACE / 'Kaggriculture/nt/latest_20260910_r2p16_route_economics'
BASELINE = WORKSPACE / 'experiments/r2p16_route_economics_20260910/baseline'
SOURCE = WORKSPACE / 'experiments/r2_route_new_opponents_20260910/source/nt/latest_20260910_r2p16'
BINARY = PACKAGE / 'build/revision5/route3.so'
PANEL = PACKAGE / 'tests/run_panel.py'
SEEDS = list(range(2609168000, 2609168020))
ARMS = ['P16', 'economic']
TAG = 'pool20'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def freeze(workers):
    roster = read(SOURCE / 'POOL.json')
    names = [r['id'] for r in roster]
    assert len(names) == len(set(names)) == 11
    manifest = read(SOURCE / 'MANIFEST_SHA256.json')
    paths = [Path(__file__), PANEL, SOURCE / 'POOL.json', SOURCE / 'OPPONENTS_ZH.md',
             BASELINE / 'run.py', BASELINE / 'policy/startupsupply2.so',
             PACKAGE / 'main.py', BINARY, BINARY.with_suffix('.BUILD.json'),
             WORKSPACE / 'experiments/r2_route_three_arm_20260910/battle.py',
             WORKSPACE / 'Kaggriculture/nt/latest_20260909_t2_r1_r2/public/R2/t2r2_runtime/policy.py']
    for name, sha in read(PACKAGE / 'SOURCE_PROVENANCE.json')['baseline_sha256'].items():
        assert digest(BASELINE / name) == sha, name
    receipt = read(BINARY.with_suffix('.BUILD.json'))
    assert digest(BINARY) == receipt['sha256']
    assert digest(BASELINE / 'policy/startupsupply2.so') == '4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b'
    for name, sha in receipt['sources'].items():
        path = PACKAGE / name
        assert digest(path) == sha, name
        paths.append(path)
    for name in names:
        assert (SOURCE / 'opponents' / name / 'main.py').is_file()
        for path in (SOURCE / 'opponents' / name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                assert digest(path) == manifest[path.relative_to(SOURCE).as_posix()]['sha256'], path
                paths.append(path)
    paths.extend(p for p in (BASELINE / 'referee').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    # Only serialization changes: prove byte-equivalent JSON on a full replay.
    prior = WORKSPACE / 'experiments/r2p16_route_economics_20260910/runs/fresh5_r5/smoke/P16_moon_v215_2609126001_seat0.json.gz'
    replay = json.loads(gzip.decompress(prior.read_bytes()))
    stream = io.StringIO(); json.dump(replay, stream, separators=(',', ':'))
    assert stream.getvalue() == json.dumps(replay, separators=(',', ':'))
    jobs = [(arm, opponent, seed, seat) for seed in SEEDS for opponent in names for seat in (0,1) for arm in ARMS]
    protocol = dict(pool='P16 frozen public-original 11-agent POOL.json, not the historical reconstructed 68-agent pool',
                    seeds=SEEDS, opponents=names, arms=ARMS, jobs=jobs, games=len(jobs), workers=workers,
                    official_version='1.32.7', realtime_opponents=True, full_replay_recheck=True,
                    serialization_parity='PASS', hidden_seed_not_given_to_agents=True,
                    hashes={str(p.relative_to(WORKSPACE)):digest(p) for p in sorted(set(paths))})
    directory = ROOT / 'runs' / TAG; directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'PROTOCOL.json'
    if path.exists():
        assert read(path) == protocol, 'Frozen inputs changed; retain this run and use a new protocol.'
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    return jobs


def run(job):
    # Preload the P16 referee explicitly: the old shared audit inserts its own
    # historical paths, which must not select a different cpu_runtime module.
    module(BASELINE / 'referee/cpu_runtime.py', 'cpu_runtime')
    panel = module(PANEL, 'frozen_route_panel')
    panel.EVIDENCE = ROOT
    original_module = panel.module
    other_times = []
    def load(path, name):
        loaded = original_module(path, name)
        if path == panel.HARNESS:
            assert loaded.load_engine.__globals__['ROOT'] == BASELINE / 'referee'
            loaded.json = SimpleNamespace(loads=json.loads, dumps=json.dumps,
                dump=lambda obj, stream, **kw: stream.write(json.dumps(obj, **kw)))
        if path == BASELINE / 'run.py':
            old = loaded.FreshPublic
            class TimedPublic(old):
                def __init__(self, directory):
                    super().__init__(directory)
                    policy = self.call
                    def call(obs, cfg):
                        start = time.perf_counter()
                        try:
                            return policy(obs, cfg)
                        finally:
                            other_times.append(time.perf_counter()-start)
                    self.call = call
            loaded.FreshPublic = TimedPublic
        return loaded
    panel.module = load
    row = panel.run((*job, TAG, str(BINARY.parent)))
    row.update(opponent_decisions=len(other_times), opponent_max_seconds=max(other_times, default=0),
               opponent_over_one_second=sum(t>1 for t in other_times))
    directory = ROOT / 'runs' / TAG / 'smoke'
    row['replay_sha256'] = digest(directory / (row['name']+'.json.gz'))
    row['audit_sha256'] = digest(directory / (row['name']+'.audit.json.gz'))
    path = directory / (row['name']+'.result.json')
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(row,indent=2)+'\n'); temp.replace(path)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    jobs = freeze(args.workers)
    rows, pending = [], []
    directory = ROOT / 'runs' / TAG
    for job in jobs:
        arm, opponent, seed, seat = job
        path = directory / 'smoke' / f'{arm}_{opponent}_{seed}_seat{seat}.result.json'
        if path.exists():
            row = read(path)
            # Keep failed games in the panel; never silently retry for a win.
            assert 'replay_sha256' in row and 'audit_sha256' in row, 'Incomplete receipt'
            rows.append(row)
        else:
            pending.append(job)
    print(json.dumps(dict(planned=len(jobs), completed=len(rows), pending=len(pending))), flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(run, job) for job in pending]):
            row = future.result(); rows.append(row)
            if len(rows)%11==0 or row['error'] or row['uncompleted_required_tasks'] or row['invalid_actions']:
                print(json.dumps(dict(completed=len(rows), elapsed_seconds=time.perf_counter()-start,
                      **{k:row[k] for k in ('name','cash','margin','wages','invalid_actions','uncompleted_required_tasks','error')})), flush=True)
    rows.sort(key=lambda r:(r['seed'],r['opponent'],r['seat'],r['arm']))
    (directory / 'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows),errors=sum(bool(r['error']) for r in rows),seconds=time.perf_counter()-start)),flush=True)


if __name__ == '__main__':
    main()
