"""Paired live matches using the already compiled P16 simulator."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, ctypes, hashlib, json, multiprocessing, sys, time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT.parent / 'r2p16_vs_route_agent_pool_20260910'))
import native_pool as host
import pool_test
from pool_test import read, digest

PACKAGES = {arm: WORKSPACE / ('Kaggriculture/nt/latest_20260910_r2p16_route_' + arm)
            for arm in ('workflow', 'recovery')}
PACKAGES['fixed'] = WORKSPACE / 'Kaggriculture/nt/latest_20260911_r2p16_route_repair'
BINARIES = {arm: p / f'build/revision{2 if arm == "workflow" else 3 if arm == "recovery" else 4}/route3.so'
            for arm, p in PACKAGES.items()}
LATEST = ['submission_56146577', 'submission_56149565']
HISTORICAL = [r['id'] for r in read(pool_test.SOURCE / 'POOL.json')] + ['submission_56140347', 'submission_56140351']
SEEDS = [int.from_bytes(hashlib.sha256(f'route-repair-20260911-holdout-{i}'.encode()).digest()[:4], 'big') for i in range(100)]
REGRESSION = [('submission_56146577', s, seat) for s in [1508212750, 347212678, 2659900475, 985149698] for seat in (0, 1)]
REGRESSION += [('submission_56140347', s, seat) for s in [407296613, 572753638, 3201375470] for seat in (0, 1)]

def source(op):
    if (ROOT / 'opponents' / op / 'main.py').exists(): return ROOT
    if op in LATEST:
        return ROOT if (ROOT / 'opponents' / op / 'main.py').exists() else ROOT.parent / 'r2p16_jointafs_diagnosis_20260911'
    if op.startswith('submission_'):
        return ROOT.parent / 'r2p16_latest_submissions_20260910'
    return pool_test.SOURCE

def run(item):
    job, tag, binary = item
    host.OUT = ROOT / 'runs' / tag
    host.PACKAGE = PACKAGES[job[0]]
    host.BINARY = Path(binary) if job[0] == 'fixed' and binary else BINARIES[job[0]]
    host.SOURCE = source(job[1])
    original_module = host.module
    def load(path, name):
        loaded = original_module(path, name)
        if path == host.PACKAGE / 'main.py':
            original_create = loaded.create_agent
            def create(*args, **kwargs):
                agent = original_create(*args, **kwargs)
                old_debug = agent.debug
                has_deferred = hasattr(agent.lib, 'td_route_deferred')
                if has_deferred:
                    agent.lib.td_route_deferred.argtypes = [ctypes.c_void_p]
                    agent.lib.td_route_deferred.restype = ctypes.c_char_p
                def debug():
                    info = old_debug()
                    info['deferred_tasks'] = json.loads(agent.lib.td_route_deferred(agent.handle)) if has_deferred else []
                    return info
                agent.debug = debug
                return agent
            loaded.create_agent = create
        return loaded
    host.module = load
    return host.run(job)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=['regression', 'historical', 'latest'], default='regression')
    p.add_argument('--arms', nargs='+', default=['workflow', 'recovery', 'fixed'], choices=PACKAGES)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--binary', type=Path)
    p.add_argument('--opponents', nargs='+')
    a = p.parse_args()
    binary = str(a.binary.resolve()) if a.binary else None
    pairs = REGRESSION if a.stage == 'regression' else [(op, s, seat) for s in SEEDS[:25 if a.stage == 'historical' else 100]
             for op in (HISTORICAL if a.stage == 'historical' else LATEST) for seat in (0, 1)]
    if a.opponents:
        assert set(a.opponents)<=set(op for op,_,_ in pairs)
        pairs=[p for p in pairs if p[0] in a.opponents]
    jobs = [(arm, op, seed, seat) for op, seed, seat in pairs for arm in a.arms]
    paths = [Path(__file__), Path(host.__file__), Path(pool_test.__file__),
             Path(host.__file__).parent / '_pool_sim.cpython-312-x86_64-linux-gnu.so',
             Path(host.__file__).parent / 'NATIVE_CALIBRATION.json',
             ROOT / 'SUBMISSIONS.json']
    for op in {j[1] for j in jobs}:
        folder = source(op) / 'opponents' / op
        assert (folder / 'main.py').exists(), ('Missing original opponent', op)
        paths.extend(f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    for arm in a.arms:
        b = Path(binary) if arm == 'fixed' and binary else BINARIES[arm]
        paths += [b, PACKAGES[arm] / 'main.py']
        paths.extend(f for f in (PACKAGES[arm] / 'policy').rglob('*') if f.is_file() and f.suffix in ['.py', '.json', '.hpp', '.cpp', '.inc'])
        if a.stage != 'regression':
            receipt = read(b.with_suffix('.BUILD.json'))
            assert digest(b) == receipt['sha256']
            for name, sha in receipt['sources'].items(): assert digest(PACKAGES[arm] / name) == sha, name
    protocol = dict(jobs=jobs, stage=a.stage, arms=a.arms, all_fresh_seeds=SEEDS,
                    hashes={str(p.relative_to(WORKSPACE)): digest(p) for p in sorted(set(paths))},
                    seed_hidden=True, simulator='Existing compiled P16 simulator, not rebuilt',
                    timeout='Runtime measured; this host does not adjudicate timeout forfeits',
                    reporting='Development regressions excluded. Historical pool: 25 fresh seeds per opponent; latest two: 100 fresh seeds each. Both seats and equal arms. Report panels separately.')
    protocol = json.loads(json.dumps(protocol))
    out = ROOT / 'runs' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'PROTOCOL.json'
    if path.exists(): assert read(path) == protocol, 'Frozen inputs changed; use a new tag'
    else: path.write_text(json.dumps(protocol, indent=2) + '\n')
    rows, pending = [], []
    for job in jobs:
        arm, op, seed, seat = job
        receipt = out / 'matches' / f'{arm}_{op}_{seed}_seat{seat}.result.json'
        if receipt.exists():
            row = read(receipt)
            for suffix, key in [('.json.gz', 'replay_sha256'), ('.audit.json.gz', 'audit_sha256')]:
                assert digest(receipt.parent / (row['name'] + suffix)) == row[key]
            rows.append(row)
        else: pending.append((job, a.tag, binary))
    print(json.dumps(dict(planned=len(jobs), pending=len(pending))), flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(run, item) for item in pending]):
            row = f.result(); rows.append(row)
            print(json.dumps(dict(completed=len(rows), seconds=round(time.perf_counter()-start),
                  **{k: row[k] for k in ['name', 'cash', 'margin', 'uncompleted_required_tasks', 'invalid_actions', 'error']})), flush=True)
    for name, sha in protocol['hashes'].items(): assert digest(WORKSPACE / name) == sha, name
    (out / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')

if __name__ == '__main__': main()
