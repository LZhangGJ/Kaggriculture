"""Reuse the existing official replay/action/work/cost audit without changing it."""
import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[2]
EVIDENCE = WORKSPACE / 'experiments/r2p16_route_economics_20260910'
BASELINE = EVIDENCE / 'baseline'
PUBLIC = WORKSPACE / 'experiments/r2_route_new_opponents_20260910/source/nt/latest_20260910_r2p16/opponents'
HARNESS = WORKSPACE / 'experiments/r2_route_three_arm_20260910/battle.py'

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def run(job):
    arm, opponent, seed, seat, tag, binary_dir = job
    sys.path.insert(0, str(BASELINE / 'referee'))
    battle = module(HARNESS, 'route_audit_harness')
    arena = module(BASELINE / 'run.py', 'p16_public_loader')
    runtime = module(PACKAGE / 'main.py', 'route_candidate_entry')
    battle.ROOT = EVIDENCE / 'runs' / tag
    battle.ARMS = {'P16':0, 'guards':1, 'delivery':2, 'economic':3}
    public = None
    def new_agent(mode):
        binary = BASELINE / 'policy/startupsupply2.so' if mode == 0 else Path(binary_dir) / f'route{mode}.so'
        agent = runtime.create_agent(binary)
        if mode:
            agent.lib.td_route_required.argtypes = [ctypes.c_void_p]
            agent.lib.td_route_required.restype = ctypes.c_char_p
        return agent
    def info(agent, mode):
        if not mode:
            return {}
        data = agent.debug()
        data['hire_changes'] = data['route_hire_changes'] + data['route_changes']
        return data
    def opponent_agent(path):
        nonlocal public
        public = arena.FreshPublic(PUBLIC / opponent)
        return public.call
    battle.new_agent = new_agent
    battle.route_info = info
    battle.load_agent = opponent_agent
    try:
        # Every match gets an uninstrumented full replay check as well.
        row = battle.run_one((arm, opponent, seed, seat, True))
        row['route_commitment_days'] = row.pop('reduced_hire_days')
        path = battle.ROOT / 'smoke' / (row['name'] + '.result.json')
        path.write_text(json.dumps(row, indent=2)+'\n')
        return row
    finally:
        if public is not None:
            public.close()

def parity(binary):
    runtime = module(PACKAGE / 'main.py', 'route_parity_entry')
    frozen = runtime.create_agent(BASELINE / 'policy/startupsupply2.so')
    rebuilt = runtime.create_agent(binary)
    # A policy with persistent plans must see its own on-policy trajectory.
    # Feeding P16 an old R2 trajectory can change staff counts underneath it.
    replay = EVIDENCE / 'runs/pilot/smoke/P16_moon_v215_2026091021_seat0.json.gz'
    frames = json.loads(gzip.decompress(replay.read_bytes()))['steps']
    for t in range(719):
        obs = frames[t][0]['observation']
        reference = frozen(obs)
        assert reference == frames[t+1][0]['action'], f'Frozen P16 reset mismatch at {t}'
        assert reference == rebuilt(obs), f'P16 mode-zero mismatch at {t}'
    frozen.close(); rebuilt.close()
    result = dict(status='PASS', observations=719, reference='frozen P16 binary, same replay observations',
                  replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
                  rebuilt_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest())
    (EVIDENCE / 'parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='pilot')
    p.add_argument('--binary-dir', type=Path, default=PACKAGE / 'build/revision5')
    p.add_argument('--arms', default='P16,guards,delivery,economic')
    p.add_argument('--opponents', default='moon_v215,market_smart_v8,thomas_955_v2')
    p.add_argument('--seeds', default='2026091021,2026091026')
    p.add_argument('--seats', default='0,1')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--parity', type=Path)
    a = p.parse_args()
    if a.parity:
        parity(a.parity); return
    assert Path(a.tag).name == a.tag and a.tag not in {'.','..'}
    out = EVIDENCE / 'runs' / a.tag
    assert not out.exists(), f'Refusing overwrite: {out}'
    out.mkdir(parents=True)
    jobs = [(arm, op, int(seed), int(seat), a.tag, str(a.binary_dir.resolve()))
            for seed in a.seeds.split(',') for op in a.opponents.split(',')
            for seat in a.seats.split(',') for arm in a.arms.split(',')]
    files = [HARNESS, BASELINE / 'policy/startupsupply2.so', PACKAGE / 'policy/config.json', Path(__file__)]
    modes = {'guards':1, 'delivery':2, 'economic':3}
    for arm in a.arms.split(','):
        if arm != 'P16':
            binary = a.binary_dir / f'route{modes[arm]}.so'
            receipt = json.loads(binary.with_suffix('.BUILD.json').read_text())
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == receipt['sha256']
            files.extend([binary, binary.with_suffix('.BUILD.json')])
    (out / 'PROTOCOL.json').write_text(json.dumps(dict(jobs=jobs, hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}), indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            r = f.result(); rows.append(r)
            print(json.dumps({k:r[k] for k in ('name','cash','margin','wages','overflow_units','invalid_actions','uncompleted_required_tasks','max_seconds','error')}), flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    assert all(r['error'] is None and r['frames'] == 720 and r['uninstrumented_replay_check'] == 'PASS' for r in rows)

if __name__ == '__main__':
    main()
