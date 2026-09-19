"""Fresh teacher games, encoded during play; no evaluation replay inspection."""
import concurrent.futures as cf
import contextlib
import copy
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('F:/Kaggriculture/experiments/local_teacher_bc_20260917')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    temp.replace(path)


def play(job):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / 'frozen_code'))
    sys.path.insert(0, str(SOURCE / 'frozen/pool'))
    from run_matches import Policy
    from cpu_runtime import LocalGame, load_engine
    from student import Codec, observation_batch
    from numeric_features import exact_numbers
    from features import encode_action
    seed, opponent, seat = job
    folder = ROOT / 'data/shards' / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DONE.json').exists():
        done = json.loads((folder / 'DONE.json').read_text(encoding='utf8'))
        assert done['job'] == list(job) and digest(folder / 'samples.npz') == done['sha256']
        return done
    # A crashed game is never replayed with its already-started seed.
    with (folder / 'STARTED.json').open('x', encoding='utf8') as stream:
        json.dump(dict(job=job, pid=os.getpid(), time_unix=time.time()), stream)
    plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
    codec = Codec(plan['quantities'])
    previous_u = np.zeros((32, 2), np.int16)
    previous_m = np.zeros((11, 2), np.int16)
    samples = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        teacher = Policy(SOURCE / 'frozen/teacher/main.py', f'teacher_{seed}')
        rival = Policy(SOURCE / f'frozen/pool/opponents/{opponent}/main.py', f'rival_{seed}')
        game = LocalGame(seed, load_engine())
        while not game.done:
            obs = game.observation(seat)
            assert game.configuration.seed is None and 'seed' not in obs
            frame = observation_batch(obs, previous_u, previous_m)
            frame['global_exact'], frame['unit_exact'] = exact_numbers(obs)
            action = teacher(copy.deepcopy(obs), copy.deepcopy(game.configuration))
            tokens, canonical, _ = encode_action(action, int(frame['unit_count']), obs['private']['seeds'])
            u, m = codec.encode(tokens, int(frame['unit_count']))
            assert codec.decode(u, m, int(frame['unit_count'])) == canonical
            frame.update(unit_y=u, market_y=m)
            samples.append(frame)
            pair = [None, None]
            pair[seat] = action
            pair[1-seat] = rival(game.observation(1-seat), copy.deepcopy(game.configuration))
            game.advance(pair)
            previous_u, previous_m = u, m
        assert len(samples) == game.t == 719
    arrays = {k: np.stack([x[k] for x in samples]) for k in samples[0]}
    assert np.array_equal(arrays['step'], np.arange(719))
    assert not arrays['previous_u'][0].any() and not arrays['previous_m'][0].any()
    assert np.array_equal(arrays['previous_u'][1:], arrays['unit_y'][:-1])
    assert np.array_equal(arrays['previous_m'][1:], arrays['market_y'][:-1])
    temp = folder / 'samples.tmp'
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(folder / 'samples.npz')
    done = dict(status='PASS', job=list(job), samples=719, sha256=digest(folder / 'samples.npz'),
                every_action_roundtrip_checked=True, teacher_sha256=plan['teacher_sha256'])
    write(folder / 'DONE.json', done)
    return done


def main():
    from cpu_budget import CpuBudget
    guard = CpuBudget(ROOT / 'collection_cpu')
    guard.wait()
    import numpy as np
    plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
    assert digest(SOURCE / 'frozen/teacher/main.py') == plan['teacher_sha256']
    assert digest(SOURCE / 'frozen/pool/POOL.json') == plan['pool_sha256']
    pool_rows = json.loads((SOURCE / 'frozen/pool/POOL.json').read_text(encoding='utf8'))
    for row in pool_rows:
        assert digest(SOURCE / f"frozen/pool/opponents/{row['id']}/main.py") == row['main_sha256']
    jobs = plan['training_jobs']
    rows = []
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as executor:
        for future in cf.as_completed([executor.submit(play, job) for job in jobs]):
            guard.wait()
            rows.append(future.result())
            status = dict(status='COLLECTING', pid=os.getpid(), completed=len(rows), games=len(jobs),
                          samples=sum(x['samples'] for x in rows), heartbeat_unix=time.time())
            write(ROOT / 'COLLECTION.json', status)
            print(json.dumps(status), flush=True)
    out = ROOT / 'data/fresh_packed'
    out.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for subdir in ('packed/train', 'numeric/train'):
        for path in (SOURCE / 'data' / subdir).glob('*.npy'):
            a = np.load(path, mmap_mode='r')
            shapes[path.stem] = (a.shape[1:], a.dtype)
    total = len(jobs)*719
    arrays = {k: np.lib.format.open_memmap(out / (k+'.npy'), mode='w+', dtype=dtype,
              shape=(total, *shape)) for k, (shape, dtype) in shapes.items()}
    episodes = []
    by_seed = {x['job'][0]: x for x in rows}
    for index, (seed, opponent, seat) in enumerate(jobs):
        guard.wait()
        path = ROOT / 'data/shards' / str(seed) / 'samples.npz'
        assert digest(path) == by_seed[seed]['sha256']
        lo, hi = index*719, (index+1)*719
        with np.load(path) as shard:
            assert set(shard.files) == set(arrays)
            for k, target in arrays.items():
                original = shard[k]
                cast = original.astype(target.dtype)
                assert np.array_equal(original, cast), f'Lossy feature cast: {k}'
                target[lo:hi] = cast
        episodes.append(dict(seed=seed, opponent=opponent, seat=seat, start=lo, stop=hi))
    for array in arrays.values():
        array.flush()
    del arrays
    write(out / 'episodes.json', episodes)
    hashes = {}
    for path in sorted(out.glob('*.npy')):
        guard.wait()
        hashes[path.name] = digest(path)
    write(out / 'READY.json', dict(status='PASS', games=len(jobs), samples=total, seeds=len(jobs),
        quantities=plan['quantities'], arrays_sha256=hashes, episodes_sha256=digest(out / 'episodes.json'),
        shared_numeric_features_sha256=digest(ROOT / 'frozen_code/numeric_features.py'),
        teacher_sha256=plan['teacher_sha256'], all_action_roundtrips_checked=total,
        provenance='Fresh online teacher rollouts; training seeds only; no evaluation trajectories'))
    write(ROOT / 'COLLECTION.json', dict(status='COMPLETE', games=len(jobs), samples=total,
                                         heartbeat_unix=time.time()))


if __name__ == '__main__':
    main()
