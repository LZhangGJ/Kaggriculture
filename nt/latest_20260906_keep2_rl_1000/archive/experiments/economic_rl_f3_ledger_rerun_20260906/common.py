"""Explicit fixed/legacy library routing; preserve original experiment."""
from pathlib import Path
import os, sys, time
for n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[n] = '1'
P = Path(__file__).resolve().parent
OLD = P.parent / 'economic_rl_three_arm_20260906'
FIX = P.parent / 'economic_rl_f3_ledger_fix_20260906'
sys.path.insert(0, str(OLD))
from runtime import make_pool, jobs, NAMES, read, save, digest, summary, store_result
import numpy as np

FINAL_START = 66000000

def frozen_inputs():
    receipt = read(FIX / 'ACCEPTANCE.json')
    assert receipt['status'] == 'PASS'
    files = dict(receipt['preserved_original'])
    files.update(receipt['fixed_hashes'])
    files.update(read(OLD / 'SOURCE_FREEZE.json'))
    for rep in (0, 1):
        protocol = read(OLD / 'training' / f'f3_r{rep}' / 'protocol.json')
        files.update(protocol['hashes'])
        for fn in ('step000.bin', 'step020.bin', 'best.json', 'FINAL.json'):
            f = OLD / 'training' / f'f3_r{rep}' / fn
            files[str(f)] = digest(f)
    for f in P.glob('*.py'):
        files[str(f)] = digest(f)
    files[str(P / 'PLAN_ZH.md')] = digest(P / 'PLAN_ZH.md')
    check_hashes(files)
    return files

def check_hashes(files):
    for path, expected in files.items():
        assert digest(path) == expected, f'Frozen input changed: {path}'

def rollout(pool, checkpoint='', mode=0, start=FINAL_START, count=100,
            sample=9901, library='fixed', trace=False):
    so = (FIX if library == 'fixed' else OLD) / 'build/f3.so'
    assert library in ('fixed', 'old')
    jj = jobs(start, count)
    tic = time.perf_counter()
    r = pool.batch(str(so), str(checkpoint), mode, [j[0] for j in jj],
                   [j[1] for j in jj], [j[2] for j in jj], sample, 16, trace)
    r['call_seconds'] = time.perf_counter() - tic
    r['bridge_seconds'] = r['call_seconds'] - r['wall_seconds']
    r['mode'] = mode
    for row in r['rows']:
        row['opponent'] = NAMES[row['opponent']]
        assert not row['error'], row
        assert row['steps'] == 719 and row['plan_calls'] == 30 and row['execute_calls'] == 719
        assert row['reference_calls'] == 0, 'F3 must remain autonomous'
    for key in ('global', 'candidates', 'probability', 'mask', 'value', 'reward', 'logp'):
        assert np.isfinite(r[key]).all(), key
    assert len(r['choice']) == len(jj) * 30
    assert (r['mask'][np.arange(len(r['choice'])), r['choice']] > 0).all()
    return r

def evaluate(pool, out, checkpoint='', mode=0, start=FINAL_START, count=100,
             sample=9901, library='fixed'):
    r = rollout(pool, checkpoint, mode, start, count, sample, library)
    store_result(out, r)
    save(Path(out) / 'provenance.json', dict(
        library=library, library_path=str((FIX if library == 'fixed' else OLD) / 'build/f3.so'),
        library_sha256=digest((FIX if library == 'fixed' else OLD) / 'build/f3.so'),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        mode=mode, seed_start=start, seed_count=count, sample_seed=sample))
    s = summary(r)
    print('EVAL', Path(out).name, s['overall'], 'non_keep', s['non_keep'], flush=True)
    return s
