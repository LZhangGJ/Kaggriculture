"""Shared paths, fixed C++ runtime, validation and paired seed statistics."""
from pathlib import Path
import os, sys, json, time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
P = Path(__file__).resolve().parent
OLD = P.parent / 'economic_rl_three_arm_20260906'
FIX = P.parent / 'economic_rl_f3_ledger_fix_20260906'
AUDIT = P.parent / 'economic_rl_f3_candidate_value_20260906'
sys.path.insert(0, str(OLD))
import numpy as np
import runtime
from runtime import read, save, digest, summary, store_result, NAMES

def rollout(pool, model='', mode=0, start=69300000, count=100, sample=9901, threads=16, trace=False):
    jobs = runtime.jobs(start, count)
    tic = time.perf_counter()
    r = pool.batch(str(FIX / 'build/f3.so'), str(model), mode,
                   [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs], sample, threads, trace)
    r['call_seconds'] = time.perf_counter() - tic
    r['bridge_seconds'] = r['call_seconds'] - r['wall_seconds']
    r['mode'] = mode
    for row in r['rows']:
        row['opponent'] = NAMES[row['opponent']]
        assert not row['error'] and row['steps'] == 719 and row['plan_calls'] == 30
        assert row['execute_calls'] == 719 and row['reference_calls'] == 0
    assert len(r['choice']) == len(jobs)*30
    for k in ('global','candidates','mask','value','probability','reward','logp'):
        assert np.isfinite(r[k]).all(), k
    assert (r['mask'][np.arange(len(r['choice'])), r['choice']] > 0).all()
    return r

def evaluate(pool, out, model='', mode=0, start=69300000, count=100, sample=9901):
    r = rollout(pool, model, mode, start, count, sample)
    store_result(out, r)
    save(Path(out)/'provenance.json', dict(checkpoint=str(model), checkpoint_sha256=digest(model) if model else None,
         library=str(FIX/'build/f3.so'), library_sha256=digest(FIX/'build/f3.so'), seed_start=start, seeds=count,
         sample=sample, mode=mode))
    s = summary(r)
    print('EVAL', Path(out).name, s['overall'], 'nonkeep', s['non_keep'], flush=True)
    return s

def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available()
    return torch

def bootstrap(values, groups, repeats=4000):
    """Seed cluster CI, not an iid per-node or per-seat interval."""
    values = np.asarray(values, dtype=float); groups = np.asarray(groups)
    unique = np.unique(groups)
    buckets = [values[groups == g] for g in unique]
    sums = np.array([v.sum() for v in buckets]); ns = np.array([len(v) for v in buckets])
    ix = np.random.default_rng(20260906).integers(0, len(unique), (repeats, len(unique)))
    samples = sums[ix].sum(1)/ns[ix].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples,[.025,.975]).tolist(), seeds=len(unique))
