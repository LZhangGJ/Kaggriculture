"""agent.md section 5, E3: paired-KEEP reward baseline.

Structural cause (agent.md section 3): on day 0 every game starts identical,
so the critic can only output a constant there; the actual variance in early
advantages comes entirely from opponent identity (KEEP win rate 42-100% across
the 7 opponents) and the seed's town-shop draw, neither of which any baseline
currently subtracts.

Fix: for every training round's 224 games (16 new seeds x 7 opponents x 2
seats, identical to E1), also play the SAME (seed, seat, opponent) triples
under mode=0 (pure KEEP, no network) and use reward_shaped = reward_model -
reward_keep as the PPO training signal instead of the raw terminal reward.
This is exactly the paired-difference statistic already used throughout this
project's own evaluation (independent1000's paired win-rate tables, the
section-9 probe's paired()); the only change is using it to train on, not
just to report. All other training/critic/GAE code is untouched -- only the
reward array fed into update_aux differs.

Native ordering guarantee this relies on: RLPool::batch() writes results into
a pre-sized games[n] vector indexed by input position n (runner.hpp), and
OpenMP's `schedule(dynamic)` parallelizes computation, not output placement --
two calls with identical (seeds, seats, opponents) lists always produce
identically-ordered decision arrays, so reward arrays from two separate calls
can be subtracted elementwise without any extra game/day matching.

3 repeats x 300 rounds, aux arm only (KEEP=2, same as E1), compared against
E1's 6 repeats: if the paired reward reduces cross-repeat variance without
lowering the mean, the signal-to-noise diagnosis in section 3 is the
correctable bottleneck; if not, the problem is candidate expressiveness, not
reward variance (agent.md section 5, E3 go/no-go).
"""
import argparse, hashlib, json, sys, time, zlib
from pathlib import Path

BASE = Path.home() / 'kag/nt/latest_20260906_c3_f3_j7c3_search'
ARCHIVE = Path.home() / 'kag/nt/latest_20260906_keep2_rl_1000/archive/experiments/economic_rl_keep2_100_20260906'
sys.path.insert(0, str(BASE / 'build'))
sys.path.insert(0, str(ARCHIVE))
from run100_common import digest, save, summary, store_result  # noqa: F401
from bias2_model import Bias2Model, update_aux
import torch
import numpy as np

OUT_ROOT = Path.home() / 'kag/e3_training'
TEST_SEED_START, TEST_SEEDS = 72100000, 100     # same panel as E1
NAMES = ['g001', 'g003', 'boatlee_v29', 'kaito_v58', 'lynn_v5', 'yhay81_six_day', 'yhay81_three_day']
LIB = ARCHIVE / 'build/keep2.so'


def make_pool():
    import _economic_rl_native as native
    rows = json.loads((BASE / 'OPPONENTS.json').read_text(encoding='utf-8-sig'))
    assert [r['name'] for r in rows] == NAMES
    data = {}
    for row in rows:
        assert hashlib.sha256((BASE / row['source']).read_bytes()).hexdigest() == row['source_sha256'], row['name']
        if row['asset']:
            data[row['name']] = json.loads(zlib.decompress((BASE / row['asset']).read_bytes()))
            assert data[row['name']]['source_sha256'] == row['source_sha256']
    return native.RLPool(data)


def raw_batch(pool, model, mode, seeds, seats, opp_idx, sample, threads=16):
    r = pool.batch(str(LIB), str(model), mode, seeds, seats, opp_idx, sample, threads, False)
    for row in r['rows']:
        assert not row['error'] and row['steps'] == 719 and row['plan_calls'] == 30 and row['execute_calls'] == 719
    return r


def jobs(start, count, opponents=NAMES):
    idx = [NAMES.index(o) for o in opponents]
    seeds = [s for s in range(start, start + count) for _ in opponents for _ in (0, 1)]
    seats = [seat for _ in range(start, start + count) for _ in opponents for seat in (0, 1)]
    opp = [o for _ in range(start, start + count) for o in idx for _ in (0, 1)]
    return seeds, seats, opp


def rollout(pool, model, mode, start, count, sample, threads=16):
    seeds, seats, opp = jobs(start, count)
    tic = time.perf_counter()
    r = raw_batch(pool, model, mode, seeds, seats, opp, sample, threads)
    r['call_seconds'] = time.perf_counter() - tic
    r['mode'] = mode
    return r


def rollout_paired(pool, model, start, count, sample, threads=16):
    """Model rollout (mode=2, sampling) plus KEEP on the identical (seed,seat,opponent)
    triples (mode=0); returns the model rollout dict with reward replaced by
    reward_model - reward_keep. See module docstring for the ordering guarantee."""
    seeds, seats, opp = jobs(start, count)
    tic = time.perf_counter()
    r = raw_batch(pool, model, 2, seeds, seats, opp, sample, threads)
    r_keep = raw_batch(pool, '', 0, seeds, seats, opp, sample, threads)
    assert np.array_equal(r['game'], r_keep['game']) and np.array_equal(r['day'], r_keep['day'])
    r = dict(r)  # shallow copy; do not mutate the native-returned dict's original reward array in place
    r['reward'] = r['reward'] - r_keep['reward']
    r['call_seconds'] = time.perf_counter() - tic
    r['keep_call_seconds'] = None
    r['raw_win_rate'] = float(np.mean([row['win'] for row in r['rows']]))
    r['keep_win_rate'] = float(np.mean([row['win'] for row in r_keep['rows']]))
    return r


def evaluate(pool, out, model, mode, start, count, sample):
    r = rollout(pool, model, mode, start, count, sample)
    s = dict(games=len(r['rows']), wins=int(sum(row['win'] for row in r['rows'])),
             win_rate=float(np.mean([row['win'] for row in r['rows']])),
             mean_margin=float(np.mean([row['margin'] for row in r['rows']])))
    if out is not None:
        store_result(out, r)
        save(Path(out) / 'summary.json', s)
    print('EVAL', Path(out).name if out else '', s, flush=True)
    return s


def train_one(repeat_i, rounds, pool):
    name = f'aux_paired_e{repeat_i}'
    out = OUT_ROOT / name
    if out.exists():
        raise FileExistsError(f'{out} already exists')
    out.mkdir(parents=True)
    torch.manual_seed(951 + repeat_i)
    m = Bias2Model().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    m.export(out / 'step000.bin')
    torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=0), out / 'step000.pt')

    seed_base = 110_000_000 + repeat_i * 1_000_000
    sample_base = 111000 + repeat_i * 1000
    history = []
    tic = time.perf_counter()
    for step in range(1, rounds + 1):
        seed_start = seed_base + (step - 1) * 16
        sample_seed = sample_base + step
        r = rollout_paired(pool, out / f'step{step-1:03}.bin', seed_start, 16, sample_seed)
        torch.cuda.synchronize(); t = time.perf_counter()
        r_for_update = {**r, 'mode': 2}
        stats = update_aux(m, opt, r_for_update, 'cuda', step + repeat_i * 10_000, .1)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        m.export(out / f'step{step:03}.bin')
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=step), out / f'step{step:03}.pt')
        history.append(dict(step=step, raw_win_rate=r['raw_win_rate'], keep_win_rate=r['keep_win_rate'],
                             mean_shaped_reward=float(r['reward'][r['reward'] != 0].mean()) if (r['reward'] != 0).any() else 0.,
                             approx_kl=stats['approx_kl'], update_seconds=dt, elapsed=time.perf_counter() - tic))
        save(out / 'PROGRESS.json', dict(status='RUNNING', history=history))
        print(f'E3 {name} step={step}/{rounds} raw_win={r["raw_win_rate"]:.4f} keep_win={r["keep_win_rate"]:.4f} '
              f'kl={stats["approx_kl"]:.5f} rollout_s={r["call_seconds"]:.2f} update_s={dt:.2f}', flush=True)

    final = evaluate(pool, out / 'final_test', out / f'step{rounds:03}.bin', 1, TEST_SEED_START, TEST_SEEDS, 9931)
    save(out / 'COMPLETE.json', dict(status='PASS', rounds=rounds, keep_bonus=2., aux_weight=.1,
                                      seconds=time.perf_counter() - tic, final_test=final))
    print(f'E3_REPEAT_COMPLETE {name} final_test_win_rate={final["win_rate"]:.4f}', flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--start-index', type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = make_pool()
    keep_dest = OUT_ROOT / 'keep_on_e1_panel'
    if keep_dest.exists():
        keep = json.loads((keep_dest / 'summary.json').read_text())
    else:
        keep = evaluate(pool, keep_dest, '', 0, TEST_SEED_START, TEST_SEEDS, 1)
    print('KEEP on E1/E3 shared test panel:', keep, flush=True)
    results = {'keep': keep, 'repeats': {}}
    for i in range(args.start_index, args.start_index + args.repeats):
        results['repeats'][i] = train_one(i, args.rounds, pool)
        save(OUT_ROOT / 'E3_SUMMARY_SO_FAR.json', results)
    print('E3_ALL_DONE', json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
