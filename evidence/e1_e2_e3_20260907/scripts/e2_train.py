"""agent.md section 5, E2: leave-one-opponent-out.

For each of the three opponents where aux_r0 showed the largest gain over KEEP
(Three-Day +24.0pp, G003 +11.5pp, Kaito +7.5pp; see agent.md section 2.3),
train 2 repeats x 300 rounds with that opponent EXCLUDED from the training
rollout pool (16 seeds x 6 opponents x 2 seats = 192 games/round instead of
224). Evaluate step300 on the held-out opponent only, both on the E1 test
panel (72100000, comparable across E1/E2) and confirm against KEEP.

Go: held-out-opponent gain >= half of the in-pool gain seen in E1 -> looks
    like transferable economic judgment.
No-go: in-pool gain present, held-out gain ~0 -> opponent-specific exploitation.

Everything else (model, PPO hyperparameters, KEEP=2 prior, reward, candidate
features) is identical to E1; only the training opponent SET changes. Uses
its own seed block (100,000,000+), disjoint from E1 (90-96M), the archive's
dev/monitor/final-test ranges, and the E1 test panel (72,100,000-72,100,099).
"""
import argparse, hashlib, json, sys, time, zlib
from pathlib import Path

BASE = Path.home() / 'kag/nt/latest_20260906_c3_f3_j7c3_search'
ARCHIVE = Path.home() / 'kag/nt/latest_20260906_keep2_rl_1000/archive/experiments/economic_rl_keep2_100_20260906'
sys.path.insert(0, str(BASE / 'build'))
sys.path.insert(0, str(ARCHIVE))
from run100_common import digest, save, summary, store_result  # noqa: F401 (archive utility functions; no external-data dependency)
from bias2_model import Bias2Model, update_aux
import torch
import numpy as np

OUT_ROOT = Path.home() / 'kag/e2_training'
TEST_SEED_START, TEST_SEEDS = 72100000, 100   # same panel as E1, for direct comparability
HELDOUT = {'three_day': 'yhay81_three_day', 'g003': 'g003', 'kaito': 'kaito_v58'}


def make_pool():
    import _economic_rl_native as native
    rows = json.loads((BASE / 'OPPONENTS.json').read_text(encoding='utf-8-sig'))
    NAMES = ['g001', 'g003', 'boatlee_v29', 'kaito_v58', 'lynn_v5', 'yhay81_six_day', 'yhay81_three_day']
    assert [r['name'] for r in rows] == NAMES
    data = {}
    for row in rows:
        assert hashlib.sha256((BASE / row['source']).read_bytes()).hexdigest() == row['source_sha256'], row['name']
        if row['asset']:
            data[row['name']] = json.loads(zlib.decompress((BASE / row['asset']).read_bytes()))
            assert data[row['name']]['source_sha256'] == row['source_sha256']
    return native.RLPool(data), NAMES


LIB = ARCHIVE / 'build/keep2.so'


def batch(pool, model, mode, seeds, seats, opp_idx, sample, threads=16, trace=False):
    r = pool.batch(str(LIB), str(model), mode, seeds, seats, opp_idx, sample, threads, trace)
    for row in r['rows']:
        assert not row['error'] and row['steps'] == 719 and row['plan_calls'] == 30 and row['execute_calls'] == 719
    for k in ('global', 'candidates', 'mask', 'probability', 'value', 'reward', 'logp'):
        assert np.isfinite(r[k]).all(), k
    return r


def rollout(pool, NAMES, model, mode, start, count, sample, opponents, threads=16):
    idx = [NAMES.index(o) for o in opponents]
    seeds = [s for s in range(start, start + count) for _ in opponents for _ in (0, 1)]
    seats = [seat for _ in range(start, start + count) for _ in opponents for seat in (0, 1)]
    opp = [o for _ in range(start, start + count) for o in idx for _ in (0, 1)]
    tic = time.perf_counter()
    r = batch(pool, model, mode, seeds, seats, opp, sample, threads)
    r['call_seconds'] = time.perf_counter() - tic
    r['mode'] = mode
    return r


def evaluate(pool, NAMES, model, mode, start, count, sample, opponents, out=None):
    r = rollout(pool, NAMES, model, mode, start, count, sample, opponents)
    s = summary_for(r, NAMES, opponents)
    if out is not None:
        store_result(out, r)
        save(Path(out) / 'summary.json', s)
    return s


def summary_for(r, NAMES, opponents):
    wins = [bool(row['win']) for row in r['rows']]
    margins = [row['margin'] for row in r['rows']]
    return dict(games=len(wins), wins=int(sum(wins)), win_rate=float(np.mean(wins)),
                mean_margin=float(np.mean(margins)), opponents=opponents)


def train_one(held_out_key, repeat_i, rounds, pool, NAMES):
    held_out_name = HELDOUT[held_out_key]
    train_opponents = [n for n in NAMES if n != held_out_name]
    assert len(train_opponents) == 6
    name = f'e2_{held_out_key}_r{repeat_i}'
    out = OUT_ROOT / name
    if out.exists():
        raise FileExistsError(f'{out} already exists')
    out.mkdir(parents=True)
    save(out / 'PROTOCOL.json', dict(held_out=held_out_name, train_opponents=train_opponents, rounds=rounds))

    # Disjoint seed block: 100,000,000 + (held_out index)*3,000,000 + repeat_i*1,000,000 + (step-1)*16.
    held_out_slot = list(HELDOUT).index(held_out_key)
    seed_base = 100_000_000 + held_out_slot * 3_000_000 + repeat_i * 1_000_000
    sample_base = 101000 + held_out_slot * 100 + repeat_i * 10

    torch.manual_seed(901 + held_out_slot * 10 + repeat_i)
    m = Bias2Model().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    m.export(out / 'step000.bin')
    torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=0), out / 'step000.pt')

    history = []
    tic = time.perf_counter()
    for step in range(1, rounds + 1):
        seed_start = seed_base + (step - 1) * 16
        sample_seed = sample_base + step
        r = rollout(pool, NAMES, out / f'step{step-1:03}.bin', 2, seed_start, 16, sample_seed, train_opponents)
        rs = summary_for(r, NAMES, train_opponents)
        torch.cuda.synchronize(); t = time.perf_counter()
        stats = update_aux(m, opt, r, 'cuda', step + held_out_slot * 100_000 + repeat_i * 10_000, .1)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        m.export(out / f'step{step:03}.bin')
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=step), out / f'step{step:03}.pt')
        history.append(dict(step=step, win_rate=rs['win_rate'], approx_kl=stats['approx_kl'], update_seconds=dt,
                             elapsed=time.perf_counter() - tic))
        save(out / 'PROGRESS.json', dict(status='RUNNING', history=history))
        print(f'E2 {name} step={step}/{rounds} win={rs["win_rate"]:.4f} kl={stats["approx_kl"]:.5f} '
              f'rollout_s={r["call_seconds"]:.2f} update_s={dt:.2f}', flush=True)

    final_heldout = evaluate(pool, NAMES, out / f'step{rounds:03}.bin', 1, TEST_SEED_START, TEST_SEEDS, 9931,
                              [held_out_name], out / 'final_heldout')
    final_full = evaluate(pool, NAMES, out / f'step{rounds:03}.bin', 1, TEST_SEED_START, TEST_SEEDS, 9931,
                           NAMES, out / 'final_full_panel')
    save(out / 'COMPLETE.json', dict(status='PASS', rounds=rounds, held_out=held_out_name,
                                      final_heldout=final_heldout, final_full_panel=final_full,
                                      seconds=time.perf_counter() - tic))
    print(f'E2_REPEAT_COMPLETE {name} heldout_win={final_heldout["win_rate"]:.4f} '
          f'full_panel_win={final_full["win_rate"]:.4f}', flush=True)
    return final_heldout, final_full


def keep_heldout_baselines(pool, NAMES):
    """KEEP win rate on each held-out opponent alone, same 72100000 panel -- the E2 comparison baseline."""
    out = {}
    for key, opp_name in HELDOUT.items():
        dest = OUT_ROOT / f'keep_{key}'
        if dest.exists():
            out[key] = json.loads((dest / 'summary.json').read_text())
            continue
        s = evaluate(pool, NAMES, '', 0, TEST_SEED_START, TEST_SEEDS, 1, [opp_name], dest)
        out[key] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held-out', choices=list(HELDOUT), required=True)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--rounds', type=int, default=300)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool, NAMES = make_pool()
    keep = keep_heldout_baselines(pool, NAMES)
    print('KEEP held-out baselines:', json.dumps(keep, indent=2), flush=True)
    results = {}
    for i in range(args.repeats):
        heldout, full = train_one(args.held_out, i, args.rounds, pool, NAMES)
        results[i] = dict(heldout=heldout, full=full)
    print('E2_DONE', args.held_out, json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
