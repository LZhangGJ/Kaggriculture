"""agent.md section 5, E1: repeat training with new seeds to test whether aux_r0's
+8.64pp gain over KEEP (1000-round independent test) is recipe or luck.

Same recipe as the archived KEEP=2 aux runs (Bias2Model, KEEP prior fixed at 2.0,
lr=3e-4 Adam, PPO 4 epochs/minibatch<=512/clip .2/entropy .01, GAE lambda .95,
aux weight 0.1, 16 new seeds x 7 opponents x 2 seats = 224 games/round), run to
300 rounds (agent.md: the aux_r0 gain was already present at 300 rounds, so 300
is sufficient -- do not need 1000). Only the run index / RNG seeds are new; no
training code, network, reward or candidate change relative to the archive.

Seed ranges (disjoint from every existing block; existing blocks found by
grep: 69300000, 69600000-69716000ish, 69800000-69800032 [dev, reused as-is],
69900000, 70300000-70300100 [monitor], 71100000-71100100 [old final test]):
  training rollout : 90000000 + i*1_000_000 + (step-1)*16   (i = repeat index)
  ppo shuffle seed  : 91000 + i*1000 + step
  dev selection     : 69800000, 32 seeds  (same fixed set as the archive, for
                       comparability of the selection criterion only)
  final test panel  : 72100000, 100 seeds (new panel, per agent.md; never used
                       for training or dev seeds of any repeat)
"""
import argparse, hashlib, json, sys, time, zlib
from pathlib import Path

BASE = Path.home() / 'kag/nt/latest_20260906_c3_f3_j7c3_search'
ARCHIVE = Path.home() / 'kag/nt/latest_20260906_keep2_rl_1000/archive/experiments/economic_rl_keep2_100_20260906'
sys.path.insert(0, str(BASE / 'build'))   # provides _economic_rl_native for the archive's runtime.py
sys.path.insert(0, str(ARCHIVE))
from run100_common import *          # noqa: F401,F403  (read, save, digest, summary, store_result, NAMES, runtime, np, configure_torch, rollout, evaluate)
from bias2_model import Bias2Model, update_aux
import torch                          # not re-exported by run100_common's star import; matches train100.py's own explicit import


def make_pool():
    """The archive's own runtime.make_pool() references an opponent registry
    under daily_dp_v7_20260903/, a directory this lightweight package excludes
    (README: "本机全量数据没有删除；仍在原...experiments/ 中"). Build the same
    RLPool object the shipped package already builds successfully, from
    latest_20260906_c3_f3_j7c3_search/OPPONENTS.json instead. Everything else
    from the archive's runtime module (read/save/digest/summary/store_result/
    NAMES, used by rollout()/evaluate()) is untouched and unaffected."""
    import _economic_rl_native as native   # already loaded via run100_common's import chain; this just fetches the cached module
    rows = json.loads((BASE / 'OPPONENTS.json').read_text(encoding='utf-8-sig'))
    assert [r['name'] for r in rows] == NAMES
    data = {}
    for row in rows:
        assert hashlib.sha256((BASE / row['source']).read_bytes()).hexdigest() == row['source_sha256'], row['name']
        if row['asset']:
            data[row['name']] = json.loads(zlib.decompress((BASE / row['asset']).read_bytes()))
            assert data[row['name']]['source_sha256'] == row['source_sha256']
    return native.RLPool(data)

OUT_ROOT = Path.home() / 'kag/e1_training'
TEST_SEED_START, TEST_SEEDS = 72100000, 100
DEV_SEED_START, DEV_SEEDS = 69800000, 32


def train_one(arm, i, rounds, pool):
    name = f'{arm}_e{i}'
    out = OUT_ROOT / name
    if out.exists():
        raise FileExistsError(f'{out} already exists; this script never resumes silently')
    out.mkdir(parents=True)
    torch.manual_seed(801 + i)
    m = Bias2Model().cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    m.export(out / 'step000.bin')
    torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=0,
                     bin_sha=digest(out / 'step000.bin')), out / 'step000.pt')
    weight = .1 if arm == 'aux' else 0.
    history, selection = [], []
    tic = time.perf_counter()

    def dev(step):
        dest = out / f'dev_{step:03}'
        s = evaluate(pool, dest, out / f'step{step:03}.bin', 1, DEV_SEED_START, DEV_SEEDS, 9301)
        selection.append(dict(step=step, score=[s['overall']['win_rate'], s['overall']['mean_margin'], -step], summary=s))
        save(out / 'selection.json', selection)

    for step in range(1, rounds + 1):
        seed_start = 90_000_000 + i * 1_000_000 + (step - 1) * 16
        sample_seed = 91000 + i * 1000 + step
        r = rollout(pool, out / f'step{step-1:03}.bin', 2, seed_start, 16, sample_seed)
        rs = summary(r)
        torch.cuda.synchronize(); t = time.perf_counter()
        stats = update_aux(m, opt, r, 'cuda', step + i * 10_000, weight)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        m.export(out / f'step{step:03}.bin')
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), step=step,
                         bin_sha=digest(out / f'step{step:03}.bin')), out / f'step{step:03}.pt')
        history.append(dict(step=step, training_games=step * 224, rollout=dict(win_rate=rs['overall']['win_rate'], call_seconds=rs['call_seconds']),
                             ppo=dict(loss=stats['loss'], approx_kl=stats['approx_kl'], aux_loss=stats.get('aux_loss', 0.)),
                             update_seconds=dt, elapsed=time.perf_counter() - tic))
        save(out / 'PROGRESS.json', dict(status='RUNNING', history=history, elapsed=time.perf_counter() - tic))
        print(f'E1 {name} step={step}/{rounds} win={rs["overall"]["win_rate"]:.4f} '
              f'kl={stats["approx_kl"]:.5f} rollout_s={rs["call_seconds"]:.2f} update_s={dt:.2f}', flush=True)
        if step % 20 == 0:
            dev(step)

    final = evaluate(pool, out / 'final_test', out / f'step{rounds:03}.bin', 1, TEST_SEED_START, TEST_SEEDS, 9931)
    save(out / 'COMPLETE.json', dict(status='PASS', rounds=rounds, training_games=rounds * 224,
                                      keep_bonus=2., aux_weight=weight, seconds=time.perf_counter() - tic,
                                      final_test=final))
    print(f'E1_REPEAT_COMPLETE {name} final_test_win_rate={final["overall"]["win_rate"]:.4f}', flush=True)
    return final


def keep_baseline(pool):
    """KEEP (no model) on the same E1 final-test panel, for the paired comparison."""
    dest = OUT_ROOT / 'keep_on_e1_panel'
    if dest.exists():
        return json.loads((dest / 'summary.json').read_text())
    s = evaluate(pool, dest, '', 0, TEST_SEED_START, TEST_SEEDS, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('control', 'aux'), default='aux')
    ap.add_argument('--repeats', type=int, default=6)
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--start-index', type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    configure_torch()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = make_pool()
    keep = keep_baseline(pool)
    print('KEEP on E1 test panel:', keep['overall'], flush=True)
    results = {'keep': keep, 'repeats': {}}
    for i in range(args.start_index, args.start_index + args.repeats):
        final = train_one(args.arm, i, args.rounds, pool)
        results['repeats'][i] = final
        save(OUT_ROOT / f'{args.arm}_SUMMARY_SO_FAR.json', results)
    print('E1_ALL_DONE', json.dumps({k: v['overall'] if k != 'repeats' else {i: r['overall'] for i, r in v.items()} for k, v in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
