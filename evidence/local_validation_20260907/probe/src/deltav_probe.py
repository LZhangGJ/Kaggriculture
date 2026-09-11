"""agent.md section 9.6 step 0: zero-parameter probe on F3's own marginal valuation (dV_j).

Runs three configurations on the same seeds / opponents / seats:
  A. build/f3.so         mode 0  -> reference KEEP (frozen library)
  B. build/f3_deltav.so  mode 0  -> KEEP through the patched library; must equal A exactly
  C. build/f3_deltav.so  mode 4  -> argmax_j dV_j, KEEP unless max dV > 0
Reports paired win-rate / margin differences with a seed-block bootstrap interval.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from runtime import P, NAMES, make_pool, jobs, summary, save_json, sha

def run(pool, library, mode, joblist, threads):
    t = time.perf_counter()
    r = pool.batch(str(library), '', mode, [j[0] for j in joblist], [j[1] for j in joblist], [j[2] for j in joblist], 17, threads, False)
    r['call_seconds'] = time.perf_counter() - t
    r['library_sha256'] = sha(library); r['plan_sha256'] = None
    for row in r['rows']:
        row['opponent_name'] = NAMES[row['opponent']]
        assert not row['error'] and row['steps'] == 719, row
    return r

def key(row):
    return (row['seed'], row['seat'], row['opponent'])

def paired(a, b, boot=2000, seed=0):
    """b minus a, paired per (seed, seat, opponent); CI by resampling whole seeds."""
    ra = {key(r): r for r in a['rows']}; rb = {key(r): r for r in b['rows']}
    assert ra.keys() == rb.keys()
    keys = sorted(ra)
    dwin = np.array([float(rb[k]['win']) - float(ra[k]['win']) for k in keys])
    dmar = np.array([rb[k]['margin'] - ra[k]['margin'] for k in keys])
    seeds = np.array([k[0] for k in keys]); uniq = np.unique(seeds)
    rng = np.random.default_rng(seed); wins = []; mars = []
    for _ in range(boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(seeds == s) for s in pick])
        wins.append(dwin[idx].mean()); mars.append(dmar[idx].mean())
    rescued = int(sum(1 for k in keys if rb[k]['win'] and not ra[k]['win']))
    lost = int(sum(1 for k in keys if ra[k]['win'] and not rb[k]['win']))
    return dict(games=len(keys), seeds=int(len(uniq)),
                win_rate_diff_pp=100 * dwin.mean(), win_ci95_pp=[100 * np.percentile(wins, 2.5), 100 * np.percentile(wins, 97.5)],
                margin_diff=float(dmar.mean()), margin_ci95=[float(np.percentile(mars, 2.5)), float(np.percentile(mars, 97.5))],
                rescued=rescued, lost=lost)

def exact_equal(a, b):
    ra = {key(r): r for r in a['rows']}; rb = {key(r): r for r in b['rows']}
    if ra.keys() != rb.keys(): return False
    return all(ra[k]['cash'] == rb[k]['cash'] and ra[k]['opponent_cash'] == rb[k]['opponent_cash'] for k in ra)

def dv_stats(r):
    """probability[] carries dV_j/1e4 for masked candidates in mode 4; value carries V_keep/1e5."""
    mask = r['mask'] > 0; dv = r['probability'] * 1e4
    active = mask.sum(1) > 1
    best = np.where(mask[:, 1:], dv[:, 1:], -np.inf).max(1)
    chosen = r['choice']
    out = dict(decisions=int(len(chosen)), active_nodes=int(active.sum()),
               changed_nodes=int((chosen > 0).sum()),
               nodes_with_positive_dv=int(((best > 0) & active).sum()),
               mean_best_dv_when_positive=float(best[(best > 0) & active].mean()) if ((best > 0) & active).any() else 0.0,
               median_best_dv_when_positive=float(np.median(best[(best > 0) & active])) if ((best > 0) & active).any() else 0.0,
               mean_vkeep=float((r['value'] * 1e5).mean()))
    kinds = {}
    for k in np.unique(r['kind'][chosen > 0]):
        kinds[int(k)] = int((r['kind'][chosen > 0] == k).sum())
    out['chosen_kind_counts'] = kinds
    byday = {}
    for d in range(30):
        sel = (r['day'] == d) & active
        if sel.any(): byday[d] = dict(nodes=int(sel.sum()), changed=int((chosen[sel] > 0).sum()), positive=int((best[sel] > 0).sum()))
    out['by_day'] = byday
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-start', type=int, default=71100000)
    ap.add_argument('--seeds', type=int, default=100)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): ap.error('output exists')
    args.out.mkdir(parents=True)
    pool = make_pool(); js = jobs(args.seed_start, args.seeds)
    ref = P / 'build/f3.so'; probe = P / 'build/f3_deltav.so'
    print('games per config', len(js), flush=True)
    A = run(pool, ref, 0, js, args.threads);   print('A f3.so keep      ', summary(A)['overall'], round(A['call_seconds'], 1), 's', flush=True)
    B = run(pool, probe, 0, js, args.threads); print('B deltav.so keep  ', summary(B)['overall'], round(B['call_seconds'], 1), 's', flush=True)
    C = run(pool, probe, 4, js, args.threads); print('C deltav.so mode4 ', summary(C)['overall'], round(C['call_seconds'], 1), 's', flush=True)
    same = exact_equal(A, B)
    print('B == A exactly (cash & opponent cash every game):', same, flush=True)
    per_opp = {n: dict(keep=summary(A)['per_opponent'][n]['win_rate'], probe=summary(C)['per_opponent'][n]['win_rate']) for n in NAMES}
    report = dict(seed_start=args.seed_start, seeds=args.seeds, games=len(js), threads=args.threads,
                  libraries=dict(f3=sha(ref), f3_deltav=sha(probe)),
                  patched_keep_identical_to_frozen=same,
                  keep=summary(A), probe=summary(C), per_opponent=per_opp,
                  paired_probe_minus_keep=paired(A, C), dv=dv_stats(C))
    save_json(args.out / 'REPORT.json', report)
    for name, r in (('keep_f3', A), ('keep_patched', B), ('probe_mode4', C)):
        save_json(args.out / f'{name}_games.json', r['rows'])
    np.savez_compressed(args.out / 'probe_mode4_decisions.npz', **{k: v for k, v in C.items() if isinstance(v, np.ndarray)})
    print(json.dumps(dict(paired=report['paired_probe_minus_keep'], per_opponent=per_opp,
                          dv={k: v for k, v in report['dv'].items() if k != 'by_day'}), indent=2), flush=True)
    print('PROBE_DONE', flush=True)

if __name__ == '__main__':
    main()
