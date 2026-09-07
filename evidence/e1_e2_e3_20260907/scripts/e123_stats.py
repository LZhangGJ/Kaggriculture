"""Paired statistics for E1/E2/E3 from the raw per-game results (seed-block bootstrap, same method as the section-9 probe)."""
import json, sys
from pathlib import Path
import numpy as np

W = Path('//wsl.localhost/Ubuntu-24.04/home/lzhang/kag')
NAMES = ['g001', 'g003', 'boatlee_v29', 'kaito_v58', 'lynn_v5', 'yhay81_six_day', 'yhay81_three_day']
def key(r):
    o = r['opponent']
    return (r['seed'], r['seat'], NAMES[o] if isinstance(o, int) else o)   # archive rollout() rewrites opponent to its name; e2/e3 keep the index


def load(p):
    return json.load(open(p, encoding='utf-8'))


def paired(a_rows, b_rows, boot=4000, seed=0):
    """b minus a, paired on (seed, seat, opponent); CI by resampling whole seeds."""
    ra = {key(r): r for r in a_rows}; rb = {key(r): r for r in b_rows}
    assert ra.keys() == rb.keys(), (len(ra), len(rb))
    keys = sorted(ra)
    dwin = np.array([float(rb[k]['win']) - float(ra[k]['win']) for k in keys])
    dmar = np.array([rb[k]['margin'] - ra[k]['margin'] for k in keys])
    seeds = np.array([k[0] for k in keys]); uniq = np.unique(seeds)
    rng = np.random.default_rng(seed); w = []; m = []
    for _ in range(boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(seeds == s) for s in pick])
        w.append(dwin[idx].mean()); m.append(dmar[idx].mean())
    return dict(games=len(keys), seeds=int(len(uniq)),
                win_b=float(np.mean([rb[k]['win'] for k in keys])), win_a=float(np.mean([ra[k]['win'] for k in keys])),
                diff_pp=float(100 * dwin.mean()), ci_pp=[float(100 * np.percentile(w, 2.5)), float(100 * np.percentile(w, 97.5))],
                margin_diff=float(dmar.mean()), margin_ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                rescued=int(sum(rb[k]['win'] and not ra[k]['win'] for k in keys)),
                lost=int(sum(ra[k]['win'] and not rb[k]['win'] for k in keys)))


out = {}

# ---------------- E1 ----------------
keep_rows = load(W / 'e1_training/keep_on_e1_panel/games.json')
keep_sum = load(W / 'e1_training/keep_on_e1_panel/summary.json')
e1 = {}
for i in range(6):
    d = W / f'e1_training/aux_e{i}'
    rows = load(d / 'final_test/games.json'); s = load(d / 'final_test/summary.json'); c = load(d / 'COMPLETE.json')
    sel = load(d / 'selection.json')
    best = max(sel, key=lambda x: tuple(x['score']))
    e1[i] = dict(paired=paired(keep_rows, rows), per_opponent={n: s['per_opponent'][n]['win_rate'] for n in NAMES},
                 non_keep=s['non_keep'], decisions=s['decisions'], seconds=c['seconds'],
                 dev_best_step=best['step'], dev_best_win=best['summary']['overall']['win_rate'],
                 dev_step300_win=next(x['summary']['overall']['win_rate'] for x in sel if x['step'] == 300))
out['E1'] = dict(keep=dict(win=keep_sum['overall']['win_rate'], per_opponent={n: keep_sum['per_opponent'][n]['win_rate'] for n in NAMES}), repeats=e1)

# ---------------- E2 ----------------
HELD = {'three_day': 'yhay81_three_day', 'g003': 'g003', 'kaito': 'kaito_v58'}
e2 = {}
for k, name in HELD.items():
    kr = load(W / f'e2_training/keep_{k}/games.json')
    e2[k] = {}
    for r in (0, 1):
        d = W / f'e2_training/e2_{k}_r{r}'
        hr = load(d / 'final_heldout/games.json'); fr = load(d / 'final_full_panel/games.json'); c = load(d / 'COMPLETE.json')
        e2[k][r] = dict(heldout=paired(kr, hr), full_panel=paired(keep_rows, fr), seconds=c['seconds'])
out['E2'] = e2

# ---------------- E3 ----------------
e3 = {}
for i in range(3):
    d = W / f'e3_training/aux_paired_e{i}'
    if (d / 'COMPLETE.json').exists():
        rows = load(d / 'final_test/games.json'); c = load(d / 'COMPLETE.json'); prog = load(d / 'PROGRESS.json')
        e3[i] = dict(paired=paired(keep_rows, rows), seconds=c['seconds'], status='complete')
    elif (d / 'PROGRESS.json').exists():
        prog = load(d / 'PROGRESS.json'); e3[i] = dict(status='running', step=prog['history'][-1]['step'])
out['E3'] = e3

json.dump(out, open(sys.argv[1], 'w'), indent=1)

# ---------------- print ----------------
print('=== E1: aux KEEP=2, 6 fresh seeds, step300 greedy vs KEEP on panel 72100000 (1400 games) ===')
print(f"KEEP {100*out['E1']['keep']['win']:.2f}%")
for i, r in e1.items():
    p = r['paired']
    print(f"aux_e{i}: {100*p['win_b']:.2f}%  {p['diff_pp']:+.2f}pp [{p['ci_pp'][0]:+.2f}, {p['ci_pp'][1]:+.2f}]  margin {p['margin_diff']:+.0f}  rescued/lost {p['rescued']}/{p['lost']}  nonkeep {r['non_keep']}/{r['decisions']}  dev-best step {r['dev_best_step']} ({100*r['dev_best_win']:.1f}% dev)  {r['seconds']/60:.0f} min")
print('per-opponent (KEEP | e0..e5 | mean of 6):')
for n in NAMES:
    vals = [100 * e1[i]['per_opponent'][n] for i in range(6)]
    print(f"  {n:18s} {100*out['E1']['keep']['per_opponent'][n]:5.1f} | " + ' '.join(f'{v:5.1f}' for v in vals) + f" | {np.mean(vals):5.1f}")
print()
print('=== E2: leave-one-opponent-out, step300 greedy ===')
for k in HELD:
    for r in (0, 1):
        h = e2[k][r]['heldout']; f = e2[k][r]['full_panel']
        print(f"{k:10s} r{r}: held-out {100*h['win_b']:.1f}% vs KEEP {100*h['win_a']:.1f}%  {h['diff_pp']:+.1f}pp [{h['ci_pp'][0]:+.1f}, {h['ci_pp'][1]:+.1f}] (200 games)   full panel {f['diff_pp']:+.2f}pp [{f['ci_pp'][0]:+.2f}, {f['ci_pp'][1]:+.2f}]")
print()
print('=== E3: paired-KEEP reward, step300 greedy vs KEEP ===')
for i, r in e3.items():
    if r['status'] == 'complete':
        p = r['paired']
        print(f"aux_paired_e{i}: {100*p['win_b']:.2f}%  {p['diff_pp']:+.2f}pp [{p['ci_pp'][0]:+.2f}, {p['ci_pp'][1]:+.2f}]  margin {p['margin_diff']:+.0f}  {r['seconds']/60:.0f} min")
    else:
        print(f"aux_paired_e{i}: running, step {r['step']}/300")
