"""Draw-aware ratings and seed-block uncertainty; no LLM arithmetic."""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def fit(rows, ids):
    index = {a: i for i, a in enumerate(ids)}
    n = len(ids)
    if not rows:
        return {a: None for a in ids}, None
    a = np.array([index[r["agents"][0]] for r in rows])
    b = np.array([index[r["agents"][1]] for r in rows])
    y = np.array([{"win0": 0, "win1": 1, "draw": 2}[r["outcome"]] for r in rows])
    counts = np.zeros((n, n, 3))
    np.add.at(counts, (a, b, y), 1)
    a, b = np.nonzero(counts.sum(axis=2))
    counts = counts[a, b]
    totals = counts.sum(axis=1)
    def objective(x):
        strengths = np.r_[0., x[:n-1]]
        d = strengths[a] - strengths[b] + x[n-1]
        logits = np.stack((d/2, -d/2, np.full(len(a), x[n])), axis=1)
        lse = logsumexp(logits, axis=1)
        loss = np.sum(totals * lse - (counts * logits).sum(axis=1)) + np.sum(x*x)/18
        residual = np.exp(logits-lse[:, None]) * totals[:, None] - counts
        grad_d = (residual[:, 0]-residual[:, 1])/2
        strength_grad = np.zeros(n)
        np.add.at(strength_grad, a, grad_d)
        np.add.at(strength_grad, b, -grad_d)
        grad = np.r_[strength_grad[1:], grad_d.sum(), residual[:, 2].sum()] + x/9
        return loss, grad
    result = minimize(objective, np.zeros(n+1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError("Rating fit failed: " + result.message)
    return dict(zip(ids, np.r_[0., result.x[:n-1]].tolist())), float(result.x[n-1])


def components(rows, ids):
    links = {a: set() for a in ids}
    for r in rows:
        a, b = r["agents"]
        links[a].add(b)
        links[b].add(a)
    result, unseen = [], set(ids)
    while unseen:
        todo, found = [min(unseen)], set()
        while todo:
            a = todo.pop()
            if a in found:
                continue
            found.add(a)
            todo.extend(links[a] - found)
        unseen -= found
        result.append(sorted(found))
    return result


def summary(rows, ids, bootstrap=0):
    groups = components(rows, ids)
    ratings, intervals = {}, {}
    for group in groups:
        subset = [r for r in rows if r["agents"][0] in group]
        values, _ = fit(subset, group)
        ratings.update(values)
        blocks = defaultdict(list)
        for r in subset:
            blocks[r["seed"]].append(r)
        keys = list(blocks)
        samples = defaultdict(list)
        rng = np.random.default_rng(20260913)
        if len(keys) > 1 and bootstrap:
            for _ in range(bootstrap):
                resampled = [r for key in rng.choice(keys, len(keys)) for r in blocks[key]]
                values, _ = fit(resampled, group)
                for aid, val in values.items():
                    samples[aid].append(val)
        for aid in group:
            intervals[aid] = np.quantile(samples[aid], [.025, .975]).tolist() if samples[aid] else None
    stats, matrix = {}, {}
    for aid in ids:
        rr = [r for r in rows if aid in r["agents"]]
        stats[aid] = tally(rr, aid)
        for other in ids:
            if other != aid:
                matrix[aid + ":" + other] = tally([r for r in rr if other in r["agents"]], aid)
    return {"ratings": ratings, "intervals": intervals, "components": groups, "stats": stats, "matrix": matrix}


def tally(rows, aid):
    wins = draws = 0
    margins = []
    seats = {"0": {"games": 0, "wins": 0, "draws": 0}, "1": {"games": 0, "wins": 0, "draws": 0}}
    for r in rows:
        seat = r["agents"].index(aid)
        win = r["outcome"] == f"win{seat}"
        draw = r["outcome"] == "draw"
        wins += win
        draws += draw
        seats[str(seat)]["games"] += 1
        seats[str(seat)]["wins"] += int(win)
        seats[str(seat)]["draws"] += int(draw)
        if r.get("cash") is not None:
            margins.append(r["cash"][seat] - r["cash"][1-seat])
    n = len(rows)
    return {"games": n, "wins": wins, "draws": draws, "losses": n-wins-draws,
            "win_rate": wins/n if n else None, "score": (wins+draws/2)/n if n else None,
            "cash_margin": sum(margins)/len(margins) if margins else None, "cash_games": len(margins), "seats": seats}
