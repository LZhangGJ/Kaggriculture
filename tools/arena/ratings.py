"""Draw-aware ratings and seed-block uncertainty; no LLM arithmetic."""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


METHOD = "half-win-bt-v2"
DESCRIPTION = "Bradley-Terry; draws half wins; centered strengths; Gaussian SD=3 regularization; no seat adjustment"


def fit(rows, ids):
    index = {a: i for i, a in enumerate(ids)}
    n = len(ids)
    if not rows:
        return {a: None for a in ids}, None
    a = np.array([index[r["agents"][0]] for r in rows])
    b = np.array([index[r["agents"][1]] for r in rows])
    points = np.array([{"win0": 1., "win1": 0., "draw": .5}[r["outcome"]] for r in rows])
    totals = np.zeros((n, n))
    wins = np.zeros((n, n))
    np.add.at(totals, (a, b), 1)
    np.add.at(wins, (a, b), points)
    a, b = np.nonzero(totals)
    totals, wins = totals[a, b], wins[a, b]

    def objective(x):
        d = x[a] - x[b]
        loss = np.sum(totals * np.logaddexp(0, d) - wins * d) + np.sum(x*x)/18
        residual = totals * expit(d) - wins
        grad = x/9
        np.add.at(grad, a, residual)
        np.add.at(grad, b, -residual)
        return loss, grad

    result = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-8, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError("Rating fit failed: " + result.message)
    strengths = result.x - result.x.mean()
    return dict(zip(ids, strengths.tolist())), None


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
    return {"method": METHOD, "method_description": DESCRIPTION, "ratings": ratings, "intervals": intervals, "components": groups, "stats": stats, "matrix": matrix}


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
