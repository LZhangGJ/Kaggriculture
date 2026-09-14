"""Predeclared matched promotion and per-opponent target checks."""
from pathlib import Path
import numpy as np
from scipy.stats import beta

from . import schedule
from .store import digest, ident, now, read, write


def plan_comparison(root, run_id, candidate, incumbent, panel, count=256):
    root = Path(root)
    if read(root / "runs" / ident(run_id) / "manifest.json"):
        raise ValueError("Comparison ID exists")
    if candidate == incumbent or any(e["agent"] in (candidate, incumbent) for e in panel):
        raise ValueError("Candidate/incumbent must be distinct from panel")
    if not all(e.get("family") and e.get("panel") in ("representative", "original", "stress") for e in panel):
        raise ValueError("Each opponent requires family and panel")
    if {e["panel"] for e in panel} != {"representative", "original", "stress"}:
        raise ValueError("All three panels required")
    inc = read(root / "agents" / (ident(incumbent) + ".json"))
    if not inc or not inc["build_verified"]:
        raise ValueError("Incumbent not verified")
    m = schedule.plan(root, run_id, "confirmation", count, candidate, [e["agent"] for e in panel])
    extra = []
    for g in m["games"]:
        h = {k: v for k, v in g.items() if k != "id"}
        h["agents"] = [incumbent if a == candidate else a for a in g["agents"]]
        extra.append({**h, "id": digest(h)})
    m["games"] += extra
    m["agents"][incumbent] = inc
    m["comparison"] = {"candidate": candidate, "incumbent": incumbent, "panel": panel,
                       "min_gain": .01, "aggregate_tolerance": .02, "protected_tolerance": .05,
                       "cell_limit": .10, "alpha": .05, "bootstrap": 10000}
    write(root / "runs" / run_id / "manifest.json", m)
    return m


def compare(root, run_id):
    root = Path(root)
    m = read(root / "runs" / ident(run_id) / "manifest.json")
    if not m.get("comparison") or not schedule.complete(root, m):
        raise ValueError("A complete frozen comparison is required")
    c = m["comparison"]
    if digest({k: v for k, v in m["contract"].items()}) != m["contract_hash"]:
        raise ValueError("Frozen contract was edited")
    panel = c["panel"]
    seeds = sorted({g["seed"] for g in m["games"]})
    lookup = {}
    for g in m["games"]:
        r = read(root / "runs" / run_id / "games" / (g["id"] + ".json"))
        schedule.validate_result(g, r)
        aid = c["candidate"] if c["candidate"] in g["agents"] else c["incumbent"]
        seat = g["agents"].index(aid)
        lookup[aid, g["agents"][1-seat], g["seed"], seat] = float(r["outcome"] == f"win{seat}")
    d = np.array([[lookup[c["candidate"], e["agent"], s, seat] - lookup[c["incumbent"], e["agent"], s, seat]
                   for e in panel for seat in (0, 1)] for s in seeds])
    if len(seeds) < 32:
        raise ValueError("Too few seed blocks for promotion")
    vectors, labels, thresholds = [], [], []
    for kind in ("representative", "original", "stress"):
        families = sorted({e["family"] for e in panel if e["panel"] == kind})
        weights = np.zeros(len(panel)*2)
        for family in families:
            indices = [i for i, e in enumerate(panel) if e["panel"] == kind and e["family"] == family]
            for i in indices:
                weights[2*i:2*i+2] = 1/(len(families)*len(indices)*2)
        vectors.append(d @ weights)
        labels.append(kind)
        thresholds.append(0 if kind == "representative" else -c["aggregate_tolerance"])
    for i, e in enumerate(panel):
        if e.get("protected", False):
            for seat in (0, 1):
                vectors.append(d[:, 2*i+seat])
                labels.append(e["agent"] + f":seat{seat}")
                thresholds.append(-c["protected_tolerance"])
    x = np.stack(vectors, axis=1)
    rng = np.random.default_rng(610)
    samples = np.array([x[rng.integers(len(x), size=len(x))].mean(axis=0) for _ in range(c["bootstrap"])])
    # Bonferroni across regression checks; primary improvement has its own stated bound.
    lows = [float(np.quantile(samples[:, i], c["alpha"] if i == 0 else c["alpha"]/max(1, len(labels)-1))) for i in range(len(labels))]
    observed = x.mean(axis=0)
    checks = [{"metric": label, "change": float(observed[i]), "lower": lows[i], "threshold": thresholds[i],
               "pass": lows[i] > thresholds[i]} for i, label in enumerate(labels)]
    passed = all(v["pass"] for v in checks) and observed[0] >= c["min_gain"] and d.mean(axis=0).min() >= -c["cell_limit"]
    report = {"candidate": c["candidate"], "incumbent": c["incumbent"], "run": run_id, "pass": bool(passed),
              "checks": checks, "worst_cell": float(d.mean(axis=0).min()), "seed_count": len(seeds),
              "method": "paired seed bootstrap; approximate operational gate", "created": now()}
    write(root / "runs" / run_id / "gate.json", report)
    return report


def promote(root, run_id):
    result = compare(root, run_id)
    old = read(Path(root) / "champion.json")
    if not result["pass"] or old["agent"] != result["incumbent"]:
        raise ValueError("Gate failed or incumbent changed")
    write(Path(root) / "champion.json", {"agent": result["candidate"], "evidence": run_id, "updated": now()})
    return result


def certify(root, run_id, threshold=.9):
    root = Path(root)
    m = read(root / "runs" / ident(run_id) / "manifest.json")
    if m["kind"] != "confirmation" or m.get("comparison") or not schedule.complete(root, m):
        raise ValueError("Complete single-candidate confirmation required")
    if m["seed_count"] < 1024:
        raise ValueError("Certification requires the predeclared 1024+ seed budget")
    aid = m["candidate"]
    cells = {}
    for g in m["games"]:
        r = read(root / "runs" / run_id / "games" / (g["id"] + ".json"))
        schedule.validate_result(g, r)
        seat = g["agents"].index(aid)
        cells.setdefault((g["agents"][1-seat], seat), []).append(r["outcome"] == f"win{seat}")
    alpha = .05 / len(cells)
    bounds = {f"{o}:{s}": float(beta.ppf(alpha, sum(v), len(v)-sum(v)+1)) if any(v) else 0.
              for (o, s), v in cells.items()}
    opponents = {o for o, _ in cells}
    lower = {o: (bounds[f"{o}:0"] + bounds[f"{o}:1"])/2 for o in opponents}
    return {"pass": all(v > threshold for v in lower.values()), "lower_bounds": lower,
            "threshold": threshold, "scope": "this frozen version set and single confirmation attempt only"}
