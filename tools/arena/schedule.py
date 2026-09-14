"""Frozen, resumable game plans and seed reservations."""
from __future__ import annotations

import itertools
import random
import secrets
from pathlib import Path

from .store import digest, event, ident, now, read, records, write


def reserve(root, owner, count, supplied=None):
    path = Path(root) / "private/seeds.json"
    ledger = read(path, {})
    previous = [int(k) for k, v in ledger.items() if v == owner]
    if previous:
        if len(previous) != count:
            raise ValueError("Reservation size changed")
        return sorted(previous)
    values = list(supplied) if supplied is not None else []
    if supplied is None:
        while len(values) < count:
            s = secrets.randbelow(2**31 - 1)
            if str(s) not in ledger and s not in values:
                values.append(s)
    if len(values) != count or len(set(values)) != count or any(str(s) in ledger for s in values):
        raise ValueError("Seeds already exposed/reserved or duplicated")
    if any(type(s) is not int or not 0 <= s < 2**31 for s in values):
        raise ValueError("Invalid seed")
    ledger.update({str(s): owner for s in values})
    write(path, ledger)
    return sorted(values)


def plan(root, run_id, kind="daily", count=None, candidate=None, references=None, supplied=None):
    root, run_id = Path(root), ident(run_id)
    folder = root / "runs" / run_id
    existing = read(folder / "manifest.json")
    if existing:
        if existing["kind"] != kind or (candidate and existing.get("candidate") != candidate):
            raise ValueError("Run ID collision")
        if count is not None and existing["seed_count"] != count:
            raise ValueError("Cannot change a frozen run's seed budget")
        if references is not None and set(existing["agents"]) != {candidate, *references}:
            raise ValueError("Cannot change frozen references")
        return existing
    cfg = read(root / "config.json")
    roster = read(root / "roster.json", [])
    if kind == "daily":
        for p in (root / "runs").glob("*/manifest.json"):
            old = read(p)
            if old["kind"] == "daily" and not complete(root, old):
                raise ValueError("Previous daily tournament incomplete; resume it")
        ids = [e["agent"] for e in roster]
        pairs = list(itertools.combinations(sorted(ids), 2))
        count = count or cfg["daily_seeds"]
    elif kind in ("placement", "confirmation", "topup", "benchmark"):
        if not candidate or not references or candidate in references or len(set(references)) != len(references):
            raise ValueError("Distinct candidate and references required")
        ids = [candidate, *references]
        pairs = [(candidate, r) for r in sorted(references)]
        count = count or cfg["placement_seeds"]
    else:
        raise ValueError("Unknown run kind")
    if len(ids) < 2 or type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("Invalid roster/sample size")
    agents = {}
    for aid in ids:
        a = read(root / "agents" / (ident(aid) + ".json"))
        if not a or not a["build_verified"]:
            raise ValueError("Agent is not sandbox verified")
        agents[aid] = a
    seeds = reserve(root, run_id, count, supplied)
    # Include sandbox identity/limits: a different execution contract is different evidence.
    contract = {**cfg["contract"], "image": cfg["image"], "turn_timeout_seconds": cfg["turn_timeout_seconds"],
                "game_timeout_seconds": cfg["game_timeout_seconds"], "cpus_per_agent": cfg["cpus_per_agent"], "memory_mb": cfg["memory_mb"]}
    ch = digest(contract)
    games = []
    for seed, (a, b), seat in itertools.product(seeds, pairs, (0, 1)):
        ordered = [a, b] if seat == 0 else [b, a]
        g = {"agents": ordered, "seed": seed, "agent_seed": seed, "contract": ch}
        games.append({**g, "id": digest(g)})
    random.Random(digest(run_id)).shuffle(games)
    manifest = {"id": run_id, "kind": kind, "created": now(), "contract": contract,
                "contract_hash": ch, "agents": agents, "roster": roster, "candidate": candidate,
                "seed_count": count, "games": games}
    write(folder / "manifest.json", manifest)
    event(root, "run_planned", run_id, {"kind": kind, "games": len(games)})
    return manifest


def complete(root, manifest):
    return all(read(Path(root) / "runs" / manifest["id"] / "games" / (g["id"] + ".json"), {}).get("resolved")
               for g in manifest["games"])


def validate_result(game, result):
    import math
    if result.get("game") != game["id"] or result.get("agents") != game["agents"]:
        raise ValueError("Result identity mismatch")
    if type(result.get("resolved")) is not bool:
        raise ValueError("resolved must be boolean")
    if not result.get("resolved"):
        return
    if result.get("outcome") not in ("win0", "win1", "draw"):
        raise ValueError("Invalid outcome")
    if result.get("reason") not in ("terminal", "agent_failure"):
        raise ValueError("Resolved game needs terminal/forfeit evidence")
    cash = result.get("cash")
    if cash is not None and (len(cash) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in cash)):
        raise ValueError("Invalid cash")
    if result["reason"] == "terminal" and (not result.get("terminal") or cash is None):
        raise ValueError("Terminal evidence missing")
    if result["reason"] == "terminal":
        expected = "draw" if cash[0] == cash[1] else ("win0" if cash[0] > cash[1] else "win1")
        if expected != result["outcome"]:
            raise ValueError("Outcome conflicts with terminal cash")


def save_result(root, manifest, game, result):
    validate_result(game, result)
    path = Path(root) / "runs" / manifest["id"] / "games" / (game["id"] + ".json")
    old = read(path)
    if old and old.get("resolved"):
        return False
    write(path, result)
    return True


def placement_queue(root):
    """One queue per author; oldest ready author rotates between ticks."""
    root = Path(root)
    grouped = {}
    for a in sorted(records(root, "agents"), key=lambda x: x["created"]):
        if a["build_verified"] and not (root / "runs" / ("placement-" + a["id"][:20]) / "manifest.json").exists():
            grouped.setdefault(a["manifest"]["author"], []).append(a)
    authors = sorted(grouped)
    cursor = read(root / "placement_cursor.json", {}).get("author", "")
    authors = [a for a in authors if a > cursor] + [a for a in authors if a <= cursor]
    return [grouped[a][0] for a in authors]


def create_placements(root):
    root = Path(root)
    cfg = read(root / "config.json")
    refs = cfg["placement_references"]
    if len(refs) != 6:
        return []
    result = []
    for agent in placement_queue(root):
        opponents = [r for r in refs if r != agent["id"]]
        if not opponents:
            continue
        run = plan(root, "placement-" + agent["id"][:20], "placement", candidate=agent["id"], references=opponents)
        result.append(run)
        write(root / "placement_cursor.json", {"author": agent["manifest"]["author"]})
    return result


def topup(root, run_id, source, candidate, opponent, target):
    root = Path(root)
    baseline = read(root / "runs" / ident(source) / "manifest.json")
    if not baseline or not complete(root, baseline):
        raise ValueError("Completed source run required")
    pair = {candidate, opponent}
    seeds = {g["seed"] for g in baseline["games"] if set(g["agents"]) == pair}
    if not seeds or target <= len(seeds):
        raise ValueError("Target must exceed existing seed coverage")
    m = plan(root, run_id, "topup", target-len(seeds), candidate, [opponent])
    if m["contract_hash"] != baseline["contract_hash"]:
        raise ValueError("Top-up contract differs from source; use a separate evaluation")
    m["topup"] = {"source": source, "existing_seeds":len(seeds), "target_seeds":target}
    write(root / "runs" / run_id / "manifest.json", m)
    return m
