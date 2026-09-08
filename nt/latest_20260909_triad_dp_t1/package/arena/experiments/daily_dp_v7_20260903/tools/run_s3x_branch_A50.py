"""A50 finite one-day investment branch audit.

This is an offline causal audit, not a deployable oracle.  At each frozen day-start
state it changes the investment plan for one day, then returns to the same frozen
S3W baseline and lets the original live opponent continue from a deep copy.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import time
import zlib


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "native/build"))
import _dp7_native as native  # noqa: E402


OPPONENTS = (
    "pass",
    "g001",
    "g003",
    "boatlee_v29",
    "kaito_v58",
    "lynn_v5",
    "yhay81_six_day",
    "yhay81_three_day",
    "ecobot_v7",
)
DAYS = (0, 1, 3, 6, 9, 12, 18, 24)
SEEDS = tuple(range(20261401, 20261451))
THREADS = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()


def load_opponents(registry: dict) -> tuple[dict, dict]:
    mapping = {
        "pass": (0, None),
        "yhay81_six_day": (5, None),
        "yhay81_three_day": (6, None),
        "ecobot_v7": (7, None),
    }
    constructors = {
        "g001": (1, native.G001),
        "g003": (1, native.G001),
        "boatlee_v29": (2, native.BoatleeV29),
        "kaito_v58": (3, native.KaitoV58),
        "lynn_v5": (4, native.LynnV5),
    }
    hashes: dict[str, str] = {}
    for name, (kind, constructor) in constructors.items():
        entry = registry[name]
        asset = EXP / entry["asset"]
        source = EXP / entry["source"]
        assert sha(source) == entry["source_sha256"], name
        if "asset_sha256" in entry:
            assert sha(asset) == entry["asset_sha256"], name
        hashes[entry["asset"]] = sha(asset)
        hashes[entry["source"]] = sha(source)
        payload = json.loads(zlib.decompress(asset.read_bytes()))
        if name == "lynn_v5":
            for rel, expected in payload["source_hashes"].items():
                assert sha(source.parent / rel) == expected, (name, rel)
        mapping[name] = (kind, constructor(payload))
    assert tuple(mapping) != OPPONENTS  # insertion order is intentionally irrelevant
    assert set(mapping) == set(OPPONENTS)
    return mapping, hashes


def load_baseline() -> tuple[dict, str]:
    path = EXP / "receipts/s3v_idle_A50_v1/results.json"
    payload = json.loads(path.read_text(encoding="utf8"))
    rows = {
        (row["opponent"], int(row["seed"]), int(row["seat"])): row
        for row in payload["rows"]
        if row["variant"] == "old"
    }
    expected = {(opponent, seed, seat) for opponent in OPPONENTS for seed in SEEDS for seat in (0, 1)}
    assert set(rows) == expected
    return rows, sha(path)


def validate_rows(name: str, rows: list, baseline: dict) -> dict:
    expected_cases = {(seed, seat) for seed in SEEDS for seat in (0, 1)}
    actual_cases = {(int(row["seed"]), int(row["seat"])) for row in rows}
    assert actual_cases == expected_cases, name
    candidate_suffixes = 0
    transitions = 0
    candidate_counts: set[int] = set()
    for row in rows:
        seed, seat = int(row["seed"]), int(row["seat"])
        root = baseline[(name, seed, seat)]
        assert int(row["cash"]) == int(root["cash"]), (name, seed, seat, "cash")
        assert int(row["opponent_cash"]) == int(root["opponent_cash"]), (
            name,
            seed,
            seat,
            "opponent_cash",
        )
        nodes = row["nodes"]
        assert [int(node["day"]) for node in nodes] == list(DAYS)
        day0_keep = nodes[0]["choices"][0]
        assert day0_keep["family"] == "KEEP"
        assert int(day0_keep["cash"]) == int(root["cash"])
        assert int(day0_keep["opponent_cash"]) == int(root["opponent_cash"])
        for node in nodes:
            assert node["choices"][0]["family"] == "KEEP"
            assert node["keep_before_after"] is True
            # Full A50 deliberately omits the expensive reverse-order rerun;
            # pilot v2 is its pre-registered order-independence witness.
            candidate_counts.add(len(node["choices"]))
            candidate_suffixes += len(node["choices"])
        transitions += int(row["transitions"])
    return {
        "opponent": name,
        "games": len(rows),
        "independent_seeds": len(SEEDS),
        "nodes": sum(len(row["nodes"]) for row in rows),
        "candidate_suffixes": candidate_suffixes,
        "transitions": transitions,
        "candidate_counts": sorted(candidate_counts),
    }


def atomic_gzip_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf8", compresslevel=1) as handle:
        json.dump(value, handle, separators=(",", ":"))
    os.replace(tmp, path)


def main() -> None:
    out = EXP / "receipts/s3x_branch_A50_v1"
    out.mkdir(exist_ok=True)
    config_path = EXP / "profiles/s3w/configs.json"
    registry_path = EXP / "opponents/registry.json"
    config = json.loads(config_path.read_text(encoding="utf8"))["old"]
    registry = json.loads(registry_path.read_text(encoding="utf8"))["opponents"]
    opponents, input_hashes = load_opponents(registry)
    baseline, baseline_hash = load_baseline()
    pilot_path = EXP / "receipts/s3x_branch_pilot_v2/acceptance.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf8"))
    assert pilot["status"] == "PASS_BRANCH_PILOT_NOT_STRENGTH_RESULT"
    assert set(item["opponent"] for item in pilot["summary"]) == set(OPPONENTS)
    build = json.loads((EXP / "native/build/build_receipt.json").read_text(encoding="utf8"))
    plan = {
        "round": "S3X",
        "evidence_role": "finite one-day offline branch audit, not deployable oracle",
        "opponents": list(OPPONENTS),
        "days": list(DAYS),
        "seeds": list(SEEDS),
        "seats": [0, 1],
        "threads": THREADS,
        "config": config,
        "config_source": "profiles/s3w/configs.json:old",
        "config_sha256": sha(config_path),
        "registry_sha256": sha(registry_path),
        "baseline_source": "receipts/s3v_idle_A50_v1/results.json:variant=old",
        "baseline_sha256": baseline_hash,
        "pilot_acceptance_sha256": sha(pilot_path),
        "build": build,
        "input_hashes": input_hashes,
        "final_holdout_used": False,
    }
    plan["plan_sha256"] = canonical_hash(plan)
    plan_path = out / "plan.json"
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text(encoding="utf8"))
        assert old_plan == plan, "existing partial run has a different immutable plan"
    else:
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf8")

    summaries = []
    started = time.perf_counter()
    flat_seeds = [seed for seed in SEEDS for _seat in (0, 1)]
    flat_seats = [seat for _seed in SEEDS for seat in (0, 1)]
    for name in OPPONENTS:
        destination = out / f"{name}.json.gz"
        tic = time.perf_counter()
        if destination.exists():
            with gzip.open(destination, "rt", encoding="utf8") as handle:
                rows = json.load(handle)
            resumed = True
        else:
            kind, agent = opponents[name]
            rows = native.investment_branch_batch(
                flat_seeds,
                flat_seats,
                config,
                kind,
                agent,
                list(DAYS),
                THREADS,
                False,
            )
            atomic_gzip_json(destination, rows)
            resumed = False
        entry = validate_rows(name, rows, baseline)
        entry.update(seconds=time.perf_counter() - tic, resumed=resumed, sha256=sha(destination))
        summaries.append(entry)
        print(json.dumps(entry), flush=True)
        progress = {
            "status": "RUNNING",
            "plan_sha256": plan["plan_sha256"],
            "completed": [item["opponent"] for item in summaries],
            "summary": summaries,
        }
        (out / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf8")

    acceptance = {
        "status": "COMPLETE_FINITE_BRANCH_AUDIT_NOT_DEPLOYABLE_ORACLE",
        "plan": plan,
        "summary": summaries,
        "total_games": sum(item["games"] for item in summaries),
        "total_candidate_suffixes": sum(item["candidate_suffixes"] for item in summaries),
        "total_transitions": sum(item["transitions"] for item in summaries),
        "seconds_this_invocation": time.perf_counter() - started,
        "all_keep_exact_against_s3v_old": True,
        "pilot_reverse_order_isolation_passed": True,
        "full_goal_complete": False,
        "caveat": (
            "True future and cloned opponent continuation are used only as offline labels. "
            "Every branch changes one day and returns to the baseline controller; this is "
            "not a deployable oracle, multistage ceiling, or final holdout result."
        ),
    }
    (out / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf8")
    print(json.dumps({key: acceptance[key] for key in ("status", "total_games", "total_candidate_suffixes", "total_transitions", "seconds_this_invocation")}), flush=True)


if __name__ == "__main__":
    main()
