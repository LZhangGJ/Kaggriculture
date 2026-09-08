"""C++-only match loop with explicit frozen opponent identities.

Pending ports fail closed; no substituting PASS, old versions or action traces.
All panels here are development evidence, not final acceptance.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
import time
import zlib

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "native/build"))
import _dp7_native as native


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opponents", required=True)
    p.add_argument("--configs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert a.count > 0 and a.repeat > 0 and 1 <= a.threads <= 16
    registry_path = EXP / "opponents/registry.json"
    registry = json.loads(registry_path.read_text())
    keys = a.opponents.split(",")
    configs = json.loads(Path(a.configs).read_text())
    configs = {k: configs[k] for k in a.labels.split(",")}
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    for rel, expected in build["source_hashes"].items():
        assert digest(EXP / rel) == expected, (rel, "stale build")
    binaries = list((EXP / "native/build").glob("_dp7_native*.so"))
    assert len(binaries) == 1 and digest(binaries[0]) == build["binary_sha256"]
    loaded = {}
    identities = {}
    for key in keys:
        if key == "pass":
            loaded[key] = None
            identities[key] = {"label": "PASS", "runtime": "native_pass"}
            continue
        entry = registry["opponents"][key]
        if entry["runtime"] not in ("searched_route_native", "fieldbook_native", "three_day_native", "boatlee_v29_native", "kaito_v58_native", "lynn_v5_native", "ecobot_v7_native"):
            raise ValueError(f"{key} has not completed native integration; refusing to substitute an opponent")
        assert digest(EXP / entry["source"]) == entry["source_sha256"]
        if entry["runtime"] in ("fieldbook_native", "three_day_native", "boatlee_v29_native", "kaito_v58_native", "lynn_v5_native", "ecobot_v7_native"):
            for field in ("initial_parity_receipt", "isolation_receipt"):
                check = json.loads((EXP / entry[field]).read_text())
                assert check["status"] == "PASS"
                assert check["build"]["binary_sha256"] == build["binary_sha256"], f"recheck updated {key} build"
            if entry["runtime"] in ("fieldbook_native", "three_day_native", "ecobot_v7_native"):
                loaded[key] = None
                identities[key] = entry
                continue
        if "asset_sha256" in entry:assert digest(EXP / entry["asset"]) == entry["asset_sha256"]
        payload = json.loads(zlib.decompress((EXP / entry["asset"]).read_bytes()))
        assert payload["source_sha256"] == entry["source_sha256"]
        if entry["runtime"] == "lynn_v5_native":
            package = (EXP / entry["source"]).parent
            for rel,h in payload["source_hashes"].items():assert digest(package / rel)==h
        loaded[key] = (native.LynnV5(payload) if entry["runtime"] == "lynn_v5_native" else
                       native.KaitoV58(payload) if entry["runtime"] == "kaito_v58_native" else
                       native.BoatleeV29(payload) if entry["runtime"] == "boatlee_v29_native" else native.G001(payload))
        identities[key] = dict(entry, asset_sha256=digest(EXP / entry["asset"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    unique_tasks = [(seed, seat) for seed in range(a.seed, a.seed + a.count) for seat in (0, 1)]
    tasks = unique_tasks * a.repeat
    receipt = dict(status="RUNNING", evidence_role="development_only", args=vars(a),
                   configurations=configs, build=build, identities=identities,
                   registry_sha256=digest(registry_path),
                   pending_required_opponents=[k for k in registry["required_opponents"] if k not in keys],
                   summary={}, rows=[])
    (out / "plan.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for label, config in configs.items():
        for key, opponent in loaded.items():
            start = time.perf_counter()
            if identities[key]["runtime"] == "fieldbook_native":
                rows = native.batch_fieldbook([t[0] for t in tasks], [t[1] for t in tasks], config, a.threads)
            elif identities[key]["runtime"] == "three_day_native":
                rows = native.batch_three_day([t[0] for t in tasks], [t[1] for t in tasks], config, a.threads)
            elif identities[key]["runtime"] == "ecobot_v7_native":
                rows = native.batch_ecobot([t[0] for t in tasks], [t[1] for t in tasks], config, a.threads)
            elif identities[key]["runtime"] == "boatlee_v29_native":
                rows = native.batch_boatlee([t[0] for t in tasks], [t[1] for t in tasks], config, opponent, a.threads)
            elif identities[key]["runtime"] == "kaito_v58_native":
                rows = native.batch_kaito([t[0] for t in tasks], [t[1] for t in tasks], config, opponent, a.threads)
            elif identities[key]["runtime"] == "lynn_v5_native":
                rows = native.batch_lynn([t[0] for t in tasks], [t[1] for t in tasks], config, opponent, a.threads)
            else:
                rows = native.batch([t[0] for t in tasks], [t[1] for t in tasks], config, opponent, a.threads)
            elapsed = time.perf_counter() - start
            assert len(rows) == len(tasks) and all(r["steps"] == 719 for r in rows)
            seen = {}
            for row in rows:
                if "g001_switched" in row:
                    row["opponent_switched"] = row.pop("g001_switched")
                k = row["seed"], row["seat"]
                signature = (row["cash"], row["opponent_cash"], row["opponent_switched"],
                             tuple(row.get("selected_segments", [])), tuple(row["overflow"]),
                             json.dumps(row.get("opponent_state", {}), sort_keys=True), row.get("weed_repair_frames"),
                             tuple(row.get("selected_route_frames", [])), tuple(row.get("preempt_units_by_route", [])),
                             tuple(row.get("weed_collisions_by_route", [])), row.get("selected_route"))
                assert k not in seen or seen[k] == signature, ("cross-game contamination", k)
                seen[k] = signature
                row.update(opponent=key, variant=label)
            unique = rows[:len(unique_tasks)]
            wins = sum(r["win"] for r in unique)
            ties = sum(r["margin"] == 0 for r in unique)
            # Seats sharing a seed are correlated; use seed-clustered SE.
            scores = [(int(unique[2*i]["win"]) + int(unique[2*i+1]["win"])) / 2 for i in range(a.count)]
            se = statistics.stdev(scores) / math.sqrt(a.count) if a.count > 1 else None
            rate = wins / len(unique)
            summary = dict(games=len(unique), independent_seeds=a.count, wins=wins, ties=ties,
                           win_rate=rate, mean_cash=statistics.fmean(r["cash"] for r in unique),
                           mean_opponent_cash=statistics.fmean(r["opponent_cash"] for r in unique),
                           mean_margin=statistics.fmean(r["margin"] for r in unique),
                           switched_games=sum(r["opponent_switched"] for r in unique) if all(r["opponent_switched"] is not None for r in unique) else None,
                           seed_cluster_se=se,
                           approximate_seed_cluster_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)] if se is not None and se > 0 else None,
                           conservative_seed_hoeffding_95pct_bound=[max(0,rate-math.sqrt(math.log(40)/(2*a.count))),min(1,rate+math.sqrt(math.log(40)/(2*a.count)))],
                           interval_caveat="Normal approximation with seed clusters; undefined at zero empirical variance. Conservative bound assumes independent random-seed clusters. Neither is a leaderboard guarantee.",
                           total_executed_games=len(rows), wall_seconds=elapsed, games_per_second=len(rows)/elapsed)
            if "selected_segments" in unique[0]:
                summary["selected_segment_counts"] = dict(collections.Counter(
                    ",".join(map(str, r["selected_segments"])) for r in unique))
            if "selected_route_frames" in unique[0]:
                summary["selected_route_game_counts"] = [sum(r["selected_route_frames"][i]>0 for r in unique) for i in range(10)]
            if "selected_route" in unique[0]:
                summary["selected_route_counts"] = dict(collections.Counter(str(r["selected_route"]) for r in unique))
            if "opponent_state" in unique[0] and "near_mirror" in unique[0]["opponent_state"]:
                summary["mirror_latched_games"] = sum(r["opponent_state"]["near_mirror"]==1 for r in unique)
                summary["weed_repair_games"] = sum(r["weed_repair_frames"]>0 for r in unique)
                summary["mean_extra_sales_requested"] = {i:statistics.fmean(r["opponent_state"]["added"][i] for r in unique)
                                                         for i in unique[0]["opponent_state"]["added"]}
            if identities[key]["runtime"] == "lynn_v5_native":
                summary["animal_assignments"] = dict(collections.Counter(
                    json.dumps(r["opponent_state"]["assignments"],sort_keys=True) for r in unique))
                summary["weed_repair_games"] = sum(r["weed_repair_frames"]>0 for r in unique)
                summary["delivery_activation_games"] = [sum(bool(r["opponent_state"]["delivery"][i]["intervention_steps"]) for r in unique) for i in range(3)]
            receipt["summary"].setdefault(label, {})[key] = summary
            receipt["rows"].extend(rows)
            (out / "results.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            print(json.dumps({"label":label, "opponent":key, **summary}), flush=True)
    receipt["status"] = "COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE"
    (out / "results.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
