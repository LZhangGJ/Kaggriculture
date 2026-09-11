#!/usr/bin/env python3
"""Freeze official Route37-vs-weak6 traces for JAX parity auditing."""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]
ROUTE37 = ROOT / "experiments/gold_adaptive_rule_v2/agents/fixed_gold_routes_market_aware_v1/route37_rank16_8c_4s_75l_market_aware/main.py"
POOL = ROOT / "experiments/expert_business_agent_v2/configs/route37_weak6_pool_v1.json"
OUTPUT = ROOT / "experiments/expert_business_agent_v2/official_eval/weak6_stepwise_parity"
RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/weak6_official_parity_traces_v1.json"
SEEDS = (159001, 159002, 159003, 159004)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def canonical_frame(frame_index: int, states: list[Any]) -> dict[str, Any]:
    obs0 = states[0].observation
    return {
        "frame": frame_index,
        "step": int(obs0.step),
        "day": int(obs0.day),
        "hour": int(obs0.hour),
        "status": [str(state.status) for state in states],
        "reward": [float(state.reward) for state in states],
        "actions": [json_safe(state.action) for state in states],
        "farms": json_safe(obs0.farms),
        "private": [json_safe(state.observation.private) for state in states],
        "market": json_safe(obs0.market),
        "town": json_safe(obs0.town),
    }


def write_trace(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def generate_one(task: tuple[str, str, int, int]) -> dict[str, Any]:
    name, opponent_path_text, seed, candidate_seat = task
    opponent_path = Path(opponent_path_text)
    agents = [str(ROUTE37), str(opponent_path)]
    if candidate_seat == 1:
        agents.reverse()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run(agents)
    frames = [canonical_frame(index, states) for index, states in enumerate(env.steps)]
    if len(frames) != 720 or frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"{name} seed={seed} seat={candidate_seat} invalid terminal trace")
    header = {
        "record_type": "header",
        "schema": "kaggriculture_weak6_official_stepwise_trace_v1",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "opponent": name,
        "opponent_path": str(opponent_path),
        "opponent_sha256": sha256(opponent_path),
        "seed": seed,
        "candidate_seat": candidate_seat,
        "route37_sha256": sha256(ROUTE37),
    }
    records = [header, *({"record_type": "frame", **frame} for frame in frames)]
    safe_name = name.replace("/", "_")
    output = OUTPUT / f"route37_{safe_name}_candidate_seat{candidate_seat}_seed{seed}.jsonl.gz"
    write_trace(output, records)
    return {
        "opponent": name,
        "opponent_path": str(opponent_path),
        "opponent_sha256": sha256(opponent_path),
        "seed": seed,
        "candidate_seat": candidate_seat,
        "path": str(output.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": sha256(output),
        "frames": len(frames),
        "terminal_reward": frames[-1]["reward"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    pool = json.loads(POOL.read_text(encoding="utf-8"))["opponents"]
    opponents = [(str(row["name"]), resolve(row["path"])) for row in pool]
    tasks = [
        (name, str(path), seed, seat)
        for name, path in opponents
        for seat in (0, 1)
        for seed in SEEDS
    ]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 18, len(tasks))) as executor:
        rows = list(executor.map(generate_one, tasks, chunksize=1))
    result = {
        "schema": "kaggriculture_weak6_official_parity_traces_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": list(SEEDS),
        "seat_swapped": True,
        "route37": str(ROUTE37),
        "route37_sha256": sha256(ROUTE37),
        "pool": str(POOL),
        "pool_sha256": sha256(POOL),
        "opponents": [
            {"name": name, "path": str(path), "sha256": sha256(path)}
            for name, path in opponents
        ],
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if result["ok"] else "FAIL",
        "opponents": len(opponents),
        "games": len(rows),
        "receipt": str(RECEIPT),
    }, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
