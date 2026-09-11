#!/usr/bin/env python3
"""Freeze official 1.32.7 candidate-vs-opponent traces for parity audits."""

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
DEFAULT_CANDIDATE = ROOT / "experiments/gold_adaptive_rule_v2/agents/fixed_gold_routes_market_aware_v1/route37_rank16_8c_4s_75l_market_aware/main.py"


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


def generate_one(task: tuple[str, str, str, str, int, int]) -> dict[str, Any]:
    name, opponent_text, candidate_text, output_text, seed, candidate_seat = task
    opponent = Path(opponent_text)
    candidate = Path(candidate_text)
    output_dir = Path(output_text)
    agents = [str(candidate), str(opponent)]
    if candidate_seat == 1:
        agents.reverse()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run(agents)
    frames = [canonical_frame(index, states) for index, states in enumerate(env.steps)]
    if len(frames) != 720 or frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"{name} seed={seed} seat={candidate_seat} invalid terminal trace")
    records = [{
        "record_type": "header",
        "schema": "kaggriculture_official_stepwise_trace_v2",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "opponent": name,
        "opponent_path": str(opponent),
        "opponent_sha256": sha256(opponent),
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_path": str(candidate),
        "candidate_sha256": sha256(candidate),
    }, *({"record_type": "frame", **frame} for frame in frames)]
    safe_name = name.replace("/", "_")
    path = output_dir / f"candidate_{safe_name}_seat{candidate_seat}_seed{seed}.jsonl.gz"
    write_trace(path, records)
    return {
        "opponent": name,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": sha256(path),
        "frames": len(frames),
        "terminal_reward": frames[-1]["reward"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=163001)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()
    pool_path = resolve(args.pool)
    candidate = resolve(args.candidate)
    output = resolve(args.output)
    receipt_path = resolve(args.receipt)
    opponents = [
        (str(row["name"]), resolve(row["path"]))
        for row in json.loads(pool_path.read_text(encoding="utf-8"))["opponents"]
    ]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    tasks = [
        (name, str(path), str(candidate), str(output), seed, seat)
        for name, path in opponents
        for seat in (0, 1)
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 18, len(tasks))) as executor:
        rows = list(executor.map(generate_one, tasks, chunksize=1))
    result = {
        "schema": "kaggriculture_official_stepwise_parity_traces_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": seeds,
        "seat_swapped": True,
        "candidate": str(candidate),
        "candidate_sha256": sha256(candidate),
        "pool": str(pool_path),
        "pool_sha256": sha256(pool_path),
        "opponents": [
            {"name": name, "path": str(path), "sha256": sha256(path)}
            for name, path in opponents
        ],
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if result["ok"] else "FAIL",
        "opponents": len(opponents),
        "games": len(rows),
        "receipt": str(receipt_path),
    }, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
