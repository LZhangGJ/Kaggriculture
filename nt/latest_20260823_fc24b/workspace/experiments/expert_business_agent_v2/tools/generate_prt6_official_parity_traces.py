#!/usr/bin/env python3
"""Generate frozen official Route37-vs-PRT6 traces for JAX parity."""

from __future__ import annotations

import contextlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]
ROUTE37 = ROOT / "experiments/gold_adaptive_rule_v2/agents/fixed_gold_routes_market_aware_v1/route37_rank16_8c_4s_75l_market_aware/main.py"
PRT6 = ROOT / "experiments/gold_adaptive_rule_v2/agents/pure_public_route_tree_v6/main.py"
OUTPUT = ROOT / "experiments/expert_business_agent_v2/official_eval/prt6_stepwise_parity"
RECEIPT = ROOT / "experiments/expert_business_agent_v2/receipts/prt6_official_parity_traces_v1.json"
SEEDS = (157001, 157002, 157003, 157004)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_agent(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


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


def generate_one(task: tuple[int, int]) -> dict[str, Any]:
    seed, candidate_seat = task
    route37 = load_agent(ROUTE37, f"route37_{seed}_{candidate_seat}")
    prt6 = load_agent(PRT6, f"prt6_{seed}_{candidate_seat}")
    agents = [route37, prt6] if candidate_seat == 0 else [prt6, route37]
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": seed},
            debug=True,
        )
        env.run(agents)
    frames = [canonical_frame(index, states) for index, states in enumerate(env.steps)]
    if len(frames) != 720 or frames[-1]["status"] != ["DONE", "DONE"]:
        raise RuntimeError(f"seed={seed} seat={candidate_seat} invalid terminal trace")
    header = {
        "record_type": "header",
        "schema": "kaggriculture_prt6_official_stepwise_trace_v1",
        "package": "kaggle-environments",
        "package_version": importlib.metadata.version("kaggle-environments"),
        "seed": seed,
        "candidate_seat": candidate_seat,
        "route37_sha256": sha256(ROUTE37),
        "prt6_sha256": sha256(PRT6),
    }
    records = [header, *({"record_type": "frame", **frame} for frame in frames)]
    output = OUTPUT / f"route37_prt6_candidate_seat{candidate_seat}_seed{seed}.jsonl.gz"
    write_trace(output, records)
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "path": str(output.relative_to(ROOT)).replace("\\", "/"),
        "file_sha256": sha256(output),
        "frames": len(frames),
        "terminal_reward": frames[-1]["reward"],
    }


def main() -> int:
    tasks = [(seed, seat) for seat in (0, 1) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(generate_one, tasks))
    result = {
        "schema": "kaggriculture_prt6_official_parity_traces_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "seeds": list(SEEDS),
        "seat_swapped": True,
        "route37": str(ROUTE37),
        "route37_sha256": sha256(ROUTE37),
        "prt6": str(PRT6),
        "prt6_sha256": sha256(PRT6),
        "traces": rows,
        "ok": all(row["frames"] == 720 for row in rows),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
