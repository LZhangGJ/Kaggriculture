#!/usr/bin/env python3
"""Freeze one official 1.32.7 action/state trace for loss diagnosis."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = resolve(args.candidate)
    opponent = resolve(args.opponent)
    output = resolve(args.output)
    agents = [str(candidate), str(opponent)]
    if args.candidate_seat == 1:
        agents.reverse()

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 720, "seed": args.seed},
            debug=True,
        )
        env.run(agents)

    header = {
        "record_type": "header",
        "schema": "kaggriculture-official-single-trace-v1",
        "official_package_version": importlib.metadata.version("kaggle-environments"),
        "candidate": str(candidate),
        "candidate_sha256": sha256(candidate),
        "opponent": str(opponent),
        "opponent_sha256": sha256(opponent),
        "seed": args.seed,
        "candidate_seat": args.candidate_seat,
    }
    records: list[dict[str, Any]] = [header]
    for frame, states in enumerate(env.steps):
        obs = states[0].observation
        records.append({
            "record_type": "frame",
            "frame": frame,
            "step": int(obs.step),
            "day": int(obs.day),
            "hour": int(obs.hour),
            "status": [str(state.status) for state in states],
            "reward": [float(state.reward) for state in states],
            "actions": [safe(state.action) for state in states],
            "farms": safe(obs.farms),
            "private": [safe(state.observation.private) for state in states],
            "market": safe(obs.market),
            "town": safe(obs.town),
        })

    terminal = records[-1]
    valid = (
        len(env.steps) == 720
        and terminal["status"] == ["DONE", "DONE"]
        and importlib.metadata.version("kaggle-environments") == "1.32.7"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    text.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({
        "status": "PASS" if valid else "FAIL",
        "frames": len(env.steps),
        "terminal_reward": terminal["reward"],
        "output": str(output),
        "output_sha256": sha256(output),
    }, ensure_ascii=False, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
