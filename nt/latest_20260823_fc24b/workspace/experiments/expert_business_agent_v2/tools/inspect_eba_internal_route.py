#!/usr/bin/env python3
"""Run one official game and expose EBA8's frozen public route decisions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from kaggle_environments import make

    candidate = load(resolve(args.candidate), "_eba_internal_candidate")
    agents = [candidate.agent, str(resolve(args.opponent))]
    if args.seat == 1:
        agents.reverse()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": args.seed}, debug=False)
    env.run(agents)

    v8 = getattr(candidate, "_EBA_V8", {})
    prt = v8.get("_PRT", {}) if isinstance(v8, dict) else {}
    payload = {
        "schema": "kaggriculture-eba-internal-route-v1",
        "seed": args.seed,
        "candidate_seat": args.seat,
        "final_rewards": [float(actor.reward) for actor in env.steps[-1]],
        "final_status": [str(actor.status) for actor in env.steps[-1]],
        "family_state": v8.get("_STATE", {}).get(args.seat, {}) if isinstance(v8, dict) else {},
        "prt_route_state": prt.get("_CGR_STATE", {}).get(args.seat, {}) if isinstance(prt, dict) else {},
        "diagnostics": candidate.expert_business_diagnostics(args.seat)
        if hasattr(candidate, "expert_business_diagnostics")
        else {},
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = resolve(str(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
