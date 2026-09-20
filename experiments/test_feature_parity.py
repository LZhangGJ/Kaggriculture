#!/usr/bin/env python3
"""Bit-level parity check between the two 147-dim route-switch encoders.

Training states come from the native ``NativeTeammateExecutor::features_at``
(C++), while serving computes ``route_switch_features.route_switch_vector``
(Python).  The names match, but nothing has ever compared the values.

Method: ask the native executor for a full joint action trace of one
route-vs-route game (``play(capture_trace=True)``), replay that trace through
``FastEnv`` -- which ``test_differential.py`` already proves matches
``kaggle_environments`` step for step -- and feed the replayed observations into
the Python encoder exactly the way the online controller does.  The resulting
vector must match the native one for the same
``(route0, route1, seed, checkpoint, player, feature_route)``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ.setdefault("KAGG_META_NUMPY_ONLY", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fast_kaggriculture import Config, FastEnv  # noqa: E402
from meta_agent.src.native_teammate_executor import NativeTeammateBundle  # noqa: E402
from meta_agent.src.route_switch_features import (  # noqa: E402
    RouteSwitchHistory,
    route_switch_feature_names,
    route_switch_vector,
)


def replay_and_encode(executor, family_tapes, names, route0, route1, seed, checkpoint, player, feature_route):
    """Replay the native trace in FastEnv and encode with the Python encoder."""
    trace = executor.play(route0, route1, seed, -1, -1, -1, -1, True)["trace"]
    if len(trace) < checkpoint:
        raise ValueError(f"trace has {len(trace)} steps, need {checkpoint}")
    env = FastEnv(Config(), seed)
    history = RouteSwitchHistory()
    observation = None
    for step in range(checkpoint + 1):
        observation = env.observation(player)
        observation = json.loads(json.dumps(dict(observation)))
        observation["player"] = player
        observation.setdefault("step", step)
        if observation.get("step") is None:
            observation["step"] = observation.get("day", 0) * 24 + observation.get("hour", 0)
        history.update(observation)
        if step == checkpoint:
            break
        joint = trace[step]
        env.step([json.loads(json.dumps(dict(joint[0]))), json.loads(json.dumps(dict(joint[1])))])
    # Serving goes family -> route_id -> tape; the tape key is the route_id.
    route_actions = family_tapes.get(names[feature_route]) if feature_route < len(names) else None
    return np.asarray(route_switch_vector(observation, history, route_actions), dtype=np.float64), observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", default="G001,G275,G379")
    parser.add_argument("--checkpoints", default="144,168,216")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--start", type=int, default=2609600000)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json",
    )
    families = list(bundle.families)
    from meta_agent.src.teammate_expanded_routes import load_action_tapes

    tapes = load_action_tapes(ROOT / "agent/route_actions.json.zlib")
    metadata = json.loads((ROOT / "agent/route_library.json").read_text())
    by_family = {str(r["family"]): str(r["route_id"]) for r in metadata["opponent_routes"]}
    family_tapes = {fam: tapes[rid] for fam, rid in by_family.items() if rid in tapes}
    missing = [fam for fam in families if fam not in family_tapes]
    if missing:
        print(f"warning: no tape for {len(missing)} families, e.g. {missing[:3]}", file=sys.stderr)
    feature_names = list(route_switch_feature_names())
    openings = [value.strip() for value in args.openings.split(",") if value.strip()]
    checkpoints = [int(value) for value in args.checkpoints.split(",") if value.strip()]

    cases = []
    errors = []
    for opening in openings:
        if opening not in families:
            errors.append(f"opening {opening} is not a native route")
            continue
        route0 = bundle.index(opening)
        for seed in range(args.start, args.start + args.seeds):
            for checkpoint in checkpoints:
                for player in (0, 1):
                    for opponent in openings:
                        if opponent not in families:
                            continue
                        route1 = bundle.index(opponent)
                        try:
                            native = np.asarray(
                                bundle.executor.features_at(
                                    route0, route1, seed, checkpoint, player, route0
                                ),
                                dtype=np.float64,
                            )
                            python_vector, observation = replay_and_encode(
                                bundle.executor, family_tapes, families, route0, route1,
                                seed, checkpoint, player, route0,
                            )
                        except Exception as exc:  # noqa: BLE001 - reported
                            errors.append(f"{opening} vs {opponent} seed {seed} ckpt {checkpoint} "
                                          f"player {player}: {exc!r}")
                            continue
                        difference = np.abs(native - python_vector)
                        scale = np.maximum(np.abs(native), 1.0)
                        cases.append({
                            "opening": opening, "opponent": opponent, "seed": seed,
                            "checkpoint": checkpoint, "player": player,
                            "max_abs_diff": float(difference.max()),
                            "max_rel_diff": float((difference / scale).max()),
                            "worst_feature": feature_names[int(np.argmax(difference))],
                            "worst_native": float(native[int(np.argmax(difference))]),
                            "worst_python": float(python_vector[int(np.argmax(difference))]),
                            "match": bool(np.allclose(native, python_vector,
                                                      atol=args.atol, rtol=args.rtol)),
                        })

    matched = sum(1 for case in cases if case["match"])
    worst = sorted(cases, key=lambda case: -case["max_abs_diff"])[:5]
    result = {
        "comparisons": len(cases), "matched": matched, "failed": len(cases) - matched,
        "atol": args.atol, "rtol": args.rtol,
        "max_abs_diff": max((c["max_abs_diff"] for c in cases), default=None),
        "errors": errors, "worst": worst, "cases": cases,
        "status": "PASS" if cases and not errors and matched == len(cases) else "FAIL",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
