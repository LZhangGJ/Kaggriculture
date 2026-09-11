from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--opponent-seat", type=int, required=True)
    parser.add_argument("--from-step", type=int, default=548)
    parser.add_argument("--to-step", type=int, default=560)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package))
    import arena

    # FreshPublic temporarily changes cwd to the opponent directory, so resolve
    # every caller-supplied path and initialize our policy before opening it.
    package = args.package.resolve()
    binary = args.binary.resolve()
    config = json.loads((package / "policy/config.json").read_text())
    own = arena.codec.Agent(config=config, binary_path=binary)
    rival = arena.FreshPublic(args.opponent)
    env = arena.LocalGame(args.seed, arena.load_engine())
    records = []
    try:
        while not env.done and env.t < 719:
            observations = [env.observation(i) for i in (0, 1)]
            own_seat = 1 - args.opponent_seat
            view = observations[own_seat]
            action = own(view)
            rival_action = rival.call(
                observations[args.opponent_seat], copy.deepcopy(env.configuration)
            )
            issued = [None, None]
            issued[own_seat] = action
            issued[args.opponent_seat] = rival_action
            if args.from_step <= env.t <= args.to_step:
                private = view.get("private", view.get("player", {}))
                farm = view["farms"][own_seat]
                debug = own.debug()
                records.append(
                    {
                        "step": env.t,
                        "action": action,
                        "farmer": farm.get("farmer"),
                        "hands": farm.get("hands"),
                        "tiles_shape": [len(farm.get("tiles", []))]
                        + ([len(farm.get("tiles", [])[0])] if farm.get("tiles") else []),
                        "tile44": (
                            farm["tiles"][4][4]
                            if farm.get("tiles") and isinstance(farm["tiles"][0], list)
                            else (farm.get("tiles", [None] * 100)[44] if len(farm.get("tiles", [])) > 44 else None)
                        ),
                        "private": private,
                        "workflow": debug.get("workflow", {}),
                        "workflow_diagnostic": debug.get("workflow_diagnostic", {}),
                        "workflow_plans": debug.get("workflow_plans", []),
                        "workflow_witness": debug.get("workflow_witness", []),
                        "workflow_latest": debug.get("workflow_latest", []),
                    }
                )
            env.advance(issued)
    finally:
        own.close()
        rival.close()
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
