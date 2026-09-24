#!/usr/bin/env python3
"""At the first warm handoff, change one public shop and compare R1 plans."""

import argparse
import copy
import ctypes
import gzip
import json
from collections import Counter
from pathlib import Path

from agent import main as production


def targets(agent, observation):
    packed = production.policy._pack(observation)
    if agent.lib.td_activate_external(agent.handle, packed, len(packed)):
        raise RuntimeError(agent.debug())
    agent.external = False
    rows = [row for row in agent.prepare_candidates(observation)
            if not row.get("diagnostic")]
    if not rows:
        raise RuntimeError("R1 produced no normal candidate")
    winner = max(rows, key=lambda row: row["score"])
    agent.install_candidate(winner["index"])
    output = (ctypes.c_int32 * 100)()
    target_export = agent.lib.td_student_live_targets
    target_export.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                              ctypes.c_size_t]
    target_export.restype = ctypes.c_int
    if target_export(agent.handle, output, 100) != 100:
        raise RuntimeError("R1 target export failed")
    return {"candidate_id": winner["id"], "score": winner["score"],
            "normal_candidates": len(rows),
            "targets": dict(sorted(Counter(int(x) for x in output if x >= 0).items()))}


def audit(path: Path, binary: Path):
    agents = [production.policy.Agent(binary_path=binary) for _ in range(2)]
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            metadata = json.loads(next(source))
            if metadata.get("format") != "kaggriculture-bc-v1" or metadata.get("diagnostic"):
                raise ValueError("expected an ordinary complete BC trajectory")
            seat = int(metadata["seat"])
            for line in source:
                frame = json.loads(line)
                if frame.get("type") != "step":
                    continue
                step = int(frame["step"])
                observation = {**frame["public"], "private": frame["private"][seat],
                               "player": seat}
                if step < 288:
                    for agent in agents:
                        agent.observe_external(observation, frame["actions"][seat])
                    continue
                if step != 288:
                    raise ValueError("trajectory skipped first warm handoff")
                changed = copy.deepcopy(observation)
                shops = changed["town"]["unlocked_shops"]
                if not shops:
                    raise ValueError("no unlocked shop at first warm handoff")
                before = shops[0]
                shops[0] = "YARN_STORE" if before == "PET_CAFE" else "PET_CAFE"
                return {"trajectory": str(path), "seed": metadata["seed"],
                        "opponent": metadata["bot"], "seat": seat,
                        "shop_before": before, "shop_after": shops[0],
                        "original": targets(agents[0], observation),
                        "counterfactual": targets(agents[1], changed)}
        raise ValueError("trajectory has no step288")
    finally:
        for agent in agents:
            agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", nargs="+", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    rows = [audit(path, args.binary) for path in args.trajectories]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "first_warm_handoff_r1_one_shop",
                                       "binary": str(args.binary), "rows": rows},
                                      indent=2) + "\n")
    print(json.dumps({"states": len(rows),
                      "winner_changed": sum(row["original"]["candidate_id"] !=
                                            row["counterfactual"]["candidate_id"]
                                            for row in rows),
                      "target_changed": sum(row["original"]["targets"] !=
                                            row["counterfactual"]["targets"]
                                            for row in rows)}))


if __name__ == "__main__":
    main()
