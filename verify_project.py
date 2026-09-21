#!/usr/bin/env python3
"""Fast structural check for the consolidated replay-switch project."""
import ctypes
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    deployment = json.loads((ROOT / "agent/replay_deployment.json").read_text())
    for name, expected in deployment["sha256"].items():
        actual = hashlib.sha256((ROOT / "agent" / name).read_bytes()).hexdigest()
        assert actual == expected, (name, actual, expected)
    assert len(list((ROOT / "data/replays/raw").glob("episode-*-replay.json"))) == 609
    corpus = json.loads((ROOT / "data/replays/corpus-selection-609.json").read_text())
    assert corpus["selected_raw_replays"] == 609
    assert len([p for p in (ROOT / "opponents").iterdir() if p.is_dir()]) == 7
    fine = json.loads((ROOT / "data/artifacts/switch-fine-26x128-summary.json").read_text())
    assert fine["games"] == 24_460_800 and fine["feature_count"] == 147
    deployed = json.loads((ROOT / "agent/route_policy.json").read_text())
    shallow = json.loads((ROOT / "data/artifacts/trees-shallow-d2-6.json").read_text())
    assert deployed.pop("target_fallbacks", None) == {"G114": "G396"}
    deployed["sources"] = shallow["sources"]
    assert deployed == shallow
    result = json.loads((ROOT / "experiments/results/r1-vs-replay-tree-warm-handoff12-public7-4seed-v1.json").read_text())
    assert sum(x["wins"] for x in result["summary"]["baseline"].values()) == 36
    assert sum(x["wins"] for x in result["summary"]["candidate"].values()) == 30
    # Headline result: 64 non-overlapping seeds, both seats, 7 strong bots,
    # 896 games per arm.  The 4-seed pair asserted above is retained only as a
    # file-integrity check of a historical artifact; its conclusion
    # ("warm handoff does not beat pure R1") was retracted, see HANDOFF_ZH.md.
    def totals(path, label):
        summary = json.loads(path.read_text())["summary"][label]
        return (sum(v["wins"] for v in summary.values()),
                sum(v["games"] for v in summary.values()))

    confirm = ROOT / "experiments/results/r1-vs-g275-handoff12-64seed-v1.json"
    assert totals(confirm, "baseline") == (531, 896), totals(confirm, "baseline")
    assert totals(confirm, "candidate") == (683, 896), totals(confirm, "candidate")
    assert deployment["opening"] == "G275", deployment["opening"]
    # Pure state trigger: the handoff waits for the third quadrant, with
    # handoff_step as the deadline for routes that never reach it.
    assert deployment.get("handoff_land") == 3
    assert not deployment.get("handoff_floor"), deployment.get("handoff_floor")
    lib = ctypes.CDLL(str(ROOT / "policy/r1/agent.so"))
    assert lib.td_observe_external and lib.td_activate_external
    from meta_agent.src.native_teammate_executor import NativeTeammateBundle
    bundle = NativeTeammateBundle(ROOT / "agent/teammate_base.py",
                                  ROOT / "agent/route_actions.json.zlib",
                                  ROOT / "agent/route_library.json")
    assert len(bundle.families) == 245
    print(json.dumps({"status": "PASS", "replays": 609, "routes": 245,
                      "strong_opponents": 7, "switch_games": fine["games"]}))

if __name__ == "__main__":
    main()
