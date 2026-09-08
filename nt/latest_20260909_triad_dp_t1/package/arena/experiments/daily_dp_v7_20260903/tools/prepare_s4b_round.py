"""Freeze feed-reserve and preparation-projection ablations before strength."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

EXP = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="s4b1")
    a = p.parse_args()
    if not a.profile.replace("_", "").isalnum():
        raise ValueError("profile name")
    out = EXP / "profiles" / a.profile
    out.mkdir(exist_ok=False)
    base = json.loads((EXP / "profiles/s4a4c/configs.json").read_text())["old"]

    def cfg(net=False, shared=False, step=False, prep=False, value=False, guard=False):
        return dict(base, net_feed_buffer=net, shared_task_atoms_v2=shared,
                    stepwise_recoordination=step, preparation_pipeline_v2=prep,
                    schedule_value_compare=value, preparation_spawn_guard=guard)

    configs = {
        "old": cfg(),
        "net_feed": cfg(net=True),
        "net_shared_step": cfg(net=True, shared=True, step=True),
        "prep_guard": cfg(prep=True, guard=True),
        "all_guard": cfg(shared=True, step=True, prep=True, value=True, guard=True),
        "all_guard_net": cfg(net=True, shared=True, step=True, prep=True, value=True, guard=True),
    }
    path = out / "configs.json"
    path.write_text(json.dumps(configs, indent=2), encoding="utf8")
    source = out / "source"
    source.mkdir()
    hashes = {}
    for name in ("policy.hpp", "module.cpp", "test_policy.cpp"):
        src = EXP / "native" / name
        shutil.copy2(src, source / name)
        hashes[src.relative_to(EXP).as_posix()] = sha(src)
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    (out / "freeze.json").write_text(json.dumps(dict(
        status="FROZEN_BEFORE_STRENGTH", config_sha256=sha(path), source_hashes=hashes,
        binary_sha256=build["binary_sha256"], development_N=[20262701, 20262750],
        confirmation_O=[20262801, 20262850], unseen_P=[20262901, 20262950],
        oracle_used=False, final_holdout_used=False,
        hypotheses=["Standing wheat reduces only future surplus feed commitment, not current feed.",
                    "A preparation prefix must not invalidate the HIRE spawn used to plan it."],
    ), indent=2), encoding="utf8")
    print(json.dumps(dict(status="FROZEN", configs=list(configs), binary_sha256=build["binary_sha256"])))


if __name__ == "__main__":
    main()
