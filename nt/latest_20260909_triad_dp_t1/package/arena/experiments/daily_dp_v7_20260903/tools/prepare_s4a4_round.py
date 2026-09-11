"""Freeze S4A-4 business-value schedule comparison and causal combinations."""
from pathlib import Path
import argparse, hashlib, json, shutil

EXP = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="s4a4")
    a = p.parse_args()
    if not a.profile.replace("_", "").isalnum():
        raise ValueError("profile name")
    out = EXP / "profiles" / a.profile
    out.mkdir(exist_ok=False)
    base = json.loads((EXP / "profiles/s3w/configs.json").read_text())["old"]

    def cfg(shared=False, step=False, prep=False, value=False):
        return dict(
            base,
            shared_task_atoms_v2=shared,
            stepwise_recoordination=step,
            preparation_pipeline_v2=prep,
            schedule_value_compare=value,
        )

    configs = {
        "old": cfg(),
        "value_only": cfg(value=True),
        "shared_value": cfg(shared=True, value=True),
        "shared_step_value": cfg(shared=True, step=True, value=True),
        "prep_value": cfg(prep=True, value=True),
        "all_four": cfg(shared=True, step=True, prep=True, value=True),
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
    receipt = dict(
        status="FROZEN_BEFORE_STRENGTH",
        config_sha256=sha(path), source_hashes=hashes,
        binary_sha256=build["binary_sha256"],
        development_N=[20262701, 20262750], confirmation_O=[20262801, 20262850],
        unseen_P=[20262901, 20262950], final_holdout_used=False, oracle_used=False,
        note="Four legal same-day schedules are ranked by authoritative completed task semantics; sort metadata and earlier equal effects cannot inflate value. No simulator future or opponent identity is queried.",
    )
    (out / "freeze.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps({"status": "FROZEN", "configs": list(configs), "config_sha256": sha(path), "binary_sha256": build["binary_sha256"]}))


if __name__ == "__main__":
    main()
